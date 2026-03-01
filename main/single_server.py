import json
import aiofiles
import asyncio
from .tools import Tool
from .utils import log
from .models_profile import RemoteModel as Model
from .backend import Backend
import aiohttp
import subprocess 
import sys
import os
import psutil
if sys.platform != 'win32':
    from signal import SIGKILL
else:
    SIGKILL = 9

class SingleServer(Backend):
    def __init__(self, models_list_path:str, system_prompts:dict[str, str], default_system_prompt, ollama_port = None) -> None:
        self.models_list_path = models_list_path
        self.system_prompts = system_prompts
        self.default_system_prompt = default_system_prompt

        if ollama_port is None:
            try:
                with open(models_list_path, 'r', encoding='utf-8') as f:
                    models_data = json.load(f)
                    if models_data and isinstance(models_data, list):
                        ollama_port = models_data[0].get('port', 11434)
            except Exception:
                ollama_port = 11434 
        self.ollama_port = ollama_port
        self.models: dict[str, Model] = {}
        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = f"http://localhost:{self.ollama_port}"
        self.ollama_env["OLLAMA_MAX_LOADED_MODELS"] = "15"
        self.ollama_env["OLLAMA_NUM_PARALLEL"] = "15"
        self.process = None
        self.running_tasks = set()
        
        self.active_model:Model | None = None
        self.generation_task = None

    def load(self, auto_resolve_ports=True):
        try:
            with open(self.models_list_path, 'r', encoding="utf-8") as f:
                models_data = json.load(f)
                for model_data in models_data:
                    role = model_data.get('role')
                    port = model_data.get('port', self.ollama_port)
                    
                    if port != self.ollama_port:
                        if auto_resolve_ports:
                            model_data['port'] = self.ollama_port
                        else:
                            raise Exception(f"Port mismatch for {role}")

                    if role:
                        system = model_data.get("system_prompt")
                        if not (system and system.strip()):
                            model_data["system_prompt"] = self.system_prompts.get(role, self.default_system_prompt)
                        
                        self.models[role] = Model(**model_data)
    
                        self.models[role].host = f"http://localhost:{self.ollama_port}"
                        
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            raise Exception("Models loading failed.", e)
    
    async def async_load(self, auto_resolve_ports = True):
        try:
            async with aiofiles.open(self.models_list_path, 'r', encoding="utf-8") as f:
                c = await f.read()
                models_data =  json.loads(c)
                if not isinstance(models_data, list): raise ValueError("The config file must contain a list of the configs.")
                for model_data in models_data:
                    role = model_data.get('role')
                    port = model_data.get('port', self.ollama_port)
                    
                    if port != self.ollama_port:
                        if auto_resolve_ports:
                            model_data['port'] = self.ollama_port
                        else:
                            raise Exception(f"Port mismatch for {role}")

                    if role:
                        system = model_data.get("system_prompt")
                        if not (system and system.strip()):
                            model_data["system_prompt"] = self.system_prompts.get(role, self.default_system_prompt)
                        
                        self.models[role] = Model(**model_data)
    
                        self.models[role].host = f"http://localhost:{self.ollama_port}"
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            raise Exception("Models loading failed.", e)

    async def init(self, tools_list:list[Tool]):    
        await log("Initiating...", "info")
        if all(m.warmed_up for m in self.models.values()) and self.process and self.process.poll() is None:
            await log(
                    f"Backend process already running and warmed. Skipping restart.",
                    "success",
            )
        else: 
            log_dir = os.path.join("main", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f"single_server.log")

            f = open(log_file_path, "w")
            self._log_file = f

            creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            start_new_session = sys.platform != "win32"

            self.process = subprocess.Popen(
                self.start_command,
                env=self.ollama_env,
                stderr=subprocess.STDOUT,
                stdout=f,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
        await log("Loading all models...", 'info')

        for model in self.models.values():
            if model.has_tools:
                await model.add_tools(*tools_list)
            if not model.warmed_up:
                await model.warm_up()
            else:
                await log(f"{model.name} ({model.ollama_name}) is already warmed, skipping... This maybe abnormal, please ensure the initilising logic.", 'warn')
    
    async def shutdown(self):
        await log(f"Shutting down Single Server...", "info")
        for m in self.models.values():
            await m.shutdown()

        if self.process is not None:
            pid = self.process.pid
            await log(f"Cleaning up process tree for Single Server (PID {pid})...", "info")

            try:
                parent = psutil.Process(pid)

                children = parent.children(recursive=True)
                all_procs = children + [parent]

                for p in all_procs:
                    try:
                        p.terminate()
                    except psutil.NoSuchProcess:
                        pass

                gone, alive = await asyncio.to_thread(psutil.wait_procs, all_procs, timeout=10)

                if alive:
                    await log(f"{len(alive)} processes resisted termination. Escalating to SIGKILL...", "warning")
                    for p in alive:
                        try:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                            
            except psutil.NoSuchProcess:
                await log(f"Process {pid} already terminated.", "debug")
            except Exception as e:
                await log(f"Error during paranoid shutdown of Single Server: {e}", "error")
            finally:
                self.process = None

        if hasattr(self, '_log_file') and self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        await log(f"Single Server shutdown complete.", "success")