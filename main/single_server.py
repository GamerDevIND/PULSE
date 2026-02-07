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
if sys.platform != 'win32':
    from signal import SIGKILL
else:
    SIGKILL = 9

class SingleServer(Backend):
    def __init__(self, models_list_path:str, system_prompts:dict[str, str], default_system_prompt, ollama_port = 11434) -> None:
        self.models_list_path = models_list_path
        self.system_prompts = system_prompts
        self.default_system_prompt = default_system_prompt
        self.ollama_port = ollama_port
        self.models: dict[str, Model] = {}
        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = f"http://localhost:{self.ollama_port}"
        self.process = None
        self.running_tasks = set()
        
        self.active_model:Model | None = None
        self.generation_task = None
        self.warmed_up = False

    def load(self, auto_resolve_ports = True):
        try:
            with open(self.models_list_path, 'r', encoding="utf-8") as f:
                models_data = json.load(f)
                for model_data in models_data:
                    role = model_data.get('role')
                    port = model_data.get('port', self.ollama_port)
                    if port != self.ollama_port:
                        message = (f"The provided configs doesn't have the set ollama port '({self.ollama_port})', it has port: '{port}'." 
                                   "Please ensure the configs has the same set port or no port key at all.")
                        if auto_resolve_ports:
                            print(message)
                            model_data['port'] = self.ollama_port
                        else:
                            raise Exception(message)
                    if role:
                        system = model_data.get("system_prompt")
                        if not (system and system.strip()): model_data["system_prompt"] = self.system_prompts.get(role, self.default_system_prompt)
                        self.models[role] = Model(**model_data)
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
                        message = (f"The provided configs doesn't have the set ollama port '({self.ollama_port})', it has port: '{port}'." 
                                   "Please ensure the configs has the same set port or no port key at all.")
                        if auto_resolve_ports:
                            await log(message, 'warn')
                            model_data['port'] = self.ollama_port
                        else:
                            raise Exception(message)
                    if role:
                        system = model_data.get("system_prompt")
                        if not (system and system.strip()): model_data["system_prompt"] = self.system_prompts.get(role, self.default_system_prompt)
                        self.models[role] = Model(**model_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            raise Exception("Models loading failed.", e)
        
    async def wait_until_ready(self, url: str, timeout: int = 30):
        await log(f"Waiting for SingleServer on port: {self.ollama_port} on {url}...", "info")
        for i in range(timeout):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(f"{url}/api/tags") as res:
                        if res.status == 200:
                            await log(f"SingleServer on port: {self.ollama_port} is ready!", "success")
                            return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                await log(f"Retries: {i+1} / {timeout}", "info")

                if self.process is not None and self.process.poll() is not None:
                    try:
                        if hasattr(self, '_log_file') and self._log_file is not None and hasattr(self._log_file, 'name'):
                            self._log_file.flush()
                            try:
                                async with aiofiles.open(self._log_file.name, 'r') as lf:
                                    content = await lf.read()
                                    tail = '\n'.join(content.splitlines()[-30:])
                                    await log(f"SingleServer on port: {self.ollama_port} process terminated. Recent logs:\n{tail}", 'error')
                            except (FileNotFoundError, PermissionError, OSError) as log_e:
                                await log(f"Could not read log file for SingleServer on port: {self.ollama_port}: {log_e}", 'error')
                    except (OSError, AttributeError) as proc_e:
                        await log(f"Error checking process status for SingleServer on port: {self.ollama_port}: {proc_e}", 'error')
            await asyncio.sleep(1)
        raise TimeoutError(f"🟥 Ollama server for SingleServer on port: {self.ollama_port} did not start in time.")

    async def init(self, tools_list:list[Tool]):    
        await log("Initiating...", "info")
        if self.warmed_up and self.process and self.process.poll() is None:
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

            await self.wait_until_ready(f"http://localhost:{self.ollama_port}/api/tags")
 
        await log("Loading all models...", 'info')

        for model in self.models.values():
            url = f"{model.host}{model._get_endpoint()}"
            payload = {
                'model':model.ollama_name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"keep_alive": -1},
            }
            try:
                if model.session:
                    async with model.session.post(url, json=payload) as response:
                        if response.status == 200:
                            print(f"Successfully sent preload request for model: {model.name} ({model.ollama_name})")
                            await response.text() 
                        else:
                            print(f"Failed to preload model {model.name} ({model.ollama_name}). Status: {response.status}")
                else:
                    await log(f"{model.name} ({model.ollama_name}) has no session. Skipping...", 'warn')
            except aiohttp.ClientConnectorError as e:
                print(f"Connection error while preloading {model.name} ({model.ollama_name}): {e}")
            except Exception as e:
                print(f"An error occurred while preloading {model.name} ({model.ollama_name}): {e}")

        await log("Attempting to warm up models...", 'info')

        for model in self.models.values():
            if model.has_tools:
                await model.add_tools(*tools_list)
            if not model.warmed_up:
                await model.warm_up()
            else:
                await log(f"{model.name} ({model.ollama_name}) is already warmed, skipping... This maybe abnormal, please ensure the initilising logic.", 'warn')
        
        self.warmed_up = True
    
    async def shutdown(self):
        await log("Shutting down all models...", 'info')
        for model in self.models.values():
            await model.shutdown()
        if hasattr(self, '_log_file') and self._log_file is not None:
            try:
                self._log_file.close()
            except:
                pass
            self._log_file = None
        if self.process is not None:
            pid = self.process.pid
            await log(f"Killing process tree for SingleServer on port: {self.ollama_port} (PID {pid})...", "info")

            try: # you can call me paranoid. I AM paranoid.
                self.process.terminate() 
                await asyncio.to_thread(self.process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                await log(f"Error terminating process for SingleServer on port: {self.ollama_port}: {e}", "error")

            try:
                if sys.platform == 'win32':
                    os.kill(pid, SIGKILL)
                else:
                    os.killpg(os.getpgid(pid), SIGKILL)
            except Exception:
                pass

            self.process = None
            await log("Shutdown succesful", "info")