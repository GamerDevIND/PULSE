import json
from main.tools import Tool
from main.utils import log
from main.models.models_profile import RemoteModel as Model, OllamaEmbedder as Embedder
from .backend import Backend
from main.resource_manager import ResourceManager
import os

class SingleServer(Backend):
    def __init__(self, models_list_path:str, system_prompts:dict[str, str], default_system_prompt, ollama_port:None | int = None) -> None:
        super().__init__(models_list_path, system_prompts, default_system_prompt)

        if ollama_port is None:
            try:
                with open(models_list_path, 'r', encoding='utf-8') as f:
                    models_data = json.load(f)
                    if models_data and isinstance(models_data, list):
                        ollama_port = models_data[0].get('port', 11434)
            except Exception:
                ollama_port = 11434 
        self.ollama_port = ollama_port
        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = f"http://localhost:{self.ollama_port}"

        
        self.resource_manager = ResourceManager("Single Server")
        
        self.generation_task = None

    def load(self, auto_resolve_ports=True):
        self.ollama_port = self.ollama_port or 11434
        self._load(Model, Embedder, True, self.ollama_port, auto_resolve_ports)
    
    async def async_load(self, auto_resolve_ports = True):
        self.ollama_port = self.ollama_port or 11434
        await self._async_load(Model, Embedder, True, self.ollama_port, auto_resolve_ports)

    async def init(self, tools_list:list[Tool]):    
        await log("Initiating...", "info")
        if all(m.warmed_up for m in self.models.values()) and self.resource_manager.process and self.resource_manager.process.poll() is None:
            await log(
                    f"Backend process already running and warmed. Skipping restart.",
                    "success",
            )
        else: 
            self.resource_manager.create_process(self.start_command, self.ollama_env)

        await log("Loading all models...", 'info')

        for model in self.models.values():
            if isinstance(model, Model):
                if model.has_tools:
                    await model.add_tools(*tools_list)
                if not model.warmed_up:
                    await model.warm_up()
                else:
                    await log(f"{model.name} ({model.model_name}) is already warmed, skipping... This maybe abnormal, please ensure the initilising logic.", 'warn')

        self._init()
    
    async def shutdown(self):
        await log(f"Shutting down Single Server...", "info")
        for m in self.models.values():
            await m.shutdown()

        await self.close_sessions()

        await self.resource_manager.shutdown(process_cleanup=True, set_session_to_None=True)

        await log(f"Single Server shutdown complete.", "success")