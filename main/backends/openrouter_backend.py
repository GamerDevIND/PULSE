from main.utils import log
from .backend import Backend
from main.models.openrouter_model import OpenRouterModel as Model, OpenRouterEmbedder as Embedder
from main.tools import Tool
from main.events import EventBus

class OpenrouterBackend(Backend):
    def __init__(self, models_list_path:str, system_prompts:dict[str, str], default_system_prompt, event_bus: None | EventBus = None,) -> None:
        super().__init__(models_list_path, system_prompts, default_system_prompt, event_bus)
    
    def load(self):
        self._load(Model, Embedder, False, require_key=True)

    async def async_load(self):
        await self._async_load(Model, Embedder, False, require_key=True)

    async def init(self, tools_list:list[Tool]):
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
        await log(f"Shutting down Openrouter Backend... (You may ignore any network related issues but double checking is recommended)", "info")
        for m in self.models.values():
            await m.shutdown()

        await self.close_sessions()

        await log(f"Openrouter Backend shutdown complete.", "success")