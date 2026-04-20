from main.models.model_instance import LocalModel as Model, DOWN, SHUTTING_DOWN, LocalEmbedder as Embedder
from main.models.ollama_models import WARMING_UP
import aiohttp
import asyncio
from main.tools import Tool
from main.utils import log
from .backend import Backend
from main.events import EventBus

class MultiServer(Backend):
    def __init__(self, models_list_path:str, system_prompts:dict[str, str], default_system_prompt, event_bus: None | EventBus = None,) -> None:
        super().__init__(models_list_path, system_prompts, default_system_prompt, event_bus)
        self.checking_event = asyncio.Event()

    def load(self):
        self._load(Model, Embedder, False)

    async def async_load(self):
        await self._async_load(Model, Embedder, False)

    async def _ping_model_tag(self, url):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/api/tags") as res:
                    return res.status == 200
        except aiohttp.ClientError:
                return False

    async def _check_models(self, interval=10):
        cooldowns = {name: 0 for name in self.models}

        while not self.checking_event.is_set():
            for name, model in self.models.items():
                if model.state in [DOWN, SHUTTING_DOWN, WARMING_UP]:
                    continue

                if cooldowns[name] > 0:
                    cooldowns[name] -= 1
                    continue

                is_alive = await self._ping_model_tag(model.host)

                if not is_alive:
                    await log(f"CRITICAL: {model.name} API is down. Restarting...", "error")
                    cooldowns[name] = 5
                    await model.warm_up()

            await asyncio.sleep(interval)

    async def init(self, tools_list:list[Tool]):
        await log("Loading all models...", 'info')

        await self._init(*tools_list)

        check_task = asyncio.create_task(self._check_models())
        self.running_tasks.add(check_task)

    async def shutdown(self):
        self.checking_event.set()

        await self.close_sessions()

        for model in self.models.values():
            await model.shutdown()