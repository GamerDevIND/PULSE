from .model_instance import LocalModel as Model, DOWN, SHUTTING_DOWN, WARMING_UP
import json
import aiofiles
import aiohttp
import asyncio
from .tools import Tool
from .utils import log
from .backend import Backend

class MultiServer(Backend):
    def __init__(self, models_list_path:str, system_prompts:dict[str, str], default_system_prompt,) -> None:
        self.models_list_path = models_list_path
        self.system_prompts = system_prompts
        self.default_system_prompt = default_system_prompt
        self.models:dict[str, Model] = {}
        self.checking_event = asyncio.Event()
        self.running_tasks = set()
        self.active_model = None 
        self.generation_task = None

    def load(self):
        try:
            with open(self.models_list_path, 'r', encoding="utf-8") as f:
                models_data = json.load(f)
                for model_data in models_data:
                    role = model_data.get('role')
                    if role:
                        system = model_data.get("system_prompt")
                        if not (system and system.strip()): model_data["system_prompt"] = self.system_prompts.get(role, self.default_system_prompt)
                        self.models[role] = Model(**model_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            raise Exception("Models loading failed.", e)

    async def async_load(self):
        try:
            async with aiofiles.open(self.models_list_path, 'r', encoding="utf-8") as f:
                c = await f.read()
                models_data =  json.loads(c)
                if not isinstance(models_data, list): raise ValueError("The config file must contain a list of the configs.")
                for model_data in models_data:
                    role = model_data.get('role')
                    if role:
                        system = model_data.get("system_prompt")
                        if not (system and system.strip()): model_data["system_prompt"] = self.system_prompts.get(role, self.default_system_prompt)

                        self.models[role] = Model(**model_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            raise Exception("Models loading failed.", e)

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
        for model in self.models.values():
            if model.has_tools:
                await model.add_tools(*tools_list)
            if not model.warmed_up:
                await model.warm_up()
            else:
                await log(f"{model.name} ({model.ollama_name}) is already warmed, skipping... This maybe abnormal, please ensure the initilising logic.", 'warn')

        check_task = asyncio.create_task(self._check_models())
        self.running_tasks.add(check_task)

    async def shutdown(self):
        self.checking_event.set()
        for task in list(self.running_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for model in self.models.values():
            await model.shutdown()