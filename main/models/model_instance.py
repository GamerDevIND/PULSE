import os
import aiohttp
from .ollama_models import OllamaModel, IDLE, BUSY, SHUTTING_DOWN, DOWN, OllamaEmbedder
from main.utils import log
from main.resource_manager import ResourceManager

class LocalModel(OllamaModel):
    def __init__(self, role: str, name: str, model_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, port: int, system_prompt: str, api_key: None | str = None):
        super().__init__(role, name, model_name, has_tools, has_CoT, has_vision, port, system_prompt, api_key)
        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = self.host
        self.warmed_up = False
        self.use_custom_keep_alive_timeout = True
        self.custom_keep_alive_timeout = "-1"
        self.use_mmap = False
        self.resource_manager:ResourceManager = ResourceManager(self.model_name)

    async def warm_up(
        self,
        use_mmap=False,
        warmup_image_path="main/test.jpg",
        use_custom_keep_alive_timeout=True,
        custom_keep_alive_timeout: str = "-1",
        has_video_processing=False,
        warmup_video_path="main/test.mp4"):

        self.resource_manager.create_process(self.start_command, self.ollama_env)

        is_actually_alive = False
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=2)
            ) as session:
                async with session.get(f"{self.host}/api/tags") as res:
                    if res.status == 200:
                        js = await res.json(encoding='utf-8')
                        downloaded_models = [m['name'] for m in js.get("models", [])]
                        if self.model_name in downloaded_models:
                            is_actually_alive = True
                        else:
                            await log(f"Server alive on {self.port}, but {self.model_name} not found in library.", "warn")
        except Exception as e:
            await log(f"An error occured while checking port: {repr(e)}", 'error')

        if is_actually_alive:
            self.resource_manager.create_session()

            self.warmed_up = True
            await self.change_state(IDLE)
            await log(f"{self.name} is already alive on {self.host}. Re-linked.", "success")
            return

        await self._warmer(use_mmap, warmup_image_path, use_custom_keep_alive_timeout, custom_keep_alive_timeout,
                           has_video_processing, warmup_video_path)

    async def generate(self, query: str | None, context: list[dict], stream: bool, think: str | bool | None = False, file_path: None | str = None, 
                   mod_ = 10, system_prompt_override: str | None = None, options:dict | None = None, format_: dict | None = None, tools_override:None | list = None):

        await self.change_state(BUSY) 
        self.generation_cancelled = False

        await log(f"Generating response from {self.name}...", "info")
        
        if options:
            if self.use_mmap:
                options["use_mmap"] = True
            if self.use_custom_keep_alive_timeout:
                options["keep_alive"] = self.custom_keep_alive_timeout

        try:
            async for chunk in self._generator(query, context, stream, think, 
                                                                              file_path, mod_, system_prompt_override=system_prompt_override, 
                                                                              options=options, format_=format_, tools_override=tools_override):
                yield chunk

        finally:
            async with self.state_lock:
                self.generation_cancelled = False
                if self.state == BUSY:
                    await self.change_state(IDLE, False)

    async def shutdown(self):
        await log(f"Shutting down {self.name}...", "info")

        async with self.state_lock:
            if self.state in (DOWN, SHUTTING_DOWN):
                return
        await self.change_state(SHUTTING_DOWN)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Shutting down model: {self.name} ({self.model_name})")

        url = f'{self.host}{self._get_endpoint()}'
        headers = {"Content-Type": "application/json"}

        if self.api_key: headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': 'bye'}],
            'options': {'keep_alive': 0}
        }

        await self.resource_manager.shutdown(True, url, headers, payload, True, True)

        await self.change_state(DOWN)
        await log(f"{self.name} shutdown complete.", "success")
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Model shutdown: {self.name} ({self.model_name})")



class LocalEmbedder(OllamaEmbedder):
    def __init__(self, role, name: str, model_name: str, port:int, **kwargs) -> None:
        super().__init__(role, name, model_name, port)
        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = self.host
        self.warmed_up = False
        self.resource_manager:ResourceManager = ResourceManager(self.model_name)

    async def warm_up(self):
        self.resource_manager.create_process(self.start_command, self.ollama_env)

        try:
            await self.resource_manager.wait_until_ready(f"{self.host}", 30)
            is_actually_alive = True
        except Exception as e:
            await log(f"An error occured while checking port: {repr(e)}", 'error')
            is_actually_alive = False

        if is_actually_alive:
            self.resource_manager.create_session()
            self.warmed_up = True
            await self.change_state(IDLE)
            await log(f"{self.name} is already alive on {self.host}. Re-linked.", "success")
            return
        
        await self._warmer()

    async def shutdown(self):
        await log(f"Shutting down {self.name}...", "info")

        async with self.state_lock:
            if self.state in (DOWN, SHUTTING_DOWN):
                return
        await self.change_state(SHUTTING_DOWN)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Shutting down model: {self.name} ({self.model_name})")

        url = f'{self.host}{self._get_endpoint()}'
        headers = {"Content-Type": "application/json"}

        if self.api_key: headers['Authorization'] = f'Bearer {self.api_key}'

        payload = {
            'model': self.model_name,
            "input":'bye',
            'options': {'keep_alive': 0}
        }

        await self.resource_manager.shutdown(True, url, headers, payload, True, True, set_session_to_None = True) 

        await self.change_state(DOWN)
        await log(f"{self.name} shutdown complete.", "success")
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Model shutdown: {self.name} ({self.model_name})")