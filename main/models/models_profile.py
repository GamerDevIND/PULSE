from .ollama_models import OllamaModel, DOWN, BUSY, SHUTTING_DOWN, IDLE, OllamaEmbedder
from main.utils import Logger
from main.resource_manager import SessionManager

class RemoteModel(OllamaModel):
    def __init__(self, role: str, name: str, model_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, port, system_prompt: str, api_key: None | str = None) -> None:
        super().__init__(role,name,model_name, has_tools, has_CoT, has_vision, port, system_prompt, api_key)
        self.resource_manager = SessionManager(self.model_name)
        
    async def warm_up(
        self,
        use_mmap=False,
        warmup_image_path="main/test.jpg",
        use_custom_keep_alive_timeout=False,
        custom_keep_alive_timeout: str = "5m",
        has_video_processing=False,
        warmup_video_path="main/test.mp4"):

        await self._warmer(use_mmap=use_mmap, custom_keep_alive_timeout="-1", use_custom_keep_alive_timeout=True, has_video_processing=has_video_processing, 
                           warmup_image_path=warmup_image_path,warmup_video_path=warmup_video_path)
        if self.warmed_up:
            await self.change_state(IDLE)

    async def generate(self, query: str | None, context: list[dict], stream: bool, think: str | bool | None = False, file_path: None | str = None, 
                   mod_ = 10, system_prompt_override: str | None = None, options: dict | None = None, format_: dict | None = None,  tools_override:None | list = None):
        await Logger.log_async(f"Generating response from {self.name}...", "info")
        await self.change_state(BUSY)
        async for chunk in self._generator(query, context, stream, think, file_path, mod_, 
                                           system_prompt_override=system_prompt_override, options = options, format_ = format_, tools_override=tools_override):
            yield chunk
        await self.change_state(IDLE)

    async def shutdown(self):
        await Logger.log_async(f"Shutting down {self.name}...", "info")
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
        if self.warmed_up:
            await self.resource_manager.shutdown(True, url, headers, payload, True,)
            
        await Logger.log_async(F"{self.name} shutting down success", 'info')
        await self.change_state(DOWN)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Model shutdown: {self.name} ({self.model_name})")


class RemoteEmbedder(OllamaEmbedder):
    def __init__(self, role, name: str, model_name: str, port:int,  **kwargs) -> None:
        super().__init__(role, name, model_name, port)
        self.resource_manager = SessionManager(self.model_name)

    async def warm_up(self):
        await self._warmer()
        if self.warmed_up:
            await self.change_state(IDLE)

    async def shutdown(self):
        await Logger.log_async(f"Shutting down {self.name}...", "info")
        await self.change_state(SHUTTING_DOWN)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Shutting down model: {self.name} ({self.model_name})")
        url = f'{self.host}{self._get_endpoint()}'
        headers = {"Content-Type": "application/json"}

        if self.api_key: headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {
            'model': self.model_name,
            "input": "bye",
            'options': {'keep_alive': 0}
        }
        await self.resource_manager.shutdown(True, url, headers, payload, True,)
        await Logger.log_async(F"{self.name} shutting down success", 'info')
        await self.change_state(DOWN)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Model shutdown: {self.name} ({self.model_name})")