import aiohttp
from .models import Model, DOWN, BUSY, SHUTTING_DOWN, IDLE
from .utils import log
import asyncio

class RemoteModel(Model):
    def __init__(self, role: str, name: str, ollama_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, port, system_prompt: str) -> None:
        super().__init__(role,name,ollama_name, has_tools, has_CoT, has_vision, port, system_prompt)
        self.role = role
        self.name = name
        self.ollama_name = ollama_name
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.has_vision = has_vision
        self.session: aiohttp.ClientSession | None = None
        self.port = port
        self.host = f"http://localhost:{self.port}"
        self.system = system_prompt
        self.has_video = False
        self.tools = []

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
        # Ensure state is IDLE after warmup
        if self.warmed_up:
            await self.change_state(IDLE)

    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                   mod_ = 10, system_prompt_override: str | None = None, options: dict | None = None, format: dict | None = None):
        await log(f"Generating response from {self.name}...", "info")
        await self.change_state(BUSY)
        async for chunk in self._generator(query, context, stream, think, image_path, mod_, system_prompt_override=system_prompt_override, options = options, format = format):
            yield chunk
        await self.change_state(IDLE)

    async def shutdown(self):
        await log(f"Shutting down {self.name}...", "info")
        await self.change_state(SHUTTING_DOWN)
        if self.session is not None and self.warmed_up:
            try:
                url = f'{self.host}{self._get_endpoint()}'
                payload = {
                    'model': self.ollama_name,
                    'messages': [{'role': 'user', 'content': 'bye'}],
                    'options': {'keep_alive': 0}
                }
                if self.session:
                    async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        resp.raise_for_status()
            except Exception:
                pass 

            await asyncio.sleep(0.5) 
            try:
                if self.session:
                    await self.session.close()
                    await asyncio.sleep(0.5)  
            except Exception:
                pass
            self.session = None
        await log(F"{self.name} shutting down success", 'info')
        await self.change_state(DOWN)