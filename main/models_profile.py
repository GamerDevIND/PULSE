import aiohttp
from typing import AsyncGenerator
from .models import Model, WARMING_UP, DOWN, BUSY, SHUTTING_DOWN, IDLE
from .utils import log

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
        self.router_role = "router"
        self.system = system_prompt
        self.has_video = False
        self.available_roles = [] # ONLY accessed by the router model
        self.tools = []

    async def warm_up(
        self,
        use_mmap=False,
        warmup_image_path="main/test.jpg",
        use_custom_keep_alive_timeout=False,
        custom_keep_alive_timeout: str = "5m",
        has_video_processing=False,
        warmup_video_path="main/test.mp4", router_role = "router"):

        self.router_role = router_role
        self.has_video = has_video_processing

        await self.change_state(WARMING_UP)

        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            await self.wait_until_ready(f"http://localhost:{self.port}/api/tags")

            async with self.session.get(f'{self.host}/api/tags') as response:
                    response.raise_for_status()
                    if response.status == 200:
                        self.warmed_up = True
        finally:
            await self.change_state(IDLE if self.warmed_up else DOWN)

    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                   mod_ = 10, system_prompt_override: str | None = None)  -> AsyncGenerator:
        await log(f"Generating response from {self.name}...", "info")
        await self.change_state(BUSY)
        async for chunk in self._generator(query, context, stream, think, image_path, mod_, system_prompt_override):
            yield chunk
        await self.change_state(IDLE)

    async def shutdown(self):
        await log(f"Shutting down {self.name}...", "info")
        await self.change_state(SHUTTING_DOWN)
        if self.session:
           await self.session.close()
        await self.change_state(DOWN)