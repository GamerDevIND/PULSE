from .model_instance import LocalModel
from .models_profile import RemoteModel
import asyncio
from .configs import ERROR_TOKEN
from .utils import log

class Backend:
    def __init__(self, models_list_path, system_prompts, default_system_prompt) -> None:
        self.models_list_path = models_list_path
        self.system_prompts = system_prompts
        self.default_system_prompt = default_system_prompt
        self.models:dict[str, LocalModel | RemoteModel] = {}
        self.active_model = None 
        self.generation_task = None

    async def cancel_generation(self):
        if self.active_model:
            self.active_model.cancel()
            self.active_model = None 
            if self.generation_task:
                if not self.generation_task.done():
                    self.generation_task.cancel()
                    try:
                        await self.generation_task
                    except asyncio.CancelledError:
                        pass
                    self.generation_task = None

    def get_model(self, role):
        model_obj = self.models.get(role)
        return model_obj

    async def generate(self, role:str, query:str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                       system_prompt_override=None, mod_ = 10, custom_active_model:LocalModel | RemoteModel | None = None, 
                       custom_generation_task: None | asyncio.Task = None):
        
        if image_path and role != "vision": 
            await log("Image / Video path provided but no vision model chosen. Role changing to Vision.", 'warn')
            role = "vision"
        
        if custom_active_model:
            model_obj = custom_active_model
        else:
            model_obj = self.models.get(role)
            self.active_model = model_obj

        if not model_obj:
            await log(f"{role} not found in the model registry!","error")
            yield (ERROR_TOKEN, ERROR_TOKEN, [])
            return
        
        if stream:
            queue = asyncio.Queue(maxsize=256)
            async def producer():
                try:
                    async for (thinking_chunk, content_chunk, tools_chunk) in model_obj.generate(query,context, True,
                                                                                                        think=think, image_path=image_path, mod_ = mod_, system_prompt_override=system_prompt_override):
                        if content_chunk == ERROR_TOKEN:
                            break
                        await queue.put((thinking_chunk, content_chunk, tools_chunk))
                except asyncio.CancelledError:
                    raise
                finally:
                    try:
                        queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
                    
            if not custom_generation_task:
                task = asyncio.create_task(producer())
                self.generation_task = task
            
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    thinking_chunk, content_chunk, tools_chunk = item

                    yield (thinking_chunk or "", content_chunk or "", tools_chunk)
            
                await task            
        else:
            await log(f"Non-streaming mode active", "info")
            async for (thinking, content, tools) in model_obj.generate(query, context, False, think=think, 
                                                                              image_path=image_path, mod_ = mod_, system_prompt_override=system_prompt_override):
                await log(f"Got non-streaming response chunk", "info")
                if content == ERROR_TOKEN:
                    break

                yield (thinking or "", content or "", tools)

        self.active_model = None 
        self.generation_task = None