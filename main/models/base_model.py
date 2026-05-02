import asyncio
from main.resource_manager import SessionManager, ResourceManager
from main.events import EventBus
from main.tools import Tool
import inspect
from typing import Literal, Type
from main.utils import log
import os
from io import BytesIO
import base64
import aiofiles
import av
from main.configs import IMAGE_EXTs, VIDEO_EXTs, ERROR_TOKEN

class Model:
    def __init__(self, role, host, name: str, model_name: str, api_key:str | None, init_state, event_bus: None | EventBus = None, **kwargs) -> None:
        '''
        `resource_manager` attribute defaults to `SessionManager`
        '''
        self.role = role
        self.name = name
        self.model_name = model_name
        self.host = host
        self.warmed_up = False
        self.state_lock = asyncio.Lock()
        self.tools = []
        self.has_video = False
        self.has_audio = False
        self.state = init_state
        self.api_key = api_key
        self.has_vision = kwargs.get('has_vision', False)
        self.port:int | None = None
        self.input_handler = InputHandler()
        self.event_bus = event_bus
        self.details_cache = None
        self.resource_manager: ResourceManager | SessionManager = SessionManager(model_name)

    async def __aenter__(self):
        await self.warm_up()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def warm_up(
        self,
        use_mmap=False,
        warmup_image_path="main/test.jpg",
        use_custom_keep_alive_timeout=True,
        custom_keep_alive_timeout: str = "-1",
        has_video_processing=False,
        warmup_video_path="main/test.mp4"): 
        raise NotImplementedError
    
    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                   mod_ = 1, system_prompt_override: str | None = None):

        raise NotImplementedError

    async def shutdown(self):
        raise NotImplementedError
    
    async def clear_details_cache(self):
        async with self.state_lock: self.details_cache = None

    async def change_state(self, new, use_lock = True):
        if use_lock:
            async with self.state_lock: 
                self.state = new
        else:
            self.state = new

    def build_payload(self, query, context, stream, system, system_prompt_override, tools = None):
        messages = []
        if system_prompt_override is None:
            if system:
                messages.append({'role': "system", 'content': system})
        else:
            messages.append({'role': "system", 'content': system_prompt_override})

        if query and query.strip(): messages += context + [{"role": "user", "content": query}]

        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }

        if tools:
            data["tools"] = tools

        return data

    async def _get_model_details_helper(self, res:dict, default_capability = "completion", auto_add_completion = True):
        capabilities:list = res.get("capabilities", [default_capability])
        inp = res.get("input_modalities", [])
        if "image" in inp:
            capabilities.append("vision")
        if auto_add_completion and ('text' in inp and not 'completion' in capabilities):
            capabilities.append("completion")

        modified_at = res.get("modified_at", "")
        details = res.get("details", {})
        parent_model = details.get("parent_model", "")
        family = details.get("family", "")
        parameter_size = details.get("parameter_size", "")
        quantization_level = details.get("quantization_level", "")
        model_info = res.get("model_info", {})
        architecture = model_info.get("general.architecture", "")
        parameter_count = model_info.get("general.parameter_count", -1)

        out = {
            "capabilities": capabilities,
            "modified_at": modified_at,
            "parent_model": parent_model,
            "family": family,
            "parameter_size": parameter_size,
            "quantization_level" : quantization_level,
            "architecture": architecture,
            "parameter_count": parameter_count,
        }
        async with self.state_lock : self.details_cache = out
        return out
    
    async def add_tools(self, *tool_dicts):
        for tool in tool_dicts:
            if isinstance(tool, Tool):
                self.tools.append(tool.schema)
            elif isinstance(tool, dict):
                self.tools.append(tool)
            elif callable(tool) and not inspect.isclass(tool):
                needs_regeneration:bool = False
                retry: Literal["none", "always", ] = "none"
                max_retries: int = 0
                retry_on: tuple[Type[Exception], ...] | None = None
                visible_to_user: bool = True
                metadata = {
                    "needs_regeneration": needs_regeneration,
                    "retry": retry,
                    "max_retries": max_retries,
                    "retry_on": retry_on or (),
                    "visible_to_user": visible_to_user,
                }
                tool = Tool(tool, metadata)
                self.tools.append(tool.schema)
            else:
                await log(f"No valid type dectected for: {tool}", 'warn')

    def set_has_video(self, has_video:bool):
        self.has_video = has_video

    def set_has_audio(self, has_audio:bool):
        self.has_audio = has_audio

    

class InputHandler:
    def __init__(self) -> None:
        pass

    def is_url(self, url:str):
        url = url.lower()
        return url.startswith('http://') or url.startswith('https://') or url.startswith('www.')
    
    async def encode_frames_from_vid(self, video_path, url_valid = False, mod_ = 1, format_ = "JPEG", max_video_size_mbs = 5):
        if url_valid:
            if self.is_url(video_path):
                return [video_path], None
        
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in VIDEO_EXTs:
            await log(f"{video_path} has file extenstion {ext} which isn't supported for video input", 'error')
            return ERROR_TOKEN, f"{video_path} has file extenstion {ext} which isn't supported for video input"
        
        if await asyncio.to_thread(os.path.getsize, video_path) / 1024**2 <= max_video_size_mbs:
            async with aiofiles.open(video_path, "rb") as video_file:
                return base64.b64encode(await video_file.read()).decode('utf-8'), None
        
        await log("Audio processing hasn't been implemented yet.", 'warn')
        def _helper():
            container = av.open(video_path)
            frames = []
            video_stream = next((s for s in container.streams if s.type == 'video'), None)
            if not video_stream:
                raise ValueError("No video stream found in the file.")
            for i, frame in enumerate(container.decode(video_stream)): # type: ignore
                if i % mod_ == 0:
                    img = frame.to_image() # type: ignore
                    buffer = BytesIO()
                    img.save(buffer, format=format_)
                    frame_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    frames.append(frame_b64)
            return frames, None
            
        return await asyncio.to_thread(_helper)
    
    async def encode_image(self, image_path, url_valid = False):
        if url_valid:
            if self.is_url(image_path):
                return image_path, None
            
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in IMAGE_EXTs:
            await log(f"{image_path} has file extenstion {ext} which isn't supported for image input", 'error')
            return ERROR_TOKEN, f"{image_path} has file extenstion {ext} which isn't supported for image input"
        try:
            async with aiofiles.open(image_path, "rb") as image_file:
                image_data = await image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            return base64_image, None
        except Exception as e:
            return ERROR_TOKEN, repr(e)
        
    async def encode_audio(self, audio_path, url_valid = False):
        if url_valid:
            if self.is_url(audio_path):
                return audio_path, None
            
        try:
            async with aiofiles.open(audio_path, "rb") as f:
                audio_data = await f.read()
                base64_audio = base64.b64encode(audio_data).decode("utf-8")
                return base64_audio, None
        except Exception as e:
            return ERROR_TOKEN, repr(e)