import os
import asyncio
import aiohttp
import aiofiles
import av
import json
from typing import Literal, Type
from .utils import log
from .tools import Tool
from .configs import IMAGE_EXTs, VIDEO_EXTs, ERROR_TOKEN
import base64
from io import BytesIO
import sys
if sys.platform != 'win32':
    from signal import SIGKILL
else:
    SIGKILL = 9

import inspect

IDLE = "idle"
BUSY = "busy"
SHUTTING_DOWN = "shutting_down"
DOWN = "down"
WARMING_UP = "warming_up"

class Model:
    def __init__(self, role: str, name: str, ollama_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, port: int, system_prompt: str):
        self.role = role
        self.name = name
        self.ollama_name = ollama_name
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.has_vision = has_vision
        self.router_role = "router"
        self.port = port
        self.host =  "localhost:11434"
        self.system = system_prompt
        self.warmed_up = False
        self.has_video = False
        self.available_roles = [] # ONLY accessed by the router model
        self.tools = []
        self.generation_cancelled = False
        self.session: aiohttp.ClientSession | None = None
        self.state_lock = asyncio.Lock()
        self.state = DOWN

    async def __aenter__(self):
        await self.warm_up()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    def _get_endpoint(self) -> str:
        return "/api/chat"

    async def change_state(self, new, use_lock = True):
        if use_lock:
            async with self.state_lock: 
                self.state = new
        else:
            self.state = new

    async def wait_until_ready(self, url: str, timeout: int = 30):
        await log(f"Waiting for {self.name} on {url}...", "info")
        for i in range(timeout):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(f"{url}/api/tags") as res:
                        if res.status == 200:
                            await log(f"{self.name} is ready!", "success")
                            return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                await log(f"Retries: {i+1} / {timeout}", "info")
            await asyncio.sleep(1)
        raise TimeoutError(f"🟥 Ollama server for {self.name} did not start in time.")

    async def init(self): 
        raise NotADirectoryError

    async def _encode_frames_from_vid(self, video_path, mod_ = 1, format = "JPEG"):
        if not self.has_vision:
            return [], None
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in VIDEO_EXTs:
            await log(f"{video_path} has file extenstion {ext} which isn't supported for video input", 'error')
            return ERROR_TOKEN, f"{video_path} has file extenstion {ext} which isn't supported for video input"
        
        try:
            container = av.open(video_path)
            frames = []
            await log("Audio processing hasn't been implemented yet.", 'warn')
            video_stream = next((s for s in container.streams if s.type == 'video'), None)
            if not video_stream:
                raise ValueError("No video stream found in the file.")
            for i, frame in enumerate(container.decode(video_stream)): # type: ignore
                if i % mod_ == 0:
                    img = frame.to_image() # type: ignore
                    buffer = BytesIO()
                    img.save(buffer, format=format)
                    frame_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    frames.append(frame_b64)
            return frames, None
        except Exception as e:
            await log(f"Error in video frame encoding for {self.name}: {e}", 'error')
            return ERROR_TOKEN, repr(e)

    async def _encode_image(self, image_path):
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

    def add_to_available_roles(self, *roles):
        roles = list(roles)
        self.available_roles.extend(roles)
        self.available_roles = list(set(self.available_roles)) # order doesn't matter 

    async def warm_up(
        self,
        use_mmap=False,
        warmup_image_path="main/test.jpg",
        use_custom_keep_alive_timeout=False,
        custom_keep_alive_timeout: str = "5m",
        has_video_processing=False,
        warmup_video_path="main/test.mp4", router_role = "router"): 
        raise NotImplementedError

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

    def cancel(self):
        self.generation_cancelled = True
    
    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                   mod_ = 1, system_prompt_override: str | None = None):

        raise NotImplementedError

    async def shutdown(self):
        raise NotImplementedError

    async def _generator(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                   mod_ = 10, system_prompt_override: str | None = None, options : dict | None = None) :
        endpoint = self._get_endpoint()
        url = f"{self.host}{endpoint}"

        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()

        messages = []
        if system_prompt_override is None:
            if self.system:
                messages.append({'role': "system", 'content': self.system})
        else:
            messages.append({'role': "system", 'content': system_prompt_override})

        if query and query.strip(): messages += context + [{"role": "user", "content": query}]

        headers = {"Content-Type": "application/json"}

        data = {
            "model": self.ollama_name,
            "messages": messages,
            "stream": stream
        }

        if self.role.lower() == self.router_role:
            if len(self.available_roles) < 1:
                await log("No role provided", "error") 
                raise Exception("No role provided")

            data["format"] = {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": self.available_roles, "description": "Selected role or model"}
                },
                "required": ["role"]
                }
        elif self.has_tools:
            data["tools"] = self.tools
    
        if options:
            data["options"] = options

        if self.has_CoT and think is not None:
            data['think'] = think

        if self.has_vision and image_path is not None:
            ext = os.path.splitext(image_path)[1].lower()
            if ext in IMAGE_EXTs:
                image, e = await self._encode_image(image_path)
                if image == ERROR_TOKEN:
                    await log(f"Error encoding image!: {repr(e)}", 'error')
                    yield (ERROR_TOKEN, ERROR_TOKEN, [])
                    return
                else:
                    data['messages'][-1]['images'] = [image]
            elif ext in VIDEO_EXTs and self.has_video:
                frames, e = await self._encode_frames_from_vid(image_path, mod_)
                if frames == ERROR_TOKEN:
                    await log(f"Error encoding video!: {repr(e)}", 'error')
                    yield (ERROR_TOKEN, ERROR_TOKEN, [])
                    return
                else:
                    data['messages'][-1]['images'] = frames
            elif ext in VIDEO_EXTs and not self.has_video:
                await log(f"Cannot process video file {image_path}: Model {self.name} is not configured for video processing.", 'warn')

        buffer = ""
        try:
            timeout = aiohttp.ClientTimeout(total=None)
            async with self.session.post(url, headers=headers, data=json.dumps(data), timeout=timeout) as response:
                response.raise_for_status()

                if stream:
                    try:
                        async for raw_chunk in response.content.iter_chunked(1024):
                           
                            if self.generation_cancelled:
                                await log(f"Cancellation requested for {self.name}, breaking stream", "info")
                                break
                            
                            chunk = raw_chunk.decode("utf-8", errors='ignore')
                            buffer += chunk
                            
                            while "\n" in buffer and not self.generation_cancelled:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                
                                try:
                                    json_line = json.loads(line)
                                    thinking_chunk = json_line.get("message", {}).get("thinking", "")
                                    content_chunk = json_line.get("message", {}).get("content", "")
                                    tools_chunk = json_line.get("message", {}).get("tool_calls", []) 
                                    yield (thinking_chunk, content_chunk, tools_chunk)
                                except json.JSONDecodeError:
                                    continue

                            if self.generation_cancelled:
                                await response.release()
                                break
                                
                    except asyncio.CancelledError:
                        await log(f"Generation cancelled for {self.name}", "info")
                        return 
                        
                else:
                    try:
                        res_json = await response.json()
                        thinking = res_json.get("message", {}).get("thinking", "")
                        content = res_json.get("message", {}).get("content", "")
                        tools = res_json.get("message", {}).get("tool_calls", [])
                        yield (thinking, content, tools)
                    except asyncio.CancelledError:
                        await log(f"Non-stream generation cancelled for {self.name}", "info")
                        return 

        except asyncio.CancelledError:
            await log(f"Request cancelled for {self.name}", "info")
            yield (ERROR_TOKEN, ERROR_TOKEN, [])
            return
            
        except Exception as e:
            await log(f"Ollama API Request Error: {e}", "error")
            yield (ERROR_TOKEN, ERROR_TOKEN, [])