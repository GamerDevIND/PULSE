import os
import asyncio
import aiohttp
import aiofiles
import av
import json
from main.utils import log
from main.configs import IMAGE_EXTs, VIDEO_EXTs, ERROR_TOKEN
import base64
from io import BytesIO
from .base_model import Model
import sys
if sys.platform != 'win32':
    from signal import SIGKILL
else:
    SIGKILL = 9

IDLE = "idle"
BUSY = "busy"
SHUTTING_DOWN = "shutting_down"
DOWN = "down"
WARMING_UP = "warming_up"

class OllamaModel(Model):
    def __init__(self, role: str, name: str, model_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, port: int, system_prompt: str, api_key: None | str = None):
        self.port = port
        self.host =  f"http://localhost:{self.port}"
        super().__init__(role, self.host, name, model_name, api_key, DOWN)       
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.has_vision = has_vision
        self.system = system_prompt
        self.has_video = False
        self.generation_cancelled = False

    def _get_endpoint(self) -> str:
        return "/api/chat"

    async def get_model_details(self):
        url = f"{self.host}/api/show"
        data = {
            "model": self.model_name
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key: headers['Authorization'] = f'Bearer {self.api_key}'

        if not self.resource_manager.session:
           self.resource_manager.create_session()
        
        if not self.resource_manager.session:
           raise
    
        async with self.resource_manager.session.post(url, headers=headers, data= json.dumps(data)) as resp:
            res = await resp.json()

        return await self._get_model_details_helper(res)

    async def _encode_frames_from_vid(self, video_path, mod_ = 1, format_ = "JPEG"):
        if not self.has_vision:
            return [], None
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in VIDEO_EXTs:
            await log(f"{video_path} has file extenstion {ext} which isn't supported for video input", 'error')
            return ERROR_TOKEN, f"{video_path} has file extenstion {ext} which isn't supported for video input"

        try:
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

    def cancel_global(self):
        self.generation_cancelled = True

    async def _generator(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, file_path: None | str = None, 
                   mod_ = 10, video_save_buffer_format = "JPEG", system_prompt_override: str | None = None, options : dict | None = None, format_: dict | None = None, tools_override:None|list = None):
        
        endpoint = self._get_endpoint()
        url = f"{self.host}{endpoint}"

        messages = []
        if system_prompt_override is None:
            if self.system:
                messages.append({'role': "system", 'content': self.system})
        else:
            messages.append({'role': "system", 'content': system_prompt_override})

        if query and query.strip(): messages += context + [{"role": "user", "content": query}]

        headers = {"Content-Type": "application/json"}

        if self.api_key: headers['Authorization'] = f'Bearer {self.api_key}'

        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream
        }

        if self.has_tools:
            data["tools"] = tools_override if tools_override else self.tools

        if options:
            data["options"] = options

        if format_:
            data["format"] = format_

        if self.has_CoT and think is not None:
            data['think'] = think

        if self.has_vision and file_path is not None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in IMAGE_EXTs:
                image, e = await self._encode_image(image_path=file_path)
                if image == ERROR_TOKEN:
                    await log(f"Error encoding image!: {repr(e)}", 'error')
                    yield (ERROR_TOKEN, ERROR_TOKEN, [])
                    return
                else:
                    data['messages'][-1]['images'] = [image]
            elif ext in VIDEO_EXTs and self.has_video:
                frames, e = await self._encode_frames_from_vid(video_path=file_path, mod_= mod_, format_=video_save_buffer_format)
                if frames == ERROR_TOKEN:
                    await log(f"Error encoding video!: {repr(e)}", 'error')
                    yield (ERROR_TOKEN, ERROR_TOKEN, [])
                    return
                else:
                    data['messages'][-1]['images'] = frames
            elif ext in VIDEO_EXTs and not self.has_video:
                await log(f"Cannot process video file {file_path}: Model {self.name} is not configured for video processing.", 'warn')

        await self.change_state(BUSY)

        buffer = ""

        if not self.resource_manager.session: raise

        try:
            timeout = aiohttp.ClientTimeout(total=None)
            async with self.resource_manager.session.post(url, headers=headers, data=json.dumps(data), timeout=timeout) as response:
                response.raise_for_status()

                if stream:
                    try:
                        async for raw_chunk in response.content.iter_chunked(1024):

                            if self.generation_cancelled:
                                await log(f"Cancellation requested for {self.name}, breaking stream", "info")
                                break

                            chunk = raw_chunk.decode("utf-8", errors='ignore')
                            buffer += chunk

                            while "\n" in buffer and not self.generation_cancelled and (not self.resource_manager.session.closed):
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
                                self.generation_cancelled = False
                                break

                            if self.resource_manager.session.closed: # type:ignore
                                await response.release()

                    except asyncio.CancelledError:
                        await log(f"Generation cancelled for {self.name}", "info")
                        await response.release()
                        response.close()
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
                        await response.release()
                        response.close()
                        return 

        except asyncio.CancelledError:
            await log(f"Request cancelled for {self.name}", "info")
           
            self.generation_cancelled = False          
            return

        except Exception as e:
            await log(f"Ollama API Request Error: {e}", "error")
            yield (ERROR_TOKEN, ERROR_TOKEN, [])

        self.generation_cancelled = False
        await self.change_state(IDLE)

    def update_port(self, port):
        self.port = port
        self.host =  f"http://localhost:{self.port}"

    async def _warmer(self,
        use_mmap=False,
        warmup_image_path="main/test.jpg",
        use_custom_keep_alive_timeout=False,
        custom_keep_alive_timeout: str = "5m",
        has_video_processing=False,
        warmup_video_path="main/test.mp4"):

        self.has_video = has_video_processing

        async with self.state_lock:
            if self.state not in (IDLE, DOWN):
                raise RuntimeError(f"{self.name} cannot warm up from state={self.state}")

        await self.change_state(WARMING_UP)

        try:

            await self.resource_manager.wait_until_ready(self.host)

            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }

            headers = {"Content-Type": "application/json"}
            if self.api_key: headers['Authorization'] = f'Bearer {self.api_key}'
            endpoint = self._get_endpoint()
            url = f"{self.host}{endpoint}"
            options = {}
            if use_mmap:
                options["use_mmap"] = True
            if use_custom_keep_alive_timeout:
                options["keep_alive"] = custom_keep_alive_timeout
            
            if self.has_vision:
                    if self.has_video and os.path.exists(warmup_video_path):
                        await log("Warming up with video frame extraction...", 'info')
                        encoded_frames, e = await self._encode_frames_from_vid(warmup_video_path, mod_=50)
                        if encoded_frames == ERROR_TOKEN:
                            await log(f"Error encoding video frames for warmup: {e}, UX will degrade.", 'warn')

                        if encoded_frames and encoded_frames != ERROR_TOKEN:
                            if "messages" in data:
                                data['messages'][-1]['images'] = encoded_frames
                    elif os.path.exists(warmup_image_path):
                        await log("Warming up with static image...", 'info')
                        encoded_image, e = await self._encode_image(warmup_image_path)
                        if encoded_image == ERROR_TOKEN:
                            await log(f"Error encoding image for warmup!: {e}, UX will degrade.", 'warn')
                        else:
                            if "messages" in data:
                                data['messages'][-1]['images'] = [encoded_image]
                    else:
                        await log("Vision model, but no video processing enabled and no warmup image / video found. Skipping vision test.", 'warn')

            data["options"] = options

            if not self.resource_manager.session: raise

            await log('Trying Non-Streaming...' ,'info')
            async with self.resource_manager.session.post(url, headers=headers, data=json.dumps(data)) as response: 
                response.raise_for_status()
                try:
                    resp = await response.json()
                    await log(f"Raw non-streaming chunk: {resp}", 'info')
                except Exception as e:
                    await log(f"Failed to load model's response while non streaming: {repr(e)}", 'error')
                    raise Exception(f"Failed to load model's response while non streaming: {repr(e)}")

            data['stream'] = True

            await log('Trying Streaming...' ,'info')
            buffer = ""
            try:
                async with self.resource_manager.session.post(url, headers=headers, data=json.dumps(data)) as response:
                    response.raise_for_status()

                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode("utf-8")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                json_line = json.loads(line)

                                _ = json_line.get("message", {}).get("thinking", "")
                                _ = json_line.get("message", {}).get("content", "")
                                _ = json_line.get("message", {}).get("tool_calls", []) 
                                await log(f"Raw streaming chunk: {json_line}", 'info')
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                    await log(f"Failed to load model's response while streaming: {repr(e)}", 'error')
                    raise Exception(f"Failed to load model's response while streaming: {repr(e)}")
                    
            self.warmed_up = True
            await log(f"{self.name} ({self.model_name}) warmed up!", "success")

        except Exception:
            self.warmed_up = False
            raise

        finally: 
            await self.change_state(IDLE if self.warmed_up else DOWN)

            if self.resource_manager.session and not self.warmed_up:
                await self.resource_manager.session.close()



class OllamaEmbedder(Model): 
    def __init__(self, role, name: str, model_name: str, port:int, api_key = None) -> None:
        self.port = port
        self.host =  f"http://localhost:{self.port}"
        super().__init__(role, self.host, name, model_name, api_key, DOWN)
        self.generation_cancelled = False
                
    def cancel_global(self):
        self.generation_cancelled = True

    def _get_endpoint(self) -> str:
        return "/api/embed"
    
    def update_port(self, port):
        self.port = port
        self.host =  f"http://localhost:{self.port}"

    async def get_model_details(self):
        url = f"{self.host}/api/show"
        data = {
            "model": self.model_name
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key: headers['Authorization'] = f'Bearer {self.api_key}'

        if not self.resource_manager.session:
           self.resource_manager.create_session()
        
        if not self.resource_manager.session:
           raise
    
        async with self.resource_manager.session.post(url, headers=headers, data= json.dumps(data)) as resp:
            res = await resp.json()

        return await self._get_model_details_helper(res, "embedding", False)

    
    async def embed(self, input_:str | list[str] | tuple[str]):
        if isinstance(input_, tuple):
            input_ = list(input_)
        elif isinstance(input_, str):
            input_ = [input_]

        endpoint = self._get_endpoint()
        url = f"{self.host}{endpoint}"

        headers = {"Content-Type": "application/json"}
        if self.api_key: headers['Authorization'] = f'Bearer {self.api_key}'

        data = {
            "model": self.model_name,
            "input": input_
        }

        await self.change_state(BUSY)

        if not self.resource_manager.session: raise

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with self.resource_manager.session.post(url, headers=headers, data=json.dumps(data), timeout=timeout) as response:
                response.raise_for_status()

                if self.generation_cancelled:
                    await response.release()
                    self.generation_cancelled = False

                if self.resource_manager.session.closed:
                    await response.release()

                try:
                    res_json = await response.json()
                    embeds = res_json.get('embeddings', [])
                    return embeds
                except asyncio.CancelledError:
                    await log(f"Non-stream generation cancelled for {self.name}", "info")

        except asyncio.CancelledError:
            await log(f"Request cancelled for {self.name}", "info")
           
            self.generation_cancelled = False
            return

        except Exception as e:
            await log(f"Ollama API Request Error: {e}", "error")
            return (ERROR_TOKEN)
        
        finally:
            self.generation_cancelled = False
            await self.change_state(IDLE)

    async def _warmer(self):

        async with self.state_lock:
            if self.state not in (IDLE, DOWN):
                raise RuntimeError(f"{self.name} cannot warm up from state={self.state}")

        await self.change_state(WARMING_UP)

        try:
            self.resource_manager.create_session()
            
            await self.resource_manager.wait_until_ready(self.host)

            data = {
            "model": self.model_name,
            "input": [
                'hi',
                'hello'
            ]
            }

            headers = {"Content-Type": "application/json"}
            if self.api_key: headers['Authorization'] = f'Bearer {self.api_key}'
            endpoint = self._get_endpoint()
            url = f"{self.host}{endpoint}"

            if not self.resource_manager.session: raise

            timeout = aiohttp.ClientTimeout(total=120)
            async with self.resource_manager.session.post(url, headers=headers, data=json.dumps(data), timeout=timeout) as response:
                response.raise_for_status()

                if self.generation_cancelled:
                    await response.release()
                    self.generation_cancelled = False

                if self.resource_manager.session.closed:
                    await response.release()

                try:
                    res_json = await response.json()
                    embeds = res_json.get('embeddings', [])
                    await log(f'Raw output: {embeds}', 'info')
                    self.warmed_up = True
                except Exception as e:
                    await log(f"Failed to load embedding model's response while warming up: {repr(e)}", 'error')
                    raise Exception(f"Failed to load embedding model's response while warming up: {repr(e)}")
        except Exception:
            self.warmed_up = False
            raise
        finally: 
            await self.change_state(IDLE if self.warmed_up else DOWN)