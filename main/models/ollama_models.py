import os
import asyncio
import aiohttp
import json
from main.utils import log, strip_thinking
from main.configs import IMAGE_EXTs, VIDEO_EXTs, AUDIO_EXTs, ERROR_TOKEN
from main.events import EventBus
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
    def __init__(self, role: str, name: str, model_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, port: int, system_prompt: str, 
                 api_key: None | str = None, event_bus: None | EventBus = None):
        self.port = port
        self.host =  f"http://localhost:{self.port}"
        super().__init__(role, self.host, name, model_name, api_key, DOWN, event_bus)       
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.has_vision = has_vision
        self.system = system_prompt
        self.generation_cancelled = False

    def _get_endpoint(self) -> str:
        return "/api/chat"

    async def get_model_details(self):
        if self.details_cache: return self.details_cache
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
        try:
            return await self.input_handler.encode_frames_from_vid(video_path, False, mod_, format_, 5)
        except Exception as e:
            await log(f"Error in video frame encoding for {self.name}: {e}", 'error')
            return ERROR_TOKEN, repr(e)
    
    async def _encode_image(self, image_path):
        return await self.input_handler.encode_image(image_path, False)
    
    async def _encode_audio(self, audio_path):
        return await self.input_handler.encode_audio(audio_path, False)


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

        if self.has_tools and (self.tools or tools_override):
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
                    await log(f"Error encoding image! Skipping: {repr(e)}", 'error')
                else:
                    data['messages'][-1]['images'] = [image]
            elif ext in VIDEO_EXTs and self.has_video:
                frames, e = await self._encode_frames_from_vid(video_path=file_path, mod_= mod_, format_=video_save_buffer_format)
                if frames == ERROR_TOKEN:
                    await log(f"Error encoding video! Skipping: {repr(e)}", 'error')
                else:
                    data['messages'][-1]['images'] = frames
            elif ext in VIDEO_EXTs and not self.has_video:
                await log(f"Cannot process video file {file_path}: Model {self.name} is not configured for video processing.", 'warn')
                
        if self.has_audio and file_path is not None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in AUDIO_EXTs:
                audio, e = await self._encode_audio(audio_path=file_path)

                if audio  == ERROR_TOKEN:
                    await log(f"Error encoding Audio! Skipping: {repr(e)}", 'error')
                else:
                    current_images = data['messages'][-1].get('images', [])
                    current_images.append(audio)
                    data['messages'][-1]['images'] = current_images

        if not self.resource_manager.session: raise

        await self.change_state(BUSY)

        buffer = ""

        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = "Generating response...")

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
                        if not (thinking or thinking.strip()):
                            c, t = strip_thinking(content)
                            if c:
                                content = c
                            if t:
                                thinking = t

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
            if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.ERROR, msg = f"Ollama API Request Error: {e}")
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
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Warming up: {self.name} ({self.model_name})")

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
            if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"{self.name} ({self.model_name}) warmed up!")

        except Exception:
            self.warmed_up = False
            raise

        finally: 
            await self.change_state(IDLE if self.warmed_up else DOWN)

            if self.resource_manager.session and not self.warmed_up:
                await self.resource_manager.session.close()



class OllamaEmbedder(Model): 
    def __init__(self, role, name: str, model_name: str, port:int, api_key = None, event_bus: None | EventBus = None) -> None:
        self.port = port
        self.host =  f"http://localhost:{self.port}"
        super().__init__(role, self.host, name, model_name, api_key, DOWN, event_bus)
        self.generation_cancelled = False
                
    def cancel_global(self):
        self.generation_cancelled = True

    def _get_endpoint(self) -> str:
        return "/api/embed"
    
    def update_port(self, port):
        self.port = port
        self.host =  f"http://localhost:{self.port}"

    async def get_model_details(self):
        if self.details_cache: return self.details_cache
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

        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = "Embedding input(s)")

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
            if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.ERROR, msg = f"Ollama API Request Error: {e}")
            return (ERROR_TOKEN)
        
        finally:
            self.generation_cancelled = False
            await self.change_state(IDLE)

    async def _warmer(self):

        async with self.state_lock:
            if self.state not in (IDLE, DOWN):
                raise RuntimeError(f"{self.name} cannot warm up from state={self.state}")

        await self.change_state(WARMING_UP)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Warming up: {self.name} ({self.model_name})")

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
                    if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Warming up done for: {self.name}({self.model_name})")
                    self.warmed_up = True
                except Exception as e:
                    await log(f"Failed to load embedding model's response while warming up: {repr(e)}", 'error')
                    raise Exception(f"Failed to load embedding model's response while warming up: {repr(e)}")
        except Exception:
            self.warmed_up = False
            raise RuntimeError
        finally:
            await self.change_state(IDLE if self.warmed_up else DOWN)