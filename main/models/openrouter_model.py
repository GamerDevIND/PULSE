import asyncio
import base64
import aiohttp
import json
from io import BytesIO
from main.resource_manager import SessionManager
from .base_model import Model
from main.utils import log
from main.configs import IMAGE_EXTs, VIDEO_EXTs, ERROR_TOKEN
import aiofiles
import av
from copy import deepcopy
import os

DOWN = "down"
IDLE = "idle"
BUSY = 'busy'

class OpenRouterModel(Model):
    def __init__(self, role: str, name: str, model_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, system_prompt: str, api_key: None | str = None, **kwargs) -> None:
        self.host = f"https://openrouter.ai/api/v1/chat/completions"
        super().__init__(role, self.host, name, model_name, api_key, DOWN)
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.has_vision = has_vision
        self.port:int|None = None
        self.system = system_prompt
        self.has_video = False
        self.resource_manager = SessionManager(self.model_name)
        self.input_handler = InputHandler()

    def update_port(self, port):
        pass

    def set_has_video(self, has_video:bool):
        self.has_video = has_video

    async def _encode_frames_from_vid(self, video_path, mod_ = 1, format_ = "JPEG"):
        if not self.has_vision:
            return [], None
        try:
            return await self.input_handler._encode_frames_from_vid(video_path, mod_, format_, 5)
        except Exception as e:
            await log(f"Error in video frame encoding for {self.name}: {e}", 'error')
            return ERROR_TOKEN, repr(e)
    
    async def _encode_image(self, image_path):
        return await self.input_handler._encode_image(image_path)

    async def get_model_details(self):
        url = "https://openrouter.ai/api/v1/models"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if not self.resource_manager.session:
           self.resource_manager.create_session()
        
        if not self.resource_manager.session:
           raise

        async with self.resource_manager.session.get(url, headers=headers) as resp:
            data = await resp.json()
            res = next((m for m in data.get('data', []) if m['id'] == self.model_name), {})


        return await self._get_model_details_helper(res)

    async def warm_up(
        self, use_mmap=False, warmup_image_path="main/test.jpg", use_custom_keep_alive_timeout=True, custom_keep_alive_timeout: str = "-1", 
        has_video_processing=False, warmup_video_path="main/test.mp4"): 
        self.resource_manager.create_session()
        self.warmed_up = True
        await self.change_state(IDLE)

    async def shutdown(self):
        if self.state == DOWN: return
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': 'bye'}],
            "stream": True
        }
        await self.resource_manager.shutdown(self.host, headers, payload)
        await self.change_state(DOWN)

    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                   mod_ = 1, video_save_buffer_format = "JPEG", system_prompt_override: str | None = None, options : dict | None = None, format_: dict | None = None, tools_override:None|list = None):
        
        if not self.api_key:
            raise ValueError

        messages = []
        if system_prompt_override is None:
            if self.system:
                messages.append({'role': "system", 'content': self.system})
        else:
            messages.append({'role': "system", 'content': system_prompt_override})

        if query and query.strip(): messages += context + [{"role": "user", "content": query}]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }

        if self.has_tools:
            data["tools"] = tools_override if tools_override else self.tools

        if format_:
            format_ = deepcopy(format_)
            format_["additionalProperties"] = False
            data["response_format"] = {
                "type": "json_schema",
                "name": "format",
                "strict": True,
                "schema": format_
            }
            
        if query and query.strip(): 
            if self.has_vision and image_path is not None:
                ext = os.path.splitext(image_path)[1].lower()
                if ext in IMAGE_EXTs:
                    image, e = await self._encode_image(image_path=image_path)
                    if image == ERROR_TOKEN:
                        await log(f"Error encoding image!: {repr(e)}", 'error')
                        yield (ERROR_TOKEN, ERROR_TOKEN, [])
                        return
                    else:
                        data_url = f"data:image/jpeg;base64,{image}"
                        data['messages'][-1] = {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'text',
                                    "text": query
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url
                                    }
                                }
                            ]
                        }
                elif (self.input_handler.is_url(image_path) or ext in VIDEO_EXTs) and self.has_video:
                    frames, e = await self._encode_frames_from_vid(video_path=image_path, mod_= mod_, format_=video_save_buffer_format)
                    if frames == ERROR_TOKEN:
                        await log(f"Error encoding video!: {repr(e)}", 'error')
                        yield (ERROR_TOKEN, ERROR_TOKEN, [])
                        return
                    else:
                        if not isinstance(frames, list):
                            data_url = image_path if self.input_handler.is_url(image_path) else f"data:video/mp4;base64,{frames}"
                            data['messages'][-1] = {
                                'role': 'user',
                                'content': [
                                    {
                                        'type': 'text',
                                        "text": query
                                    },
                                    {
                                        "type": "video_url",
                                        "video_url": {
                                            "url": data_url
                                        }
                                    }
                                ]
                            }
                        else:
                            data['messages'][-1] = {
                                'role': 'user',
                                'content': [
                                    {
                                        'type': 'text',
                                        "text": query
                                    }
                                ]
                            }
                            for frame in frames:
                                data['messages'][-1]['content'].append({
                                        "type": "image_url",
                                        "image_url": {
                                        "url": f"data:image/jpeg;base64,{frame}"
                                        }
                                    })

                elif ext in VIDEO_EXTs and not self.has_video:
                    await log(f"Cannot process video file {image_path}: Model {self.name} is not configured for video processing.", 'warn')

        if not self.resource_manager.session: self.resource_manager.create_session()

        if not self.resource_manager.session: raise

        buffer = ""

        try:
            timeout = aiohttp.ClientTimeout(total=None)
            async with self.resource_manager.session.post(self.host, headers=headers, data=json.dumps(data), timeout=timeout) as response:
                if response.status != 200:
                    error_data = await response.json()
                    await log(f"Error during generation of {self.name}: {error_data['error']['message']}", 'error')
                    yield (ERROR_TOKEN, ERROR_TOKEN, [])
                    return
                
                response.raise_for_status()

                if stream:
                    try:
                        async for raw_chunk in response.content.iter_chunked(1024):

                            chunk = raw_chunk.decode("utf-8", errors='ignore')
                            buffer += chunk

                            while "\n" in buffer and (not self.resource_manager.session.closed):
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()

                                if line.startswith('data: '):
                                    line = line[6:]
                                    if line == '[DONE]':
                                        break

                                if not line:
                                    continue

                                try:
                                    json_line = json.loads(line)

                                    if "error" in json_line:
                                        await log(f"Error during generation of {self.name}: {json_line['error']['message']}", 'error')
                                        yield (ERROR_TOKEN, ERROR_TOKEN, [])
                                        return

                                    choices = list(json_line['choices'])
                                    message = choices[0].get('delta', choices[0].get('message'))
                                    thinking = message.get('reasoning', message.get('thinking', message.get("reasoning_content" ,"")))
                                    content = message.get('content', "")
                                    tools = message.get("tool_calls", message.get('tools',[]))
                                    yield (thinking, content, tools)
                                except json.JSONDecodeError:
                                    continue

                            if self.resource_manager.session.closed:
                                await response.release()

                    except asyncio.CancelledError:
                        await log(f"Generation cancelled for {self.name}", "info")
                        await response.release()
                        response.close()
                        return 

                else:
                    try:
                        res_json = await response.json()
                        choices = list(res_json['choices'])
                        message = choices[0]['message']
                        thinking = message.get('reasoning', message.get('thinking', ""))
                        content = message.get('content', "")
                        tools = message.get("tool_calls", message.get('tools', []))
                        yield (thinking, content, tools)
                    except asyncio.CancelledError:
                        await log(f"Non-stream generation cancelled for {self.name}", "info")
                        await response.release()
                        response.close()
                        return 

        except asyncio.CancelledError:
            await log(f"Request cancelled for {self.name}", "info")    
            return

        except Exception as e:
            await log(f"Openrouter API Request Error: {e}", "error")
            yield (ERROR_TOKEN, ERROR_TOKEN, [])

class InputHandler:
    def __init__(self) -> None:
        pass

    def is_url(self, url:str):
        url = url.lower()
        return url.startswith('http://') or url.startswith('https://') or url.startswith('www.')
    
    async def _encode_frames_from_vid(self, video_path, mod_ = 1, format_ = "JPEG", max_video_size_mbs = 5):
        if self.is_url(video_path):
            return video_path, None
        
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

class OpenRouterEmbedder(Model):
    def __init__(self, role, name: str, model_name: str, api_key:str, **kwargs) -> None:
        self.host =  f"https://openrouter.ai/api/v1/embeddings"
        super().__init__(role, self.host, name, model_name, api_key, DOWN, **kwargs)
        self.resource_manager = SessionManager(self.model_name)
    
    def update_port(self, port):
        pass

    async def warm_up(
        self): 
        self.resource_manager.create_session()
        self.warmed_up = True
        await self.change_state(IDLE)

    async def shutdown(self):
        if self.state == DOWN: return
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            'model': self.model_name,
            "input": "bye",
        }
        await self.resource_manager.shutdown(self.host, headers, payload)
        await self.change_state(DOWN)

    async def get_model_details(self):
        url = "https://openrouter.ai/api/v1/models"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if not self.resource_manager.session:
           self.resource_manager.create_session()
        
        if not self.resource_manager.session:
           raise

        async with self.resource_manager.session.get(url, headers=headers) as resp:
            data = await resp.json()
            res = next((m for m in data.get('data', []) if m['id'] == self.model_name), {})

        return await self._get_model_details_helper(res, "embedding", False)

    async def embed(self, input_:str | list[str] | tuple[str]):
        if isinstance(input_, tuple):
            input_ = list(input_)
        elif isinstance(input_, str):
            input_ = [input_]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "model": self.model_name,
            "input": input_,
            "provider": {
            "data_collection": "deny",
            }
        }

        await self.change_state(BUSY)

        if not self.resource_manager.session: self.resource_manager.create_session()
        if not self.resource_manager.session: raise   

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with self.resource_manager.session.post(self.host, headers=headers, data=json.dumps(data), timeout=timeout) as response:
                response.raise_for_status()
                if self.resource_manager.session.closed:
                    await response.release()

                try:
                    res_json = await response.json()
                    data_list = res_json.get('data', [])
                    embeds = [item.get('embedding', []) for item in data_list]
                    return embeds if isinstance(input_, list) else embeds[0]
                except asyncio.CancelledError:
                    await log(f"Non-stream generation cancelled for {self.name}", "info")

        except asyncio.CancelledError:
            await log(f"Request cancelled for {self.name}", "info")
            return

        except Exception as e:
            await log(f"Ollama API Request Error: {e}", "error")
            return (ERROR_TOKEN)
        
        finally:
            await self.change_state(IDLE)