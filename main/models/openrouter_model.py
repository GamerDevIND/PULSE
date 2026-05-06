import asyncio
import aiohttp
import json
from main.resource_manager import SessionManager
from main.events import EventBus
from .base_model import Model
from main.utils import Logger, strip_thinking
from main.configs import IMAGE_EXTs, VIDEO_EXTs, AUDIO_EXTs, ERROR_TOKEN
from copy import deepcopy
import os

DOWN = "down"
IDLE = "idle"
BUSY = 'busy'

class OpenRouterModel(Model):
    def __init__(self, role: str, name: str, model_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, system_prompt: str, api_key: None | str = None, event_bus: None | EventBus = None, **kwargs) -> None:
        self.host = f"https://openrouter.ai/api/v1/chat/completions"
        super().__init__(role, self.host, name, model_name, api_key, DOWN, event_bus)
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.has_vision = has_vision
        self.port:int|None = None
        self.url_media_valid = False
        self.system = system_prompt
        self.has_video = False
        self.resource_manager = SessionManager(self.model_name)

    def update_port(self, port):
        pass

    async def get_model_details(self):
        if self.details_cache: return self.details_cache
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
    
    def set_url_valid_for_media(self, val:bool):
        self.url_media_valid = val

    async def warm_up(
        self, use_mmap=False, warmup_image_path="main/test.jpg", use_custom_keep_alive_timeout=True, custom_keep_alive_timeout: str = "-1", 
        has_video_processing=False, warmup_video_path="main/test.mp4"):
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Warming up: {self.name} ({self.model_name})")
        self.resource_manager.create_session()
        self.warmed_up = True
        await self.change_state(IDLE)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"{self.name} ({self.model_name}) warmed up!")

    async def _encode_frames_from_vid(self, video_path, mod_ = 1, format_ = "JPEG"):
        if not self.has_vision:
            return [], None
        try:
            return await self.input_handler.encode_frames_from_vid(video_path, self.url_media_valid, mod_, format_, 5)
        except Exception as e:
            await Logger.log_async(f"Error in video frame encoding for {self.name}: {e}", 'error')
            return ERROR_TOKEN, repr(e)
    
    async def _encode_image(self, image_path):
        return await self.input_handler.encode_image(image_path, self.url_media_valid)
    
    async def _encode_audio(self, audio_path):
        return await self.input_handler.encode_audio(audio_path, self.url_media_valid)

    async def get_multimodal_data(self, data, query:str | None, file_path, mod_, video_save_buffer_format):
        data = data.copy()
        if (query and query.strip()) and file_path and (self.has_vision or self.has_audio): 
            last_msg = data['messages'][-1]
            if isinstance(last_msg.get('content'), str):
                last_msg['content'] = [{'type': 'text', 'text': last_msg['content']}]
            else:
                last_msg = data['messages'][-1]
            if isinstance(last_msg.get('content'), list):
                text_content = next((item['text'] for item in last_msg['content'] if item['type'] == 'text'), "")
                last_msg['content'] = text_content

        if self.has_vision and file_path is not None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in IMAGE_EXTs:
                image, e = await self._encode_image(image_path=file_path)
                if image == ERROR_TOKEN:
                    await Logger.log_async(f"Error encoding image! Skipping: {repr(e)}", 'error')
                else:
                    data_url = f"data:image/jpeg;base64,{image}"
                    d = data['messages'][-1].get('content', [])
                    if not isinstance(d, list):
                        d = [d]
                    
                    d.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                    })
                            
            if (self.input_handler.is_url(file_path) or ext in VIDEO_EXTs) and self.has_video:
                frames, e = await self._encode_frames_from_vid(video_path=file_path, mod_= mod_, format_=video_save_buffer_format)
                if frames == ERROR_TOKEN:
                    await Logger.log_async(f"Error encoding video! Skipping: {repr(e)}", 'error')

                else:

                    if not isinstance(frames, list):
                        data_url = file_path if self.input_handler.is_url(file_path) else f"data:video/mp4;base64,{frames}"
                        d = data['messages'][-1].get('content', [])
                        if not isinstance(d, list):
                            d = [d]
                        d.append({
                                "type": "video_url",
                                "video_url": {
                                    "url": data_url
                                }
                            })
                    else:
                        d = data['messages'][-1].get('content', [])
                        if not isinstance(d, list):
                            d = [d]
                        for frame in frames:

                            d.append({
                                    "type": "image_url",
                                    "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame}"
                                        }
                                })
                                
                        data['messages'][-1]["content"] = d

            elif ext in VIDEO_EXTs and not self.has_video:
                await Logger.log_async(f"Cannot process video file {file_path}: Model {self.name} is not configured for video processing.", 'warn')

        if self.has_audio and file_path is not None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in AUDIO_EXTs:
                audio, e = await self._encode_audio(audio_path=file_path)

                if audio  == ERROR_TOKEN:
                    await Logger.log_async(f"Error encoding Audio! Skipping: {repr(e)}", 'error')
                else:
                    d = data['messages'][-1]['content'] 
                    
                    if not isinstance(d, list):
                        d = [d]

                    d.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio,
                            "format": f"{ext.removeprefix('.')}"
                        }
                    })

        return data

    async def shutdown(self):
        if self.state == DOWN: return
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Shutting down model: {self.name} ({self.model_name})")
        await self.resource_manager.shutdown(send_shutdown_paylod=False)
        await self.change_state(DOWN)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Model shutdown: {self.name} ({self.model_name})")

    async def generate(self, query: str | None, context: list[dict], stream: bool, think: str | bool | None = False, file_path: None | str = None, 
                   mod_ = 1, video_save_buffer_format = "JPEG", system_prompt_override: str | None = None, options : dict | None = None, format_: dict | None = None, tools_override:None|list = None):
        
        if not self.api_key:
            raise ValueError
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        tools = None

        if self.has_tools and (self.tools or tools_override):
            tools = tools_override if tools_override else self.tools

        data = self.build_payload(query, context, stream, self.system, system_prompt_override, tools)

        if format_:
            format_ = deepcopy(format_)
            format_["additionalProperties"] = False
            data["response_format"] = {
                "type": "json_schema",
                "name": "format",
                "strict": True,
                "schema": format_
            }
            
        data = await self.get_multimodal_data(data, query, file_path, mod_, video_save_buffer_format)

        if not self.resource_manager.session: self.resource_manager.create_session()

        if not self.resource_manager.session: raise
        await self.change_state(BUSY)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Generating response for {self.name}({self.model_name}) [{self.role}]...")

        buffer = ""

        try:
            timeout = aiohttp.ClientTimeout(total=None)
            async with self.resource_manager.session.post(self.host, headers=headers, data=json.dumps(data), timeout=timeout) as response:
                if response.status != 200:
                    error_data = await response.json()
                    await Logger.log_async(f"Error during generation of {self.name}: {error_data['error']['message']}", 'error')
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
                                if line:
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
                                        await Logger.log_async(f"Error during generation of {self.name}: {json_line['error']['message']}", 'error')
                                        yield (ERROR_TOKEN, ERROR_TOKEN, [])
                                        await self.change_state(IDLE)
                                        return

                                    choices = list(json_line['choices'])
                                    message = choices[0].get('delta', choices[0].get('message'))
                                    thinking = message.get('reasoning', message.get('thinking', message.get("reasoning_content" ,"")))
                                    content = message.get('content', "")
                                    tools = message.get("tool_calls", message.get('tools',[]))
                                    if not isinstance(tools, (list, tuple)):
                                        tools = [tools]

                                    yield (thinking, content, tools)
                                except json.JSONDecodeError:
                                    continue

                            if self.resource_manager.session.closed:
                                await response.release()

                    except asyncio.CancelledError:
                        await Logger.log_async(f"Generation cancelled for {self.name}", "info")
                        await response.release()
                        response.close()
                        await self.change_state(IDLE)
                        return 

                else:
                    try:
                        res_json = await response.json()
                        choices = list(res_json['choices'])
                        message = choices[0]['message']
                        thinking = message.get('reasoning', message.get('thinking', ""))
                        content = message.get('content', "")
                        tools = message.get("tool_calls", message.get('tools',[]))

                        if not (thinking and thinking.strip()):
                            c, t = strip_thinking(content)
                            if c:
                                content = c
                            if t:
                                thinking = t
                        
                        if not isinstance(tools, (list, tuple)):
                            tools = [tools]
                                        
                        yield (thinking, content, tools)
                    except asyncio.CancelledError:
                        await Logger.log_async(f"Non-stream generation cancelled for {self.name}", "info")
                        await response.release()
                        response.close()
                        await self.change_state(IDLE)
                        return 

        except asyncio.CancelledError:
            await Logger.log_async(f"Request cancelled for {self.name}", "info")    
            return

        except Exception as e:
            if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.ERROR, msg = f"Openrouter API Request Error: {e}")
            yield (ERROR_TOKEN, ERROR_TOKEN, [])
            
        finally:
            await self.change_state(IDLE)       


class OpenRouterEmbedder(Model):
    def __init__(self, role, name: str, model_name: str, api_key:str, event_bus: None | EventBus = None, **kwargs) -> None:
        self.host =  f"https://openrouter.ai/api/v1/embeddings"
        super().__init__(role, self.host, name, model_name, api_key, DOWN, event_bus, **kwargs)
        self.resource_manager = SessionManager(self.model_name)
    
    def update_port(self, port):
        pass

    async def warm_up(
        self): 
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Warming up: {self.name} ({self.model_name})")
        self.resource_manager.create_session()
        self.warmed_up = True
        await self.change_state(IDLE)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"{self.name} ({self.model_name}) warmed up!")

    async def shutdown(self):
        if self.state == DOWN: return
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Shutting down model: {self.name} ({self.model_name})")

        await self.resource_manager.shutdown(send_shutdown_paylod=False)
        await self.change_state(DOWN)
        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Model shutdown: {self.name} ({self.model_name})")

    async def get_model_details(self):
        if self.details_cache: return self.details_cache
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

        if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.INFO, msg = f"Embedding input(s) with {self.name}({self.model_name})")

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
                    await Logger.log_async(f"Non-stream generation cancelled for {self.name}", "info")

        except asyncio.CancelledError:
            await Logger.log_async(f"Request cancelled for {self.name}", "info")
            return

        except Exception as e:
            await Logger.log_async(f"Openrouter API Request Error: {e}", "error")
            if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.ERROR, msg = f"Openrouter API Request Error: {e}")
            return (ERROR_TOKEN)
        
        finally:
            await self.change_state(IDLE)