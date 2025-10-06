import os
import json
import asyncio
import aiohttp
import aiofiles
import av
from utils import log
from configs import ERROR_TOKEN, VIDEO_EXTs, IMAGE_EXTs
import base64
from io import BytesIO

class Model:
    def __init__(self, role: str, name: str, ollama_name: str, has_tools: bool, has_CoT: bool, has_vision:bool, port: int, system_prompt: str):
        self.role = role
        self.name = name
        self.ollama_name = ollama_name
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.has_vision = has_vision
        self.port = port
        self.system = system_prompt
        self.host = f"http://localhost:{self.port}"
        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = self.host
        self.warmed_up = False
        self.session: aiohttp.ClientSession | None = None
        self.process = None
        self._start_lock = asyncio.Lock()

    def _get_endpoint(self) -> str:
        return "/api/chat"

    async def wait_until_ready(self, url: str, timeout: int = 30):
        await log(f"Waiting for {self.name} on {url}...", "info")
        for i in range(timeout):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/api/tags") as res:
                        if res.status == 200:
                            await log(f"{self.name} is ready!", "success")
                            return
            except aiohttp.ClientError:
                await log(f"Retries: {i+1} / {timeout}", "info")
            await asyncio.sleep(1)
        raise TimeoutError(f"🟥 Ollama server for {self.name} did not start in time.")

    async def _encode_frames_from_vid(self, video_path, mod_ = 1):
        if not self.has_vision:
            return [], None
        try:
            container = av.open(video_path)
            frames = []
            await log("Audio processing hasn't been implemented yet.", 'warn')
            for i, frame in enumerate(container.decode(video=0)): # type: ignore
                if i % mod_ == 0:
                    img = frame.to_image().rotate(-90)
                    buffer = BytesIO()
                    img.save(buffer,  format="PNG")
                    frame = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    frames.append(frame)
            return frames, None
        except Exception as e:
            return ERROR_TOKEN, e

    async def _encode_image(self,image_path):
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            base64_image = base64.b64encode(image_data).decode("utf-8")
            return base64_image, None
        except Exception as e:
            return ERROR_TOKEN, e

    async def warm_up(self, use_mmap= False, warmup_image_path="main/test.jpg", use_custom_keep_alive_timeout = False, custom_keep_alive_timeout: str = "5m", has_video_processing = False, warmup_video_path = "main/test.mp4"):
        async with self._start_lock:
            self.use_custom_keep_alive_timeout = use_custom_keep_alive_timeout
            self.custom_keep_alive_timeout = custom_keep_alive_timeout
            self.use_mmap = use_mmap
            self.has_video = has_video_processing
            if self.warmed_up:
                if self.process and (self.process.returncode is None):
                    return
            
            if self.process is not None:
                await log(f"Cleaning up old process for {self.name}.", "warn")
                self.process.terminate() 
                self.process = None

            if self.session is not None:
                await self.session.close()
                self.session = None

            log_dir = os.path.join( "main", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f"{self.ollama_name}.log")
            
            await log(f"{self.name} ({self.ollama_name}) warming up...", "info")
            async with aiofiles.open(log_file_path, "w") as f:
                self.process = await asyncio.create_subprocess_exec(*self.start_command, env=self.ollama_env, stderr=asyncio.subprocess.STDOUT, stdout=f.fileno())
                # self.process = subprocess.Popen(
                #     self.start_command,
                #     env=self.ollama_env,
                #     stdout=f,
                #     stderr=subprocess.STDOUT
                # )

            await self.wait_until_ready(self.host)
            
            if not self.session:
                self.session = aiohttp.ClientSession()

            endpoint = self._get_endpoint()
            url = f"{self.host}{endpoint}"
            headers = {"Content-Type": "application/json"}

            data = {
                "model": self.ollama_name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }

            options = {}
            if self.use_mmap:
                options["use_mmap"] = True
            if self.use_custom_keep_alive_timeout:
                options["keep_alive"] = self.custom_keep_alive_timeout

            if self.has_vision:
                if not self.has_video:
                    encoded_image, e = await self._encode_image(warmup_image_path)
                    if encoded_image == ERROR_TOKEN:
                        await log(f"Error encoding image!: {e}", 'error')
                        raise Exception(f"Error encoding image!: {e}")
                    else:
                        data['messages'][-1]['images'] = [encoded_image]
                elif self.has_video:
                    encoded_frames, e = await self._encode_frames_from_vid(warmup_video_path)
                    if encoded_frames == ERROR_TOKEN:
                        await log(f"Error encoding video!: {e}", 'error')
                        raise Exception(f"Error encoding video!: {e}")
                    else:
                        data['messages'][-1]['images'] = encoded_frames

            data["options"] = options
            data_print = data
            if data["messages"][-1].get("images", False):
                data_print["messages"][-1]['images'] = "[Image Encoded String...]"
            await log(f'Sending request on {url}:\n{data_print}', 'info')
            try:
                async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                    response.raise_for_status()
                    res_json = await response.json()
                    await log(f"recieved:\n{res_json}", 'info')
                    if not ('message' in res_json and 'content' in res_json['message']):
                        raise ValueError("Unexpected API response format during non-streaming test.")
            except (aiohttp.ClientError, ValueError) as e:
                await log(f"🟥 Non-streaming test failed for {self.name}: {e}", "error")
                return

            await log(f'🟩 Non-Streaming works. Trying Streaming...', "success")

            data['stream'] = True
            try:
                async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                    response.raise_for_status()
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode("utf-8")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line.strip():
                                try:
                                    json_line = json.loads(line.strip())
                                    if 'message' in json_line:
                                        await log(json_line, "info")
                                except json.JSONDecodeError:
                                    continue
                        
            except (aiohttp.ClientError, json.JSONDecodeError) as e:
                await log(f"🟥 Streaming test failed for {self.name}: {e}", "error")
                return

            self.warmed_up = True
            await log(f"🟩 [INFO] {self.name} ({self.ollama_name}) warmed up!", "success")

    async def generate(self, query: str, context: list[dict], stream: bool, image_path:None|str=None, mod_ = 1):
        await log(f"Generating response from {self.name}...", "info")
        endpoint = self._get_endpoint()
        url = f"{self.host}{endpoint}"
        messages = [{'role': "system", 'content': self.system}] + context + [{"role": "user", "content": query}]
        headers = {"Content-Type": "application/json"}
        data = {"model": self.ollama_name, "messages": messages, "stream": stream}
        
        options = {}
        if self.use_mmap:
            options["use_mmap"] = True
        if self.use_custom_keep_alive_timeout:
            options["keep_alive"] = self.custom_keep_alive_timeout

        if self.has_vision:
            if image_path is not None:
                ext = os.path.splitext(image_path)[1].lower()
                if ext in IMAGE_EXTs:
                    image, e = await self._encode_image(image_path)
                    if image == ERROR_TOKEN:
                        await log(f"Error encoding image!: {repr(e)}", 'error')
                        yield f"Error encoding image!: {repr(e)}"
                        return
                    else:
                        data['messages'][-1]['images'] = [image]
                elif ext in VIDEO_EXTs and self.has_video:
                    frames, e = await self._encode_frames_from_vid(image_path, mod_)
                    if frames == ERROR_TOKEN:
                        await log(f"Error encoding video!: {repr(e)}", 'error')
                        yield f"Error encoding video!: {repr(e)}"
                        return
                    else:
                        data['messages'][-1]['images'] = frames
        
        if options:
            data['options'] = options
        if self.session:
            try:
                async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                    response.raise_for_status()

                    if stream:
                        buffer = ""
                        async for chunk in response.content.iter_any():
                            buffer += chunk.decode("utf-8")
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    json_line = json.loads(line)
                                    yield json_line.get("message", {}).get("content", "")
                                except json.JSONDecodeError:
                                    continue
                    else:
                        res_json = await response.json()
                        yield res_json.get("message", {}).get("content", "ERROR")

            except Exception as e:
                await log(f"{e}", "error")
                yield f"\n[Error: {e}]"
                yield ERROR_TOKEN
        else:
            await log("Session was not initialised", "error")
            yield "[Error: Session was not initialised]"
            yield ERROR_TOKEN

    async def shutdown(self):
        async with self._start_lock:
            await log(f"Shutting down {self.name}...", "info")
            if self.session is not None:
                await self.session.close()
                self.session = None
            if self.process is not None:
                try:
                    self.process.terminate()
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                except Exception as e:
                    await log(f"Error terminating process for {self.name}: {e}", "error")
                self.process = None
                
            await log(f"{self.name} process terminated.", "success")
