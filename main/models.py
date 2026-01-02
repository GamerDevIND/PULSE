import os
import json
import asyncio
import subprocess
import aiohttp
import aiofiles
import av
from .utils import log, Tool
from .configs import ERROR_TOKEN, VIDEO_EXTs, IMAGE_EXTs
import base64
from io import BytesIO
import sys
from signal import SIGKILL

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
        self.port = port
        self.system = system_prompt
        self.host = f"http://localhost:{self.port}"
        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = self.host
        self.warmed_up = False
        self.session: aiohttp.ClientSession | None = None
        self.process = None
        self.use_custom_keep_alive_timeout = False
        self.custom_keep_alive_timeout = "5m"
        self.use_mmap = False
        self.has_video = False
        self.available_roles= [] # ONLY accessed by the router model
        self.tools = []
        self.state = IDLE
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        await self.warm_up()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    def _get_endpoint(self) -> str:
        return "/api/chat"

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

                if self.process is not None and self.process.poll() is not None:
                    try:
                        if hasattr(self, '_log_file') and self._log_file is not None and hasattr(self._log_file, 'name'):
                            self._log_file.flush()
                            try:
                                async with aiofiles.open(self._log_file.name, 'r') as lf:
                                    content = await lf.read()
                                    tail = '\n'.join(content.splitlines()[-30:])
                                    await log(f"{self.name} process terminated. Recent logs:\n{tail}", 'error')
                            except (FileNotFoundError, PermissionError, OSError) as log_e:
                                await log(f"Could not read log file for {self.name}: {log_e}", 'error')
                    except (OSError, AttributeError) as proc_e:
                        await log(f"Error checking process status for {self.name}: {proc_e}", 'error')
            await asyncio.sleep(1)
        raise TimeoutError(f"🟥 Ollama server for {self.name} did not start in time.")

    async def _encode_frames_from_vid(self, video_path, mod_ = 1):
        if not self.has_vision:
            return [], None
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
                    img.save(buffer, format="PNG")
                    frame_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    frames.append(frame_b64)
            return frames, None
        except Exception as e:
            await log(f"Error in video frame encoding for {self.name}: {e}", 'error')
            return ERROR_TOKEN, e

    async def _encode_image(self, image_path):
        try:
            async with aiofiles.open(image_path, "rb") as image_file:
                image_data = await image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            return base64_image, None
        except Exception as e:
            return ERROR_TOKEN, e

    def add_to_available_roles(self, *roles):
        roles = list(roles)
        self.available_roles.extend(roles)
        self.available_roles= list(set(self.available_roles)) # order doesn't matter 
   
    async def warm_up(
        self,
        use_mmap=False,
        warmup_image_path="main/test.jpg",
        use_custom_keep_alive_timeout=False,
        custom_keep_alive_timeout: str = "5m",
        has_video_processing=False,
        warmup_video_path="main/test.mp4",):
        async with self.lock:
            if self.state not in (IDLE, DOWN):
                raise RuntimeError(f"{self.name} cannot warm up from state={self.state}")
            self.state = WARMING_UP

        try:
            self.use_custom_keep_alive_timeout = use_custom_keep_alive_timeout
            self.custom_keep_alive_timeout = custom_keep_alive_timeout
            self.use_mmap = use_mmap
            self.has_video = has_video_processing

            is_actually_alive = False
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as session:
                    async with session.get(f"{self.host}/api/tags") as res:
                        if res.status == 200:
                            is_actually_alive = True
            except Exception:
                pass

            if is_actually_alive:
                if not self.session or self.session.closed:
                    self.session = aiohttp.ClientSession()
                self.warmed_up = True
                await log(f"{self.name} is already alive on {self.host}. Re-linked.", "success")
                return
            if self.warmed_up and self.process and self.process.poll() is None:
                await log(
                    f"{self.name} process already running and warmed. Skipping restart.",
                    "success",
                )
                return

            log_dir = os.path.join("main", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f"{self.ollama_name}.log")

            f = open(log_file_path, "w")
            self._log_file = f

            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            start_new_session = sys.platform != "win32"

            self.process = subprocess.Popen(
                self.start_command,
                env=self.ollama_env,
                stderr=subprocess.STDOUT,
                stdout=f,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )

            await self.wait_until_ready(self.host)

            if not self.session:
                self.session = aiohttp.ClientSession()

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

            if self.role.lower() == 'router':
                if len(self.available_roles) < 1:
                    await log("No role provided", "error") 
                    raise Exception("No role provided")
                data["format"] = {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": self.available_roles, "description": "Selected role or model"}
                    }, # I'll add more
                    "required": ["role"]
                }

            if self.has_vision:
                if self.has_video and os.path.exists(warmup_video_path):
                    await log("Warming up with video frame extraction...", 'info')
                    encoded_frames, e = await self._encode_frames_from_vid(warmup_video_path, mod_=10)
                    if encoded_frames == ERROR_TOKEN:
                        await log(f"Error encoding video frames for warmup: {e}", 'error')
                        raise Exception(f"Error encoding video frames for warmup: {e}")
                    if encoded_frames:
                        if "messages" in data:
                            data['messages'][-1]['images'] = encoded_frames
                elif os.path.exists(warmup_image_path):
                    await log("Warming up with static image...", 'info')
                    encoded_image, e = await self._encode_image(warmup_image_path)
                    if encoded_image == ERROR_TOKEN:
                        await log(f"Error encoding image for warmup!: {e}", 'error')
                        raise Exception(f"Error encoding image for warmup!: {e}")
                    else:
                        if "messages" in data:
                            data['messages'][-1]['images'] = [encoded_image]
                else:
                    await log("Vision model, but no video processing enabled and no warmup image / video found. Skipping vision test.", 'warn')

            data["options"] = options

            async with self.session.post(
                f"{self.host}{self._get_endpoint()}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(data),
            ) as response:
                response.raise_for_status()

            self.warmed_up = True
            await log(f"{self.name} ({self.ollama_name}) warmed up!", "success")

        except Exception:
            self.warmed_up = False
            raise

        finally:
            async with self.lock:
                self.state = IDLE if self.warmed_up else DOWN

        
    async def add_tools(self, *tool_dicts):
        for tool in tool_dicts:
            if isinstance(tool, Tool):
                self.tools.append(tool.schema)
            else:
                self.tools.append(tool)

    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                       mod_ = 1, system_prompt_override: str | None = None):
        
        async with self.lock:
            if self.state != IDLE:
                await log(f"{self.name} is busy", "warn")
                yield (None, ERROR_TOKEN, None)
                return
            self.state = BUSY


        await log(f"Generating response from {self.name}...", "info")

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
                
        messages += context + [{"role": "user", "content": query}]

        headers = {"Content-Type": "application/json"}

        data = {
            "model": self.ollama_name,
            "messages": messages,
            "stream": stream
        }

        if self.role.lower() == 'router':
            if len(self.available_roles) < 1:
                await log("No role provided", "error") 
                raise  Exception("No role provided")


            data["format"] = {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": self.available_roles, "description": "Selected role or model"}
                }, # I'll add more
                "required": ["role"]
                }
        elif self.has_tools:
            data["tools"] = self.tools

        options = {}
        if self.use_mmap:
            options["use_mmap"] = True
        if self.use_custom_keep_alive_timeout:
            options["keep_alive"] = self.custom_keep_alive_timeout
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
                    yield (None, ERROR_TOKEN, None)
                    return
                else:
                    data['messages'][-1]['images'] = [image]
            elif ext in VIDEO_EXTs and self.has_video:
                frames, e = await self._encode_frames_from_vid(image_path, mod_)
                if frames == ERROR_TOKEN:
                    await log(f"Error encoding video!: {repr(e)}", 'error')
                    yield (None, ERROR_TOKEN, None)
                    return
                else:
                    data['messages'][-1]['images'] = frames
            elif ext in VIDEO_EXTs and not self.has_video:
                await log(f"Cannot process video file {image_path}: Model {self.name} is not configured for video processing.", 'warn')

        async with self.lock:
            if self.state != BUSY:
                yield (None, ERROR_TOKEN, None)
                return

        buffer = ""
        try:
            async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:  # type: ignore
                response.raise_for_status()

                if stream:
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode("utf-8")
                        while "\n" in buffer:
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
                else:
                    res_json = await response.json()
                    thinking = res_json.get("message", {}).get("thinking", "")
                    content = res_json.get("message", {}).get("content", "")
                    tools = res_json.get("message", {}).get("tool_calls", [])
                    yield (thinking, content, tools)

        except Exception as e:
            await log(f"Ollama API Request Error: {e}", "error")
            yield (None, ERROR_TOKEN, None)
        finally:
            async with self.lock:
                if self.state == BUSY:
                    self.state = IDLE


    async def shutdown(self):
        await log(f"Shutting down {self.name}...", "info")
        async with self.lock:
            if self.state in (DOWN, SHUTTING_DOWN):
                return
            self.state = SHUTTING_DOWN

        if self.session is not None:
            try:
                url = f'{self.host}{self._get_endpoint()}'
                await self.session.post(url, json={'model': self.ollama_name, 'keep_alive': 0})
            except Exception:
                pass 
            await asyncio.sleep(0.5) # wait for the request to be processed
            try:
                await self.session.close()
            except:
                pass
            self.session = None

        if self.process is not None:
            pid = self.process.pid
            await log(f"Killing process tree for {self.name} (PID {pid})...", "info")

            try: # you can call me paranoid. I AM paranoid.
                self.process.terminate() 
                await asyncio.to_thread(self.process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                await log(f"Error terminating process for {self.name}: {e}", "error")

            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    os.killpg(os.getpgid(pid), SIGKILL)
            except Exception:
                pass

            self.process = None

        if hasattr(self, '_log_file') and self._log_file is not None:
            try:
                self._log_file.close()
            except:
                pass
            self._log_file = None

        async with self.lock:
            self.state = DOWN
        await log(f"{self.name} shutdown complete.", "success")
