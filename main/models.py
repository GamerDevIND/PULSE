import os
import json
import asyncio
import subprocess
import aiohttp
import aiofiles
import av
from utils import log
from configs import ERROR_TOKEN, VIDEO_EXTs, IMAGE_EXTs
import base64
from io import BytesIO

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
        self.tools = []

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

    async def warm_up(self, use_mmap=False, warmup_image_path="main/test.jpg", use_custom_keep_alive_timeout=False, custom_keep_alive_timeout: str = "5m", has_video_processing=False, warmup_video_path="main/test.mp4"):
        self.use_custom_keep_alive_timeout = use_custom_keep_alive_timeout
        self.custom_keep_alive_timeout = custom_keep_alive_timeout
        self.use_mmap = use_mmap
        self.has_video = has_video_processing

        if self.warmed_up:
            if self.process and self.process.poll() is None:
                await log(f"✅ {self.name} process is already running and warmed up. Skipping restart.", "success")
                return

        if self.process is not None:
            await log(f"Cleaning up old terminated process for {self.name}.", "warn")
            try:
                self.process.terminate()
                await asyncio.to_thread(self.process.wait, timeout=1)
            except Exception:
                pass
            self.process = None
            if hasattr(self, '_log_file') and self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None

        if self.session is not None:
            await self.session.close()
            self.session = None

        log_dir = os.path.join("main", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{self.ollama_name}.log")

        await log(f"{self.name} ({self.ollama_name}) warming up...", "info")
        f = None
        try:
            f = open(log_file_path, "w")
            self.process = subprocess.Popen(
                self.start_command, 
                env=self.ollama_env, 
                stderr=subprocess.STDOUT, 
                stdout=f,
            )
            self._log_file = f
            await self.wait_until_ready(self.host)
        except Exception as e:
            await log(f"Failed to start {self.name} on {self.host}: {e}", 'error')
            if self.process is not None:
                try:
                    self.process.terminate()

                except Exception as cleanup_e:
                    await log(f"Error cleaning up process for {self.name}: {cleanup_e}", 'error')
                finally:
                    self.process = None
            if f is not None:
                try:
                    f.close()
                except Exception as close_e:
                    await log(f"Error closing log file for {self.name}: {close_e}", 'error')
                self._log_file = None
            raise

        if not self.session:
            self.session = aiohttp.ClientSession()

        endpoint = self._get_endpoint()
        url = f"{self.host}{endpoint}"
        headers = {"Content-Type": "application/json"}

        data = {
            "model": self.ollama_name,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False
        }

        options = {}
        if self.use_mmap:
            options["use_mmap"] = True
        if self.use_custom_keep_alive_timeout:
            options["keep_alive"] = self.custom_keep_alive_timeout

        if self.role.lower() == 'router':
            data["format"] = {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["chat", "cot"], "description": "Selected role or model"}
                },
                "required": ["role"]
            }

        if self.has_vision:
            if self.has_video:
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
                await log("Vision model, but no video processing enabled and no warmup image found. Skipping vision test.", 'warn')

        data["options"] = options

        await log(f'Sending non-streaming warm-up request on {url}', 'info')
        try:
            async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:
                response.raise_for_status()
                res_json = await response.json()
                if not ("message" in res_json and "content" in res_json["message"]):
                    raise ValueError("Unexpected API response format during non-streaming test.")
        except (aiohttp.ClientError, ValueError, Exception) as e:
            await log(f"🟥 Non-streaming test failed for {self.name}: {e}", "error")
            if self.session is not None:
                await self.session.close()
                self.session = None
            return

        await log(f'🟩 Non-Streaming works. Trying Streaming...', "success")

        data["stream"] = True

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
                                if "message" in json_line and "content" in json_line["message"]:
                                    await log(f'Streaming test chunk received: {json_line["message"]["content"][:30]}...', "info")
                            except json.JSONDecodeError:
                                continue
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            await log(f"Streaming test failed for {self.name}: {e}", "error")
            return

        self.warmed_up = True
        await log(f"{self.name} ({self.ollama_name}) warmed up!", "success")
        
    async def add_tools(self, *tool_dict):
        self.tools.extend(tool_dict)

    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, mod_ = 1):
        await log(f"Generating response from {self.name}...", "info")
        await log(f"Model {self.name} has_tools={self.has_tools}, tools count={len(self.tools)}", "info")
        endpoint = self._get_endpoint()
        url = f"{self.host}{endpoint}"

        messages = []
        if self.system:
            messages.append({'role': "system", 'content': self.system})
        messages += context + [{"role": "user", "content": query}]

        headers = {"Content-Type": "application/json"}

        data = {
            "model": self.ollama_name,
            "messages": messages,
            "stream": stream
        }

        if self.role.lower() == 'router':
            data["format"] = {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["chat", "cot"], "description": "Selected role or model"}
                },
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
                    yield (None, f"Error encoding image!: {repr(e)}", None)
                    return
                else:
                    data['messages'][-1]['images'] = [image]
            elif ext in VIDEO_EXTs and self.has_video:
                frames, e = await self._encode_frames_from_vid(image_path, mod_)
                if frames == ERROR_TOKEN:
                    await log(f"Error encoding video!: {repr(e)}", 'error')
                    yield (None, f"Error encoding video!: {repr(e)}", None)
                    return
                else:
                    data['messages'][-1]['images'] = frames
            elif ext in VIDEO_EXTs and not self.has_video:
                await log(f"Cannot process video file {image_path}: Model {self.name} is not configured for video processing.", 'warn')

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
            yield (None, f"\n[Error: {type(e).__name__}: {e}]", None)

    async def shutdown(self):
        await log(f"Shutting down {self.name}...", "info")
        if self.session is not None:
            try:
                await self.session.close()
            except Exception as e:
                await log(f"Error closing session for {self.name}: {e}", 'error')
            self.session = None
        if self.process is not None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    await asyncio.to_thread(self.process.wait, timeout=2)
            except Exception as e:
                await log(f"Error terminating process for {self.name}: {e}", 'error')
            finally:
                self.process = None
        if hasattr(self, '_log_file') and self._log_file is not None:
            try:
                self._log_file.close()
            except Exception as e:
                await log(f"Error closing log file for {self.name}: {e}", 'error')
            self._log_file = None
        await log(f"{self.name} process terminated.", "success")
