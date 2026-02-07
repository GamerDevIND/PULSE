import os
import json
import asyncio
import subprocess
import aiohttp
import aiofiles
from .models import Model, IDLE, BUSY, SHUTTING_DOWN, DOWN, WARMING_UP
from .utils import log
from .configs import ERROR_TOKEN
import sys
if sys.platform != 'win32':
    from signal import SIGKILL
else:
    SIGKILL = 9

class LocalModel(Model):
    def __init__(self, role: str, name: str, ollama_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, port: int, system_prompt: str):
        self.role = role
        self.name = name
        self.ollama_name = ollama_name
        self.has_tools = has_tools
        self.has_CoT = has_CoT
        self.has_vision = has_vision
        self.port = port
        self.router_role = "router"
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
        self.available_roles = [] # ONLY accessed by the router model
        self.tools = []
        self.state = DOWN

    async def __aenter__(self):
        await self.warm_up()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def warm_up(
        self,
        use_mmap=False,
        warmup_image_path="main/test.jpg",
        use_custom_keep_alive_timeout=False,
        custom_keep_alive_timeout: str = "5m",
        has_video_processing=False,
        warmup_video_path="main/test.mp4", router_role = "router"):
        async with self.state_lock:
            if self.state not in (IDLE, DOWN):
                raise RuntimeError(f"{self.name} cannot warm up from state={self.state}")
            
        await self.change_state(WARMING_UP)

        try:
            self.use_custom_keep_alive_timeout = use_custom_keep_alive_timeout
            self.custom_keep_alive_timeout = custom_keep_alive_timeout
            self.use_mmap = use_mmap
            self.router_role = router_role
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

            headers = {"Content-Type": "application/json"}
            endpoint = self._get_endpoint()
            url = f"{self.host}{endpoint}"

            options = {}
            if self.use_mmap:
                options["use_mmap"] = True
            if self.use_custom_keep_alive_timeout:
                options["keep_alive"] = self.custom_keep_alive_timeout

            if self.role.lower() == self.router_role:
                available_roles = self.available_roles
                if len(available_roles) < 1:
                    await log("No role provided. Testing with fallback roles.", "error")
                    available_roles = ["chat", 'cot']
                    
                data["format"] = {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": available_roles, "description": "Selected role or model"}
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

            await log('Trying Non-Streaming...' ,'info')
            async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:  # type: ignore
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
                async with self.session.post(url, headers=headers, data=json.dumps(data)) as response:  # type: ignore
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
            await log(f"{self.name} ({self.ollama_name}) warmed up!", "success")

        except Exception:
            self.warmed_up = False
            raise

        finally: 
            await self.change_state(IDLE if self.warmed_up else DOWN)
    
    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                   mod_ = 10, system_prompt_override: str | None = None):
    
        async with self.state_lock:
            if self.state != IDLE:
                await log(f"{self.name} is busy", "warn")
                yield (ERROR_TOKEN, ERROR_TOKEN, [])
                return
            await self.change_state(BUSY, False) 
            self.generation_cancelled = False

        await log(f"Generating response from {self.name}...", "info")


        options = {}
        if self.use_mmap:
            options["use_mmap"] = True
        if self.use_custom_keep_alive_timeout:
            options["keep_alive"] = self.custom_keep_alive_timeout
        try:
            async for chunk in self._generator(query, context, stream, think, 
                                                                              image_path, mod_, system_prompt_override, options):
                yield chunk

        finally:
            async with self.state_lock:
                self.generation_cancelled = False
                if self.state == BUSY:
                    await self.change_state(IDLE, False)

    async def shutdown(self):
        await log(f"Shutting down {self.name}...", "info")
        async with self.state_lock:
            if self.state in (DOWN, SHUTTING_DOWN):
                return
        await self.change_state(SHUTTING_DOWN)

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
                if sys.platform == 'win32':
                    os.kill(pid, SIGKILL)
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

        await self.change_state(DOWN)
        await log(f"{self.name} shutdown complete.", "success")
