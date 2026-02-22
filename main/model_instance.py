import os
import asyncio
import subprocess
import aiohttp
from .models import Model, IDLE, BUSY, SHUTTING_DOWN, DOWN, WARMING_UP
from .utils import log
from .configs import ERROR_TOKEN
import psutil
import sys
if sys.platform != 'win32':
    from signal import SIGKILL, SIGTERM
else:
    SIGKILL = 9
    from signal import SIGTERM

class LocalModel(Model):
    def __init__(self, role: str, name: str, ollama_name: str, has_tools: bool, has_CoT: bool, has_vision: bool, port: int, system_prompt: str):
        super().__init__(role, name, ollama_name, has_tools, has_CoT, has_vision, port, system_prompt)
        self.start_command = ["ollama", "serve"]
        self.ollama_env = os.environ.copy()
        self.ollama_env["OLLAMA_HOST"] = self.host
      
        self.warmed_up = False
        self.process = None
        self.use_custom_keep_alive_timeout = True
        self.custom_keep_alive_timeout = "-1"
        self.use_mmap = False

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
            await self.change_state(IDLE)
            await log(f"{self.name} is already alive on {self.host}. Re-linked.", "success")
            return

        await self._warmer(use_mmap, warmup_image_path, use_custom_keep_alive_timeout, custom_keep_alive_timeout,
                           has_video_processing, warmup_video_path)

    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                   mod_ = 10, system_prompt_override: str | None = None, options:dict | None = None, format: dict | None = None):

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
                                                                              image_path, mod_, system_prompt_override=system_prompt_override, options=options, format=format):
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
                payload = {
                    'model': self.ollama_name,
                    'messages': [{'role': 'user', 'content': 'bye'}],
                    'options': {'keep_alive': 0}
                }
                async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    resp.raise_for_status()
            except Exception:
                pass 

            await asyncio.sleep(0.5) 
            try:
                await self.session.close()
                await asyncio.sleep(0.5)  # Allow connector cleanup
            except Exception:
                pass
            self.session = None

        if self.process is not None:
            pid = self.process.pid
            await log(f"Cleaning up process tree for {self.name} (PID {pid})...", "info")

            try:
                parent = psutil.Process(pid)

                children = parent.children(recursive=True)
                all_procs = children + [parent]

                for p in all_procs:
                    try:
                        p.terminate()
                    except psutil.NoSuchProcess:
                        pass

                gone, alive = await asyncio.to_thread(psutil.wait_procs, all_procs, timeout=10)

                if alive:
                    await log(f"{len(alive)} processes resisted termination. Escalating to SIGKILL...", "warning")
                    for p in alive:
                        try:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass

            except psutil.NoSuchProcess:
                await log(f"Process {pid} already terminated.", "debug")
            except Exception as e:
                await log(f"Error during paranoid shutdown of {self.name}: {e}", "error")
            finally:
                self.process = None

        if hasattr(self, '_log_file') and self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        await self.change_state(DOWN)
        await log(f"{self.name} shutdown complete.", "success")