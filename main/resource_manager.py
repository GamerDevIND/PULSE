import aiohttp
import subprocess
import psutil
from .utils import log
import asyncio
import os
import sys

class SessionManager:
    def __init__(self, model_name) -> None:
        self.session:None | aiohttp.ClientSession = None
        self.model_name = model_name
    
    def create_session(self,):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def shutdown(self, send_shutdown_paylod = True, url= None, headers:dict | None = None, payload=None, set_session_to_None=False):
        if self.session is not None:
            if not send_shutdown_paylod:
                try:
                    await self.session.close()
                except Exception as e:
                    await log(f"An error occurred during session cleanup: {e}", "error")
                
                if set_session_to_None and (self.session and (not self.session.closed)):
                    try:
                        await self.session.close()
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        await log(f"An error occurred during session closing: {e}", "error") 
                    self.session = None
                
                return

            if url is None:
                await log("url cannot be none during session cleanup, aborting...", 'warn')
                return
            try:
                async with self.session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    resp.raise_for_status()
            except Exception as e:
                await log(f"An error occurred during session cleanup: {e}", "error") 

            await asyncio.sleep(1.5) 
            try:
                await self.session.close()
            except Exception as e:
                await log(f"An error occurred during session cleanup: {e}", "error")

        if set_session_to_None and (self.session and (not self.session.closed)):
            try:
                await self.session.close()
                await asyncio.sleep(0.5)
            except Exception as e:
                await log(f"An error occurred during session closing: {e}", "error") 
            self.session = None

    async def wait_until_ready(self, url: str, timeout: int = 30):
        await log(f"Waiting for {self.model_name} on {url}...", "info")
        try:
            check_session = self.session if self.session and not self.session.closed else aiohttp.ClientSession()
            for i in range(timeout):
                try:
                    async with check_session.get(f"{url}/api/tags") as res:
                        if res.status == 200:
                            await log(f"{self.model_name} is alive! Testing response...", "success")
                            js = await res.json(encoding='utf-8')
                            models = [m['name'] for m in js.get("models", [])]
                            if self.model_name in models or f'{self.model_name}:latest' in models:
                                    
                                await log(f'{self.model_name} is online!', 'info')
                                return
                            else:
                                await log(f"Model not found in the HTTP model library! Avaliable models: {', '.join(models)}", 'warn')
                                raise ModuleNotFoundError("Model not found in the HTTP model library")
                            
                                
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    await log(f"Retries: {i+1} / {timeout}", "info")
                await asyncio.sleep(1)
            raise TimeoutError(f"🟥 Ollama server for {self.model_name} did not start in time.")
        finally:
            if check_session is not self.session:
                await check_session.close()

class ResourceManager(SessionManager):
    def __init__(self, ref_name) -> None:
        self.ref_name = ref_name
        self.process = None
        self._log_file = None
        super().__init__(self.ref_name)

    def create_process(self, command, env):
        try:
            log_dir = os.path.join("main", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f"{self.ref_name}.log")

            f = open(log_file_path, "w")
            self._log_file = f

            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )
            start_new_session = sys.platform != "win32"

            self.process = subprocess.Popen(
                    command,
                    env=env,
                    stderr=subprocess.STDOUT,
                    stdout=f,
                    creationflags=creationflags,
                    start_new_session=start_new_session,
            )
        except Exception as e:
            if hasattr(self, '_log_file') and self._log_file is not None:
                try:
                    self._log_file.flush()
                    self._log_file.close()
                except Exception as e:
                    print(f"An error occurred during process creation: {e}")
                self._log_file = None 

    async def shutdown(self, send_shutdown_paylod = True, url = None, headers:dict|None = None, payload=None, session_cleanup= True, process_cleanup=False, set_session_to_None = True):
        if session_cleanup:
            if send_shutdown_paylod and (url is None or payload is None):
                await log("url or paylaod cannot be none during session cleanup, aborting...", 'warn')
                return
            
            await super().shutdown(send_shutdown_paylod, url, headers, payload, set_session_to_None)

        if process_cleanup:
            if self.process is not None:
                pid = self.process.pid
                await log(f"Cleaning up process tree for {self.ref_name} (PID {pid})...", "info")

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
                    await log(f"Error during shutdown of {self.ref_name}: {e}", "error")
                finally:
                    self.process = None

        if hasattr(self, '_log_file') and self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception as e:
                await log(f"An error occurred during process file cleanup: {e}", "info")
            self._log_file = None 