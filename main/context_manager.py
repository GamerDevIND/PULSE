from .utils import log
from .configs import IMAGE_EXTs, VIDEO_EXTs, FILE_NAME_KEY
import json 
import asyncio
import aiofiles 
import os
import shutil
from copy import deepcopy
import datetime
import hashlib
from .model_instance import LocalModel
from .models_profile import RemoteModel
from .garbage_collector import GarbageCollector
from .summariser import Summariser

class ContextManager:
    def __init__(self, context_path, summary_model: LocalModel | RemoteModel | None,summary_max_tokens = 4000, keep_tokens_after_summary = 2000, 
                 min_recent_turns = 1, cache_folder = './cache', 
                 gc_time_limit = 259200, gc_limit_size_MBs = 50, gc_interval = 1800):
        self.context_path = context_path

        self.summary = None
        self.facts = None
        self.context = []

        self.lock = asyncio.Lock()

        self.running_tasks = set()
        self.gc_interval = gc_interval

        self.summariser = Summariser(summary_model,  summary_max_tokens, keep_tokens_after_summary, min_recent_turns)

        self.cache_dir = cache_folder
        self.cache_index_file = os.path.join(self.cache_dir, 'index.json')
        self.images_dir = os.path.join(self.cache_dir, "images")
        self.videos_dir = os.path.join(self.cache_dir, "videos")

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        self.cache_index = {}

        self.queue = asyncio.Queue()
        
        self.gc = GarbageCollector(self.images_dir, self.videos_dir, gc_time_limit, gc_limit_size_MBs) # idk what to do abt the time_limit

    async def init(self):
        check_task = asyncio.create_task(self._garbage_collect())
        self.running_tasks.add(check_task)
        await self.load()

    async def _garbage_collect(self):
        while True:
            await asyncio.sleep(self.gc_interval)
            idx = await self.gc.gc(deepcopy(self.cache_index))
            async with self.lock:
                self.cache_index = idx
            await self.save()

    def _hash_file(self, file_path, sample_size=1024 * 1024):
        file_size = os.path.getsize(file_path)
        sha256_hash = hashlib.sha256()

        sha256_hash.update(str(file_size).encode())

        with open(file_path, "rb") as f:
            if file_size <= sample_size * 3:
                sha256_hash.update(f.read())
            else:
                sha256_hash.update(f.read(sample_size))

                f.seek(file_size // 2)
                sha256_hash.update(f.read(sample_size))

                f.seek(-sample_size, 2)
                sha256_hash.update(f.read(sample_size))

        return sha256_hash.hexdigest()

    async def cache_file(self, file_path:str, skip_if_missing = True):
        if not (file_path and os.path.exists(file_path)):
            await log(F"{file_path} doesn't exist!", 'error')
            if skip_if_missing:
                return
            else:
                raise FileNotFoundError(F"{file_path} doesn't exist!")
        
        dest = None

        ext = os.path.splitext(file_path)[-1]

        if ext in IMAGE_EXTs:
            dest = self.images_dir
        elif ext in VIDEO_EXTs:
            dest = self.videos_dir
        else:
            await log(F"{file_path} isn't a supported format!", 'error')
            if skip_if_missing:
                return
            else:
                raise FileNotFoundError(f"{file_path} isn't a supported format!")
        
        if not dest:
            await log(f"An error occurred processing the filename of {file_path}", 'error')
            return
        
        try:
            hashed = await asyncio.to_thread(self._hash_file, file_path)
        except Exception as e:
            await log(f"An error occurred while hashing: {repr(e)}", 'error')
            hashed = None

        if not hashed:
            await log(f"Hashing for {file_path} failed.", 'error')
            if skip_if_missing:
                return
            else:
                raise Exception(f"Hashing for {file_path} failed.")
        
        ext = f".{ext}" if not ext.startswith('.') else ext

        cached = f"{hashed}{ext}"

        dest = os.path.join(dest, cached)

        d = datetime.datetime.now()

        async with self.lock:
            if not os.path.exists(dest):
                await asyncio.to_thread(shutil.copy, src=file_path, dst=dest)
                file_size = await asyncio.to_thread(os.path.getsize, dest)
            else:
                self.cache_index[hashed]['last_used'] = d.timestamp()
                file_size = self.cache_index[hashed]['size']

            self.cache_index[hashed] = {'path': dest, "last_used": d.timestamp(), 'size': file_size}

        await self.save()
        return dest

    async def file_path_resolver(self, file_path, auto_cache = True):
        try:
            hashed = await asyncio.to_thread(self._hash_file, file_path)
        except Exception as e:
            await log(f"An error occurred while hashing: {repr(e)}", 'error')
            hashed = None

        if not hashed:
            await log(f"Hashing for {file_path} failed.", 'error')
            raise Exception(f"Hashing for {file_path} failed.")

        async with self.lock:
            if hashed in self.cache_index:
                self.cache_index[hashed]["last_used"] = datetime.datetime.now().timestamp()
                return self.cache_index[hashed]['path']
                
        if auto_cache:
            return await self.cache_file(file_path) 
        else:
            raise FileNotFoundError(f"{file_path} isn't in the cache")

    async def context_resolver(self, context, file_name_key=FILE_NAME_KEY, max_keeps=1, auto_save=True):
        async with self.lock:
            await self.flush_queue()
            
            context = list(reversed(deepcopy(context)))
            
            found_files = 0
            for c in context:
                if file_name_key in c:
                    found_files += 1
                    if found_files > max_keeps:
                        file = c[file_name_key]
                        del c[file_name_key]
                        c["content"] += f"\n\n[System: Previous media '{file}' has been removed from memory by the system. User had upload that media previously during the conversation]"

            context = list(reversed(context))
            if auto_save:
                self.context = context

        if auto_save:
            await self.save()
            
        return context

    async def append(self, data:dict | list[dict] | tuple[dict]): 
        if type(data) == dict:
            await self.queue.put(data)
        elif type(data) in [list, tuple]:
            for d in data: await self.queue.put(d)
        else: await log("unknown data type, skipping item.", "warn")

    async def add_and_maintain(self, data:dict | list[dict] | tuple[dict]):
        await self.append(data)
        await self.flush_queue()

        r = await self.summariser.maybe_summarise_context(self.context, self.summary, self.facts)
        if r:
            s, f, c = r
            async with self.lock:
                self.summary = s
                self.facts = f
                self.context = c

        await self.save()

    async def flush_queue(self):
        items = []

        while not self.queue.empty():
            item = await self.queue.get()
            items.append(item)
        async with self.lock:
            self.context.extend(items)

    async def get_context(self):
        async with self.lock:
            return deepcopy(list(self.context))

    async def get_summary(self):
        async with self.lock:
            return str(self.summary) if self.summary is not None else ""

    async def get_facts(self):
        async with self.lock:
            return deepcopy(self.facts) if self.facts is not None else ""

    async def shut_down(self):  
        index = await self.gc.gc(deepcopy(self.cache_index))
        self.cache_index = index
        await self.flush_queue()
        await self.save()
        for task in list(self.running_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await log("Context saved and context manager shut down.", "info")

    async def load(self):
        try:
            async with aiofiles.open(self.context_path) as file:
                content = await file.read()
                content = json.loads(content)
                self.summary = content.get("summary")
                self.facts = content.get("facts")
                self.context = content.get("conversation",[])
        except (FileNotFoundError, json.JSONDecodeError):
            await log("Context missing or corrupted. Starting over.", "warn")
            self.context = [] 
            self.summary = None
            self.facts = None
        
        try:
            async with aiofiles.open(self.cache_index_file) as file:
                content = await file.read()
                content = json.loads(content)
                self.cache_index = content
        except (FileNotFoundError, json.JSONDecodeError):
            await log("Cache index missing or corrupted. Starting over.", "warn")
            self.cache_index = {}

    async def save(self):
        try:
            await self.flush_queue()
            async with self.lock:
                data = {"summary":self.summary, "facts": self.facts, "conversation": self.context}
                data_to_save = json.dumps(data, indent=2)

            async with aiofiles.open(self.context_path, "w") as file:
                await file.write(data_to_save)

            data_to_save = json.dumps(self.cache_index, indent=2)

            async with aiofiles.open(self.cache_index_file,'w') as f:
                await f.write(data_to_save)

        except IOError as e:
            await log(f"Error saving context: {e}", "error")