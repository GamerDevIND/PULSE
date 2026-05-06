import asyncio
import hashlib
import os
import json
import aiofiles
import shutil
import datetime
from .configs import IMAGE_EXTs, VIDEO_EXTs, FILE_NAME_KEY
from .utils import Logger
from copy import deepcopy
from .events import EventBus

class GarbageCollector:
    def __init__(self, image_cache_dir, video_cache_dir, time_limit, size_threshold_in_mbs, event_bus: None | EventBus = None) -> None:
        self.image_cache_dir = image_cache_dir
        self.video_cache_dir = video_cache_dir
        self.time_limit = time_limit
        self.size_threshold_in_mbs = size_threshold_in_mbs * 1024 * 1024
        self.event_bus = event_bus
    
    async def gc(self, index):
        image_dir_list = os.listdir(self.image_cache_dir)
        image_dir_list = list(map(lambda f: os.path.join(self.image_cache_dir, f), image_dir_list))
        video_dir_list = os.listdir(self.video_cache_dir)
        video_dir_list = list(map(lambda f: os.path.join(self.video_cache_dir, f), video_dir_list))

        total_dir = image_dir_list + video_dir_list

        indexed_paths = {item['path'] for item in index.values()}
        keys_to_delete = []

        now = datetime.datetime.now().timestamp()

        for full_path in total_dir:
            if full_path not in indexed_paths:
                await asyncio.to_thread(os.remove, full_path)

        for hash_key, data in index.items():
            if (now - data['last_used']) > self.time_limit:
                if os.path.exists(data['path']):
                    if self.event_bus:
                        await self.event_bus.sequence_emit(self.event_bus.GARBAGE_COLLECTOR, path = data['path'])
                    await asyncio.to_thread(os.remove, data['path'])
                keys_to_delete.append(hash_key)
            
            elif data['size'] > self.size_threshold_in_mbs and (now - data['last_used']) > (self.time_limit // 2):
                if os.path.exists(data['path']):
                    if self.event_bus:
                        await self.event_bus.sequence_emit(self.event_bus.GARBAGE_COLLECTOR, path = data['path'])
                    await asyncio.to_thread(os.remove, data['path'])
                keys_to_delete.append(hash_key)
            
        for k in keys_to_delete:
            del index[k]

        global_size_threshold = self.size_threshold_in_mbs * 100

        current_size = sum(item['size'] for item in index.values())
        if current_size > global_size_threshold:
            sorted_keys = sorted(index.keys(), key=lambda k: index[k]['last_used'])
            
            for k in sorted_keys:
                if current_size <= global_size_threshold:
                    break
                try:
                    if self.event_bus:
                        await self.event_bus.sequence_emit(self.event_bus.GARBAGE_COLLECTOR, path = index[k]['path'])
                    await asyncio.to_thread(os.remove, index[k]['path'])
                    current_size -= index[k]['size']
                    del index[k]
                except Exception as e: await Logger.log_async(f"Error deleting: {index[k]['path']}: {repr(e)}", "error")
        if self.event_bus:
            await self.event_bus.sequence_emit(self.event_bus.GARBAGE_COLLECTED)
        return index

class CacheManager:
    def __init__(self, gc_time_limit, gc_limit_size_MBs, gc_interval, cache_folder, event_bus: None | EventBus = None) -> None:
        self.gc_interval = gc_interval
        self.lock = asyncio.Lock()
        self.cache_dir = cache_folder
        self.cache_index_file = os.path.join(self.cache_dir, 'index.json')
        self.images_dir = os.path.join(self.cache_dir, "images")
        self.videos_dir = os.path.join(self.cache_dir, "videos")

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        self.running_tasks = set()

        self.gc = GarbageCollector(self.images_dir, self.videos_dir, gc_time_limit, gc_limit_size_MBs)

        self.cache_index = {}
        self.event_bus = event_bus

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
            await Logger.log_async(F"{file_path} doesn't exist!", 'error')
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
            await Logger.log_async(F"{file_path} isn't a supported format!", 'error')
            if skip_if_missing:
                return
            else:
                raise FileNotFoundError(f"{file_path} isn't a supported format!")
        
        if not dest:
            await Logger.log_async(f"An error occurred processing the filename of {file_path}", 'error')
            return
        
        try:
            hashed = await asyncio.to_thread(self._hash_file, file_path)
        except Exception as e:
            await Logger.log_async(f"An error occurred while hashing: {repr(e)}", 'error')
            hashed = None

        if not hashed:
            await Logger.log_async(f"Hashing for {file_path} failed.", 'error')
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
    
    async def context_resolver(self, context, file_name_key=FILE_NAME_KEY, max_keeps=1):
        resolved_context = deepcopy(context) 
        found_files = 0
        l = []

        for msg in reversed(resolved_context):
            if file_name_key in msg and msg[file_name_key]:
                found_files += 1
                if found_files > max_keeps:
                    file_name = msg.pop(file_name_key, "Unknown File")
                    note = f"\n\n[System: Media '{file_name}' removed from active memory.]"
                    if isinstance(msg["content"], str):
                        msg["content"] += note
                    elif isinstance(msg["content"], list):
                        msg["content"].append({"type": "text", "text": note})
            
            l.append(msg)

        return list(reversed(l))
    
    async def file_path_resolver(self, file_path, auto_cache = True):
        try:
            hashed = await asyncio.to_thread(self._hash_file, file_path)
        except Exception as e:
            await Logger.log_async(f"An error occurred while hashing: {repr(e)}", 'error')
            hashed = None

        if not hashed:
            await Logger.log_async(f"Hashing for {file_path} failed.", 'error')
            raise Exception(f"Hashing for {file_path} failed.")

        async with self.lock:
            if hashed in self.cache_index:
                self.cache_index[hashed]["last_used"] = datetime.datetime.now().timestamp()
                return self.cache_index[hashed]['path']
                
        if auto_cache:
            return await self.cache_file(file_path) 
        else:
            raise FileNotFoundError(f"{file_path} isn't in the cache")

    async def save(self):
        data_to_save = json.dumps(self.cache_index, indent=2)

        async with aiofiles.open(self.cache_index_file,'w') as f:
            await f.write(data_to_save)

    async def load(self):
        try:
            async with aiofiles.open(self.cache_index_file) as file:
                content = await file.read()
                content = json.loads(content)
                self.cache_index = content
        except (FileNotFoundError, json.JSONDecodeError):
            await Logger.log_async("Cache index missing or corrupted. Starting over.", "warn")
            self.cache_index = {}

    async def shutdown(self):
        index = await self.gc.gc(deepcopy(self.cache_index))
        self.cache_index = index
        await self.save()
        for task in list(self.running_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass