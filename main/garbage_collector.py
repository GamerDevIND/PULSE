import os
import asyncio
import datetime

class GarbageCollector:
    def __init__(self, image_cache_dir, video_cache_dir, time_limit, size_threshold_in_mbs) -> None:
        self.image_cache_dir = image_cache_dir
        self.video_cache_dir = video_cache_dir
        self.time_limit = time_limit
        self.size_threshold_in_mbs = size_threshold_in_mbs * 1024 * 1024

    
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
                    await asyncio.to_thread(os.remove, data['path'])
                keys_to_delete.append(hash_key)
            
            elif data['size'] > self.size_threshold_in_mbs and (now - data['last_used']) > (self.time_limit // 2):
                if os.path.exists(data['path']):
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
                    await asyncio.to_thread(os.remove, index[k]['path'])
                    current_size -= index[k]['size']
                    del index[k]
                except: pass

        return index