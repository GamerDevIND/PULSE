from .utils import log
from .configs import FILE_NAME_KEY
import json 
import asyncio
import aiofiles 
from copy import deepcopy
from .model_instance import LocalModel
from .models_profile import RemoteModel
from .summariser import Summariser
from .cache_manager import CacheManager

class ContextManager:
    def __init__(self, context_path, summary_model: LocalModel | RemoteModel | None,summary_max_tokens = 4000, keep_tokens_after_summary = 2000, 
                 min_recent_turns = 3, cache_folder = './cache', 
                 gc_time_limit = 259200, gc_limit_size_MBs = 50, gc_interval = 1800):
        self.context_path = context_path

        self.summary = None
        self.facts = None
        self.context = []

        self.lock = asyncio.Lock()

        self.summariser = Summariser(summary_model,  summary_max_tokens, keep_tokens_after_summary, min_recent_turns)

        self.cache_manager = CacheManager(gc_time_limit, gc_limit_size_MBs, gc_interval, cache_folder)

        self.queue = asyncio.Queue()

    async def init(self):
        await self.load()

    async def context_resolver(self, context, file_name_key=FILE_NAME_KEY, max_keeps=1, auto_save=True):
        await self.flush_queue()
        async with self.lock:

            context = await self.cache_manager.context_resolver(context, file_name_key ,max_keeps)

            if auto_save:
                self.context = context

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
        await self.flush_queue()
        await self.save()
        await self.cache_manager.shutdown()
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

    async def save(self):
        try:
            await self.flush_queue()
            async with self.lock:
                data = {"summary":self.summary, "facts": self.facts, "conversation": self.context}
                data_to_save = json.dumps(data, indent=2)

            async with aiofiles.open(self.context_path, "w") as file:
                await file.write(data_to_save)

        except IOError as e:
            await log(f"Error saving context: {e}", "error")