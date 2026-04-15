from .utils import log
from .configs import FILE_NAME_KEY
import json 
import asyncio
import uuid
import aiofiles 
from copy import deepcopy
from .models.model_instance import LocalModel
from .models.models_profile import RemoteModel
from .summariser import Summariser
from .cache_manager import CacheManager
import os
import datetime
from .events import EventBus

class ContextManager:
    def __init__(self, context_dir, summary_model: LocalModel | RemoteModel | None, summary_max_tokens = 4000, keep_tokens_after_summary = 2000, 
                 min_recent_turns = 3, cache_folder = './cache', 
                 gc_time_limit = 259200, gc_limit_size_MBs = 50, gc_interval = 1800, event_bus : None | EventBus = None):
        
        self.context_dir = context_dir
        self.conversations:dict[str, Conversation] = {}

        self.lock = asyncio.Lock()

        self.summariser = Summariser(summary_model,  summary_max_tokens, keep_tokens_after_summary, min_recent_turns, event_bus)

        self.cache_manager = CacheManager(gc_time_limit, gc_limit_size_MBs, gc_interval, cache_folder, event_bus)

        os.makedirs(self.context_dir, exist_ok=True)
        self.event_bus = event_bus

    async def init(self):
        await self.load_all()

    async def context_resolver(self, cid, file_name_key=FILE_NAME_KEY, max_keeps=1, auto_save=True):
        if not cid in self.conversations:
            await log(f"'{cid}' isn't in the registry", 'warn')
            return []
        c = self.conversations[cid]
        await c.flush_queue()
        async with c.lock:
            
            context = await self.cache_manager.context_resolver(c.messages, file_name_key ,max_keeps)
            
            if auto_save:
                c.messages = context
            
        return context

    async def add_and_maintain(self, cid, data:dict | list[dict] | tuple[dict], update:bool = False):
        convo = self.conversations[cid]
        await convo.append(data, update)
        await convo.flush_queue()

        r = await self.summariser.maybe_summarise_context(convo.messages, convo.summary, convo.facts)
        if r:
            s, f, c = r
            async with convo.lock:
                convo.summary = s
                convo.facts = f
                convo.messages = c

        await self.save(cid)

    async def get_context(self, cid):
        if not cid in self.conversations:
            await log(f"'{cid}' isn't in the registry", 'warn')
            return []
        c = self.conversations[cid]
        return await c.get_context()
        
    async def rename_convo(self, cid, name):
        if not cid in self.conversations:
            await log(f"'{cid}' isn't in the registry", 'warn')
            return 
        await self.conversations[cid].rename(name)

    async def get_summary(self, cid):
        if not cid in self.conversations:
            await log(f"'{cid}' isn't in the registry", 'warn')
            return ''
        c = self.conversations[cid]

        return await c.get_summary()

    async def get_facts(self, cid):
        if not cid in self.conversations:
            await log(f"'{cid}' isn't in the registry", 'warn')
            return []
        c = self.conversations[cid]
        return await c.get_facts()
        
    async def new_conversation(self, name="New Chat"):
        cid = str(uuid.uuid4())
        while cid in self.conversations.keys():
            cid = str(uuid.uuid4())

        path = os.path.join(self.context_dir, f"{cid}.json")
        
        convo = Conversation(path, cid)
        convo.id = cid
        convo.name = name
        
        async with self.lock:
            self.conversations[cid] = convo
            
        await convo.save()
        return convo
    
    async def delete_conversation(self, cid):
        if not cid in self.conversations:
            await log(f"'{cid}' isn't in the registry", 'warn')
            return
        
        async with self.lock:
            c = self.conversations[cid]
            p = c.path
            if not os.path.exists(p):
                await log(f"'{cid}'doesn't exist.", 'warn')
                return 
            await asyncio.to_thread(os.remove, p)

            del self.conversations[cid]
    
    async def list_conversations(self):
        async with self.lock:
            convos = list(self.conversations.values())

        l = [
            {
                'id': c.id,
                'name': c.name,
                'last_used': c.last_used
             
            }
            for c in convos
        ]
        l = sorted(l, key=lambda x: x.get('last_used', 0), reverse=True)
        return l
    
    async def get_conversation(self, cid, create_new = False):
        if not cid in self.conversations:
            if not create_new:
                raise KeyError
            return await self.new_conversation()
        
        return self.conversations[cid]

    async def shut_down(self): 
        await self.save_all()
        await self.cache_manager.shutdown()
        await log("Context saved and context manager shut down.", "info")

    async def load(self, cid):
        if not cid in self.conversations:
            await log(f"'{cid}' isn't in the registry", 'warn')
            return
        await self.conversations[cid].load()

    async def save(self, cid, path_to_save = None):
        if not cid in self.conversations:
            await log(f"'{cid}' isn't in the registry", 'warn')
            return
        await self.conversations[cid].save(path_to_save)

    async def load_all(self):
        if not os.path.exists(self.context_dir):
            os.makedirs(self.context_dir, exist_ok=True)
            return 
        
        files = await asyncio.to_thread(os.listdir, self.context_dir)
        convos = []
        for i in files:
            if not i.endswith('.json'): continue
            path = os.path.join(self.context_dir, i)
            cid = i.removesuffix('.json')
            convo = Conversation(path, cid)
            await convo.load()
            convos.append((cid, convo))

        convos = sorted(convos, key=lambda x: x[1].last_used, reverse=True)
        async with self.lock:
            for i, c in convos:
                self.conversations[i] = c
    
    async def save_all(self):
        for i in self.conversations.values():
            await i.save()

class Conversation:
    def __init__(self, path:str, uuid_= None) -> None:
        self.path = path
        self.summary = None
        self.facts = None
        self.messages = []
        self.lock = asyncio.Lock()
        self.queue = asyncio.Queue()
        self.id = uuid_ or str(uuid.uuid4())
        self.name = "New chat"
        self.last_used = 0

    async def load(self):
        try:
            async with aiofiles.open(self.path) as file:
                content = await file.read()
                content = json.loads(content)
                self.summary = content.get("summary")
                self.facts = content.get("facts")
                self.messages = content.get("conversation",[])
                self.id = content.get("id", self.id or str(uuid.uuid4()))
                self.name = content.get("name", "New chat")
                self.last_used = content.get('last_used', 0)

        except (FileNotFoundError, json.JSONDecodeError):
            await log("Context missing or corrupted. Starting over.", "warn")
            self.messages = [] 
            self.summary = None
            self.facts = None
            self.id = self.id or str(uuid.uuid4())
            self.name = "New chat"
            self.last_used = 0

    async def save(self, path:str | None = None):
        try:
            await self.flush_queue()
            async with self.lock:
                data = {"summary":self.summary, "facts": self.facts, "conversation": self.messages, "id": self.id, 'name': self.name, 'last_used': self.last_used}
                data_to_save = json.dumps(data, indent=2)

            async with aiofiles.open(path or self.path, "w") as file:
                await file.write(data_to_save)

        except IOError as e:
            await log(f"Error saving context: {e}", "error")

    async def append(self, data:dict | list[dict] | tuple[dict], update:bool = False): 
        if not update:
            if type(data) == dict:
                await self.queue.put(data)
                async with self.lock: 
                    self.last_used=  datetime.datetime.now().timestamp()
            elif type(data) in [list, tuple]:
                for d in data: await self.queue.put(d)
                async with self.lock: 
                    self.last_used=  datetime.datetime.now().timestamp()
            else: await log("unknown data type, skipping item.", "warn")
        else:
            if not type(data) in [list, tuple]:
                raise TypeError
            async with self.lock:
                self.queue = asyncio.Queue()
                self.messages = list(data)

    async def flush_queue(self):
        if self.queue.empty():
            return

        items = []
        while not self.queue.empty():
            items.append(await self.queue.get())

        async with self.lock:
            self.messages.extend(items)

    async def get_context(self):
        async with self.lock:
            return deepcopy(list(self.messages))

    async def get_summary(self):
        async with self.lock:
            return str(self.summary) if self.summary is not None else ""

    async def get_facts(self):
        async with self.lock:
            return deepcopy(self.facts) if self.facts is not None else []

    async def rename(self, name):
        async with self.lock: 
            self.name = name
        await self.save()

    async def reset(self):
        async with self.lock:
            self.messages = []
            self.facts = None
            self.summary = None
        await self.save()