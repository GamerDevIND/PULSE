from .utils import Logger
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
import traceback
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

        self._summarization_tasks = {}

        os.makedirs(self.context_dir, exist_ok=True)
        self.event_bus = event_bus

    async def init(self):
        await self.load_all()

    async def context_resolver(self, cid, file_name_key=FILE_NAME_KEY, max_keeps=1, auto_save=True):
        if not cid in self.conversations:
            await Logger.log_async(f"'{cid}' isn't in the registry", 'warn')
            return []
        c = self.conversations[cid]
        await c.flush_queue()
        async with c.lock:
            
            context = await self.cache_manager.context_resolver(c.messages, file_name_key ,max_keeps)
            
            if auto_save:
                c.messages = context
            
        return context
    
    async def add_and_maintain(self, cid, data:dict | list[dict] | tuple[dict], update:bool = False):
        if not cid in self.conversations:
            await Logger.log_async(f"{cid} not in registry", 'error')
            return
        
        convo = self.conversations[cid]
        await convo.append(data, update)
        await convo.flush_queue()

        meta = await convo.get_states()

        async with convo.lock:
            snapshot_context = list(convo.messages)
            last_snapshot_msg_id = snapshot_context[-1]['msg_id'] if snapshot_context else None

        if self._summarization_tasks.get(cid) and not self._summarization_tasks[cid].done():
            return

        async def background_summarize():
            try:
                res = await self.summariser.maybe_summarise_context(snapshot_context, meta)
                if not res: return

                new_meta, pruned_context = res
                
                async with convo.lock:
                    pruned_ids = {m.get('msg_id') for m in pruned_context if m.get('msg_id')}
                    new_arrivals = [msg for msg in convo.messages if msg.get('msg_id') not in pruned_ids]

                    if last_snapshot_msg_id and len(new_arrivals) == len(convo.messages):
                        await Logger.log_async(
                            f"Snapshot message id {last_snapshot_msg_id} not found during merge for {cid}; using ids fallback.",
                            "warn"
                        )

                    final_context = pruned_context + new_arrivals
                    
                    convo.summary = new_meta.get('summary', convo.summary)
                    convo.facts = new_meta.get('facts', convo.facts)
                    convo.decisions = new_meta.get('key_decisions', convo.decisions)
                    convo.threads = new_meta.get('open_threads', convo.threads)
                    convo.messages = final_context
                    
                if not convo.temp:
                    await self.save(cid)
                    
                await Logger.log_async(f"Background summary for {cid} complete.", "success")
            
            except Exception as e:
                await Logger.log_async(f"Background summary failed: {e}", "error")

        async with self.lock:
            self._summarization_tasks[cid] = asyncio.create_task(background_summarize())

    async def get_context(self, cid):
        if not cid in self.conversations:
            await Logger.log_async(f"'{cid}' isn't in the registry", 'warn')
            return []
        c = self.conversations[cid]
        return await c.get_context()
        
    async def rename_convo(self, cid, name):
        if not cid in self.conversations:
            await Logger.log_async(f"'{cid}' isn't in the registry", 'warn')
            return 
        await self.conversations[cid].rename(name)

    async def get_summary(self, cid):
        if not cid in self.conversations:
            await Logger.log_async(f"'{cid}' isn't in the registry", 'warn')
            return ''
        c = self.conversations[cid]

        return await c.get_summary()

    async def get_facts(self, cid):
        if not cid in self.conversations:
            await Logger.log_async(f"'{cid}' isn't in the registry", 'warn')
            return []
        c = self.conversations[cid]
        return await c.get_facts()

    async def create_temp_chat(self):
        cid = str(uuid.uuid4())
        while cid in self.conversations.keys():
            cid = str(uuid.uuid4())

        convo = Conversation("", cid)
        convo.id = cid
        convo.name = "Temporary chat"
        convo.set_temp(True)
        async with self.lock:
            self.conversations[cid] = convo
        return convo
    
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
            await Logger.log_async(f"'{cid}' isn't in the registry", 'warn')
            return
        
        async with self.lock:
            c = self.conversations[cid]
            if not c.temp:
                p = c.path
                if not os.path.exists(p):
                    await Logger.log_async(f"'{cid}'doesn't exist.", 'warn')
                    return
                
                if not c.temp and os.path.exists(c.path):
                    await asyncio.to_thread(os.remove, c.path)

                async with self.lock:
                    t = self._summarization_tasks.get(cid)
                    if t:
                        t.cancel()
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass

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
            for c in convos if not c.temp
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
        for t in self._summarization_tasks.values():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        await Logger.log_async("Context saved and context manager shut down.", "info")

    async def load(self, cid):
        if not cid in self.conversations:
            await Logger.log_async(f"'{cid}' isn't in the registry", 'warn')
            return
        await self.conversations[cid].load()

    async def save(self, cid, path_to_save = None):
        if not cid in self.conversations:
            await Logger.log_async(f"'{cid}' isn't in the registry", 'warn')
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
        for i in list(self.conversations.values()):
            if not i.temp:
                await i.save()

class Conversation:
    def __init__(self, path:str, uuid_= None) -> None:
        self.path = path
        self.summary = ""
        self.threads = []
        self.decisions = []
        self.facts = []
        self.messages = []
        self.lock = asyncio.Lock()
        self.queue = asyncio.Queue()
        self.id = uuid_ or str(uuid.uuid4())
        self.name = "New chat"
        self.last_used = -1
        self.temp = False

    def set_temp(self, val:bool):
        self.temp = val

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
                self.last_used = content.get('last_used', -1)
                self.decisions = content.get("key_decisions", [])
                self.threads = content.get("open_threads", [])

        except (FileNotFoundError, json.JSONDecodeError):
            await Logger.log_async("Context missing or corrupted. Starting over.", "warn")
            self.messages = [] 
            self.summary = None
            self.facts = None
            self.id = self.id or str(uuid.uuid4())
            self.name = "New chat"
            self.last_used = -1
            self.decisions = []
            self.threads = []

        await Logger.log_async(f"Loading conversation: {self.name} ({self.id})", 'info')

    async def save(self, path:str | None = None):
        if self.temp:
            await Logger.log_async("Temporary conversations cannot be saved", 'warn')
            return
        await Logger.log_async(f"Saving conversation: {self.name} ({self.id})", 'info')
        try:
            await self.flush_queue()
            async with self.lock:
                data = {"summary":self.summary, "facts": self.facts,"key_decisions": self.decisions, "open_threads": self.threads, 
                        "conversation": self.messages, "id": self.id, 'name': self.name, 'last_used': self.last_used}
                data_to_save = json.dumps(data, indent=2)

            async with aiofiles.open(path or self.path, "w") as file:
                await file.write(data_to_save)

        except IOError as e:
            await Logger.log_async(f"Error saving context: {e}; {traceback.format_exc()}", "error")

    async def append(self, data:dict | list[dict] | tuple[dict], update:bool = False): 
        if not update:
            if isinstance(data, dict):
        
                await self.queue.put(data)
                async with self.lock: 
                    self.last_used=  datetime.datetime.now().timestamp()
            elif isinstance(data, (list, tuple)):
                for d in data: 
                    await self.queue.put(d)
        
                async with self.lock: 
                    self.last_used = datetime.datetime.now().timestamp()
            else: 
                await Logger.log_async("unknown data type, skipping item.", "warn")
        else:
            if not isinstance(data, (list, tuple)):
                raise TypeError
            
            async with self.lock:
                self.queue = asyncio.Queue()
                self.messages = list(data)
                self.last_used = datetime.datetime.now().timestamp()

    async def flush_queue(self):
        if self.queue.empty():
            return

        items = []
        
        async with self.lock:
            while True:
                try:
                    items.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            items = self.attach_meta(items)
            self.messages.extend(items)

    async def get_context(self):
        await self.flush_queue()
        async with self.lock:
            context = deepcopy(list(self.messages))
            return context
        
    async def get_states(self):
        meta = {
            'context': await self.get_context(),
            'summary': await self.get_summary(),
            'facts': await self.get_facts(),
            'key_decisions': await self.get_key_decisions(),
            'open_threads': await self.get_threads(),
        }
        async with self.lock:
            m = {
                'name': self.name,
                'id': self.id,
                'last_used': self.last_used
            }
            meta.update(m)
        
        return meta
        
    async def get_summary(self):
        async with self.lock:
            return str(self.summary) if self.summary is not None else ""

    async def get_facts(self):
        async with self.lock:
            return deepcopy(self.facts) if self.facts is not None else []
        
    async def get_threads(self):
        async with self.lock:
            return deepcopy(self.threads) if self.threads is not None else []
        
    async def get_key_decisions(self):
        async with self.lock:
            return deepcopy(self.decisions) if self.decisions is not None else []
        
    def attach_meta(self, new_messages:list[dict] | dict):
        new_messages = deepcopy(new_messages)
        if isinstance(new_messages, (list, tuple)):
            c = []
            for m in new_messages:
                m['edited'] =  m.get('edited') or False
                m['gen_idx'] = m.get("gen_idx") or 0
                m['msg_id'] = m.get('msg_id') or str(uuid.uuid4())
                m['msg_idx'] = m.get("msg_idx") or ((len(c) + len(self.messages)) + 1)
                c.append(m)

            return c
        
        elif isinstance(new_messages, dict):
            new_messages['edited'] =  new_messages.get('edited') or False
            new_messages['gen_idx'] = new_messages.get("gen_idx") or 0
            new_messages['msg_id'] = new_messages.get('msg_id') or str(uuid.uuid4())
            new_messages['msg_idx'] = new_messages.get("msg_idx") or (len(self.messages) + 1)
            return new_messages

    def attach_msg_extra_meta(self, new_messages:list[dict] | dict, new_meta: list[dict | None] | dict | None = None):
        m = self.attach_meta(new_messages)
        
        if isinstance(new_messages, (list, tuple)) and isinstance(new_meta, (list, tuple)):
            m = list(m)
            new_meta = list(new_meta)
            if len(new_messages) > len(new_meta):
                new_meta.extend(None for _ in range(len(new_messages) - len(new_meta)))
            if len(new_meta) > len(new_messages):
                raise ValueError("Metadata len too big")

            for msg, meta in zip(m, new_meta):
                if meta:
                    msg.update(meta)

            return m

        elif new_meta and (isinstance(new_messages, dict) and isinstance(new_meta, dict)):
            new_messages.update(new_meta)
            return new_messages

        else:
            raise TypeError("Metadata datatype doesn't match.")

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
