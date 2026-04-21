import asyncio
from typing import Literal
from .utils import log
from .router import Router
from .tools import ToolRegistry
from .models.model_instance import LocalModel, LocalEmbedder
from .models.models_profile import RemoteModel, RemoteEmbedder
from .models.openrouter_model import OpenRouterModel, OpenRouterEmbedder
from .backends.multi_server import MultiServer
from .backends.openrouter_backend import OpenrouterBackend
from .backends.single_server import SingleServer
from .backends.backend import Generation
import inspect
from .events import EventBus
from .context_manager import ContextManager
from .configs import ( 
    CoT_PROMPT, 
    CHAT_PROMPT, 
    ROUTER_PROMPT, 
    DEFAULT_PROMPT, 
    STREAM_DISABLED,
    VISION_PROMPT,
    SUMMARIZER_PROMPT,
    CHAOS_PROMPT,
    EMBEDDING_MODEL_ROLE,
    RAG_MIN_SCORE,
    USERNAME,
)
    
class AI:
    def __init__(self, model_config_path="main/Models_config.json", context_dir="main/saves/", 
                 mode:Literal['single'] | Literal['multi'] | Literal['openrouter'] = "multi" , max_turns = 5,  absolute_max_turns = 50, use_RAG = True, memory_db_path = "./RAG_DB",
                   table_name = 'memories', summary_max_tokens = 4000, keep_tokens_after_summary = 2000, 
                 min_recent_turns = 3, cache_folder = './cache', 
                 gc_time_limit = 259200, gc_limit_size_MBs = 50, gc_interval = 1800, embedder_auto_warm_up = True, max_memory_rag_chars = 1000):
        self.model_config_path = model_config_path
        self.context_dir = context_dir
        self.system_prompts = {
            "chat": CHAT_PROMPT,
            "router": ROUTER_PROMPT,
            "cot": CoT_PROMPT,
            'vision': VISION_PROMPT,
            'summarizer': SUMMARIZER_PROMPT,
        }
        self.use_RAG = use_RAG
        self.system_prompts = {k: v + f"\n\nThe **user's username** is: {USERNAME}" for k, v in self.system_prompts.items()}
        self.default_role = 'chat'
        self.running_tasks = set()
        self.max_memory_rag_chars = max_memory_rag_chars

        self.mode = mode.lower()
        self.backend = None
        self.last_cid:None | str = None

        self.max_turns = max_turns
        self.turns = 0
        self.total_turns = 0
        self.abs_max_turns = absolute_max_turns

        self.router = None
        self.status : dict = {"status":"Not initialised", "message": ""}
        self.lock = asyncio.Lock()

        self.regen_consent_callback = None
        self.mem_save_confirm_callback = None
        self.platform = None
        self.event_bus = EventBus()
        self.context_manager = ContextManager(self.context_dir, None,summary_max_tokens, keep_tokens_after_summary, 
                 min_recent_turns, cache_folder, 
                 gc_time_limit, gc_limit_size_MBs, gc_interval, self.event_bus)
        self.tools_regis = ToolRegistry()
        if self.use_RAG:
            from .RAG.manager import RAG_manager
            self.RAG_Manager = RAG_manager(None, embedder_auto_warm_up, memory_db_path, table_name)
        else:
            self.RAG_Manager = None

    def load_models(self):
        d = DEFAULT_PROMPT + f"\n\nThe user's username is: {USERNAME}"
        if self.mode == 'multi':
            self.backend = MultiServer(self.model_config_path, self.system_prompts, d, self.event_bus)
            self.backend.load()
        elif self.mode == 'single':
            self.backend = SingleServer(self.model_config_path, self.system_prompts, d, event_bus=self.event_bus)
            self.backend.load()
        elif self.mode == 'openrouter':
            self.backend = OpenrouterBackend(self.model_config_path, self.system_prompts, d, self.event_bus)
            self.backend.load()
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Please ensure the mode is 'multi' or 'single'.")

    def event(self, event_name:str):
        def wrapper(func):
            self.event_bus.add_listener(event_name, func)
            func.__event_name__ = event_name
            return func
        
        return wrapper

    async def init(self, platform: str, summarising_model_role = "summariser", max_turns = 5, absolute_max_turns = 50,
                   mode: Literal['single'] | Literal['multi'] | Literal['openrouter'] | None = None, regeneration_consent_callback = None, 
                   mem_save_confirm_callback = None, router_role = "router"): 
        
        await self.event_bus.parrallel_emit(self.event_bus.INITIALISING)
        meta = {
            "needs_regeneration": False,
            "retry": "none",
            "max_retries": 1,
            "retry_on": (Exception,)
        }
        self.tools_regis.register(self.propose_memory, meta)


        self.max_turns = max_turns
        self.abs_max_turns = absolute_max_turns

        if mode: self.mode = mode.lower()
        async with self.lock:
            self.status = {"status":"Loading models", "message": "Loading models from configs..."}
        self.load_models()
        self.platform = platform
        self.regen_consent_callback = regeneration_consent_callback
        self.mem_save_confirm_callback = mem_save_confirm_callback

        await log("Warming up all models...", "info", append=False)

        if not self.backend:
            await log("No backend provided!", 'error')
            return
        
        async with self.lock:
            self.status = {"status":"Warming up models", "message": "Warming up models for inference..."}
        
        meta = {
            "needs_regeneration": False,
            "retry": "none",
            "max_retries": 1,
            "retry_on": (Exception,)
        }

        self.tools_regis.register(self.propose_memory, meta)

        await self.backend.init(self.tools_regis.list_tools())

        m = self.backend.models.get(summarising_model_role)

        if m and not isinstance(m, (RemoteModel, LocalModel, OpenRouterModel)):
            raise TypeError

        self.context_manager.summariser.model = m

        m = self.backend.models.get(router_role)

        if m and  not isinstance(m, (RemoteModel, LocalModel, OpenRouterModel)):
            raise TypeError

        self.router = Router(m, self.default_role, "chat", "cot", auto_warmup=True)

        m = self.backend.models.get(EMBEDDING_MODEL_ROLE)

        if m and not isinstance(m, (RemoteEmbedder, LocalEmbedder, OpenRouterEmbedder)):
            raise TypeError
        if self.RAG_Manager:
            self.RAG_Manager.embedder = m

        async with self.lock:
            self.status = {"status":"Loading conversations and memories", "message": ""}
        await self.context_manager.init()

        if self.RAG_Manager:
            await self.RAG_Manager.load()

        for t in self.backend.running_tasks:
            self.running_tasks.add(t)

        async with self.lock:
            self.status = {"status":self.event_bus.INITIALISED, "message": ""}

        await self.event_bus.parrallel_emit(self.event_bus.INITIALISED)
                
    async def cancel_generation(self, session_id:str):
        await self.event_bus.sequence_emit(self.event_bus.CANCELLING_SESSION, session_id = session_id)
        if not self.backend:
            await log("No backend provided!", 'error')
            raise
        
        await self.backend.cancel_generation(session_id)
            
        await log('Generation cancelled by user', 'info')

    async def propose_memory(self, memory: str):
        '''
        This tool SUBMITS A PROPOSAL for the user to review. It does NOT save directly. Use ONLY for facts the user explicitly asked to persist.
        The memory is reviewed by the user before saving. The user may or may not choose to save it. 
        Invoke this tool ONLY when explicitly said by the user or you're 100% sure it'll be to save the fact for better response generation in the future, it'll still be reviewed by the user. You can only propose to save to memory.
        `memory` field takes a string (the actual memory to be saved) and saves it to the long term memory if the user confirms so.
        '''

        if not memory or not memory.strip() or len(memory) < 10:
            return "Empty or extrememly short memory. Skipping."
        
        await self.event_bus.sequence_emit(self.event_bus.PROPOSED_MEMORY, memory = memory)
        
        blacklist = ["i am a language model", "i'm a language model", "i am a god", "i'm a god", "as a helpful assistant", "i think the user", 
                                "i believe we should remember", "i believe i should remember", "i believe its worth remembering", "i think we should remember", 
                                "i think i should remember", "i think its worth remembering"]
        
        if any(phrase in memory.lower() for phrase in blacklist):
            return "Rejected: sounds like model self-talk."

        if not self.mem_save_confirm_callback:
            await log("Memory save confirmation callback missing!", "error")
            return "Confirmation system not ready. Skipping."
        
        true_calls = [True, 1, "1" ,"true", "y", "yes", "consent", "confirm",]

        confirmed = self.mem_save_confirm_callback(memory)
        if inspect.isawaitable(confirmed):
            confirmed = await confirmed

        if type(confirmed) == str: confirmed = confirmed.lower().strip()

        if not confirmed in true_calls:
            return "User refused to save to memory. Do not insist."

        if self.RAG_Manager:
            await self.RAG_Manager.index_text(
            memory, 
            {'source': "User-confirmed model memory"}
            )
        else:
            return "RAG manager not set for this instance, do not retry without restart!"
        
        return "Memory saved successfully."
    
    async def update_memory(self, mem_id:str, new:str):
        if not new or not new.strip() or len(new) < 10:
            return False
        
        if self.RAG_Manager:
            return await self.RAG_Manager.update(mem_id, new)
        else:
            return False
    
    async def list_memories(self, limit:int = 10, offset:int = 0):
        if self.RAG_Manager:
            return await self.RAG_Manager.list_memories(limit, offset)
        else:
            return []
    
    async def delete_memory(self, mem_id:str):
        if self.RAG_Manager:
            return await self.RAG_Manager.delete(mem_id)
        else:
            return False
    
    async def get_prompted_query(self, context:list, summary, facts, use_memory = True, query=""):

        '''
        USE THIS METHOD UNDER A LOCK FOR CONCURRENCY SAFETY!
        '''

        f = f"""
            The system has also generated a set of facts extracted from the previous turns. These are still lossy facts and maybe inaccurate. 
            For reference only, use for user profile modeling.
            <FACTS>
                {"\n\n-".join(facts).strip()}
            </FACTS>
            """ if facts and "".join(facts).strip() else ''
       
        if summary:
            context.insert(0, {"role": "assistant", "content": f"""[PERSISTED MEMORY - NOT DIALOGUE]
                [INTERNAL MEMORY - DO NOT REPEAT - NOT PART OF CONVERSATION]

                The system generated summary of previous messages/turns. **Do NOT** treat this as the part of the conversation. For reference only. 
                <SUMMARY>
                    {summary}
                </SUMMARY>

                {f}
                                           
                THESE ARE NOT A PART OF THE CONVERSATION. THESE ARE SYSTEM GENERATED PERSISTED MEMORY"""}) # role:system is fatal.
            
        if use_memory:
            if not self.RAG_Manager:
                await log("RAG manager not set for the instance!", 'error')
                return context, query
            
            self.status = {"status": "Retrieving memory...", "message": ""}
            await self.event_bus.parrallel_emit(self.event_bus.RETRIEVING_MEMORY, query = query)
            rag_results = await self.RAG_Manager.retrieve(query, min_score=RAG_MIN_SCORE)

            if rag_results:
                scores = [r["score"] for r in rag_results]
                await log(f"RAG: {len(rag_results)} chunks, scores {min(scores):.3f}–{max(scores):.3f}, avg {sum(scores)/len(scores):.3f}", 'info')
                current_chars = 0
                rag_text = ""
                for r in rag_results:
                    score = r["score"]
                    text = r["text"]
                    meta = r["metadata"]
                    if not text.strip():
                        continue
                    if current_chars + len(text) > self.max_memory_rag_chars:
                        break
                    source = meta.get('source', 'unknown') if meta else 'unknown'
                    rag_text += f"\n\nSource: {source} | Score: {score:.3f} | {text}"
                    current_chars += len(text)

                async with self.lock:
                    self.status['message'] = f'Retrieved {len(rag_results)} results'

                query = f'''
                    The system has retrieved a few *maybe* relevent user memory, saved by the system.
                    <important>
                    THESE ARE **NOT** PART OF THE CONVERSATION. THESE ARE SYSTEM MAINTAINED MEMORY.
                    </important>
                    <RETRIEVED CHUNKS>
                    {rag_text}
                    </RETRIEVED CHUNKS>
                    <UserQuery>
                    {query}
                    </UserQuery>
                    '''   
                
        return context, query

    async def create_generation(self, query, cid= None, stream: None | bool = None, manual_routing=False, think=None, file_path=None, video_frames_mod= 10, 
                                user_save_prefix = None, save_thinking = True, options = None, format_: dict | None = None, use_memory = True):
        if not self.backend:
            await log("No backend provided!", 'error')
            raise

        await self.event_bus.sequence_emit(self.event_bus.CREATING_SESSION)

        cid = cid or self.last_cid
        if not cid:
            await log(f"conversation for '{cid}' not found, creating a new conversation...", 'warn')
            c = await self.context_manager.new_conversation()
            cid = c.id

        c = self.context_manager.conversations.get(cid)

        if not c:
            await log(f"conversation for '{cid}' not found, creating a new conversation...", 'warn')
            c = await self.context_manager.new_conversation()
            self.last_cid = c.id

        context = await c.get_context()
        self.last_cid = c.id
        async with self.lock:
            self.status = {"status":"Routing", "message": ""}
        
        await self.event_bus.sequence_emit(self.event_bus.ROUTING)

        query, role = await self.router.route_query(query, context, manual_routing) if self.router else (query, self.default_role)
        if not role:
            role = self.default_role

        await self.event_bus.sequence_emit(self.event_bus.ROUTING_ROLE, role = role)

        async with self.lock:
            self.status['message'] = f"Routing to: {role}"

        system = (CHAOS_PROMPT + f"\n\nThe user's username is: {USERNAME}") if role == "chaos" else None

        role = "chat" if role == "chaos" else role
        model = self.backend.get_model(role)
        if not model:
            await log(f"No model found for role: {role} in {self.mode} mode!", 'error')
            return
            
        if stream is None:
            stream = self.platform not in STREAM_DISABLED

        context = await self.context_manager.context_resolver(cid)

        if file_path:
            file_path = await self.context_manager.cache_manager.file_path_resolver(file_path)

        summary = await  c.get_summary()
        facts = await c.get_facts()

        async with self.lock:
            self.status = {"status": "Getting prompts", "message": "Prompting the query for better UX"}
            context, query = await self.get_prompted_query(context, summary, facts, use_memory, query)
            print(query)
            self.status = {"status": "Generating session", 'message': ""}
                

        await log('Generating session...', 'info')
            
        sid, session = await self.backend.create_session(query, context, self.tools_regis, role, system, options, format_,
                                                         self.max_turns, self.abs_max_turns, self.regen_consent_callback)
       
        gen = Generation(session, self.backend.remove_session, lambda x: self.context_manager.add_and_maintain(cid, x, True), 
                         stream,user_save_prefix, think, file_path, video_frames_mod, save_thinking, self.event_bus)
    
        async with self.lock:
            self.status = {"status": "Session generation succcessful", "message": ""}
        
        await self.event_bus.sequence_emit(self.event_bus.SESSION_CREATED)
        
        return gen

    async def shut_down(self):
        await log("Shutting Down all services...", "info")
        await self.event_bus.parrallel_emit(self.event_bus.SHUTTING_DOWN)
        async with self.lock:
            self.status = {"status": "Shutting down", "message": ""}

        for task in list(self.running_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await self.context_manager.shut_down()
        
        if self.backend:
            await self.backend.shutdown()
        else:
            await log("No backend provided! Skipping backend shutdown.", 'error')
        
        await self.event_bus.parrallel_emit(self.event_bus.SHUTDOWN)

        await log("Full System Offline.", "success")
        
        await asyncio.sleep(1.0)

        async with self.lock:
            self.status = {"status": "Down", 'message': ""}
        
        print("Done.")