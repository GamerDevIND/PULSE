import asyncio
from typing import Literal
from .utils import log
from .router import Router
from .tools import ToolRegistry
from .multi_server import MultiServer
from .single_server import SingleServer
from .backend import Generation
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
)

class AI:
    def __init__(self, model_config_path="main/Models_config.json", context_path="main/saves/context.json", 
                 mode:Literal['single'] | Literal['multi'] = "multi" , max_turns = 5,absolute_max_turns = 50,):
        self.model_config_path = model_config_path
        self.context_path = context_path
        self.system_prompts = {
            "chat": CHAT_PROMPT,
            "router": ROUTER_PROMPT,
            "cot": CoT_PROMPT,
            'vision': VISION_PROMPT,
            'summarizer': SUMMARIZER_PROMPT,
        }
        self.default_role = 'chat'
        self.running_tasks = set()

        self.mode = mode.lower()
        self.backend = None

        self.max_turns = max_turns
        self.turns = 0
        self.total_turns = 0
        self.abs_max_turns = absolute_max_turns

        self.router = None

        self.regen_consent_callback = None
        self.platform = None
        self.context_manager = ContextManager(self.context_path, None)
        self.tools_regis = ToolRegistry()

    def load_models(self):
        if self.mode == 'multi':
            self.backend = MultiServer(self.model_config_path, self.system_prompts, DEFAULT_PROMPT)
            self.backend.load()
        elif self.mode == 'single':
            self.backend = SingleServer(self.model_config_path, self.system_prompts, DEFAULT_PROMPT)
            self.backend.load()
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Please ensure the mode is 'multi' or 'single'.")

    async def init(self, platform: str, summarising_model_role = "summariser", max_turns = 5, absolute_max_turns = 50,
                   mode: Literal['single'] | Literal['multi'] | None = None, regeneration_consent_callback = None, router_role = "router"): 
        self.max_turns = max_turns
        self.abs_max_turns = absolute_max_turns
        if mode: self.mode = mode.lower()
        self.load_models()
        self.platform = platform
        self.regen_consent_callback = regeneration_consent_callback

        await log("Warming up all models...", "info", append=False)

        if not self.backend:
            await log("No backend provided!", 'error')
            return

        await self.backend.init(self.tools_regis.list_tools())
        await self.context_manager.init()

        self.context_manager.summariser.model = self.backend.models.get(summarising_model_role)
        self.router = Router(self.backend.models.get(router_role), self.default_role, "chat", "cot", auto_warmup=True)

        for t in self.backend.running_tasks:
            self.running_tasks.add(t)

    async def cancel_generation(self, session_id:str):
        self.regen = False

        if not self.backend:
            await log("No backend provided!", 'error')
            return

        await self.backend.cancel_generation(session_id)

        await log('Generation cancelled by user', 'info')


    async def create_generation(self, query, stream: None | bool = None, manual_routing=False, think=None, file_path=None, video_frames_mod= 10, user_save_prefix = None, 
                       save_thinking = True, options = None, format_: dict | None = None):

        if not self.backend:
            await log("No backend provided!", 'error')
            raise

        context = await self.context_manager.get_context()
        query, role = await self.router.route_query(query, context, manual_routing) if self.router else (query, self.default_role)
        if not role: 
            role = self.default_role

        system = CHAOS_PROMPT if role == "chaos" else None

        role = "chat" if role == "chaos" else role
        model = self.backend.get_model(role)
        if not model:
            await log(f"No model found for role: {role} in {self.mode} mode!", 'error')
            return

        if stream is None:
            stream = self.platform not in STREAM_DISABLED

        context = await self.context_manager.context_resolver(context)

        if file_path:
            file_path = await self.context_manager.cache_manager.file_path_resolver(file_path)

        summary = await  self.context_manager.get_summary()
        facts = await self.context_manager.get_facts()
        f = f"""
                    The system has also generated a set of facts extracted from the previous turns. These are still lossy facts and maybe inaccurate. 
                    or reference only, use for user profile modeling.
                    <FACTS>
                    {"\n\n-".join(facts).strip()}
                    </FACTS>
                    """ if facts and "".join(facts).strip() else ''

        if summary: context.insert(0, {"role": "assistant", "content": f"""[PERSISTED MEMORY - NOT DIALOGUE]
                                           [INTERNAL MEMORY - DO NOT REPEAT - NOT PART OF CONVERSATION]

                                           The system generated summary of previous messages/turns. **Do NOT** treat this as the part of the conversation. For reference only. 
                                           <SUMMARY>
                                           {summary}
                                           </SUMMARY>

                                           {f}
                                           
                                           THESE ARE NOT A PART OF THE CONVERSATION. THESE ARE SYSTEM GENERATED PERSISTED MEMORY"""}) # role:system is fatal.

        sid, session = await self.backend.create_session(query, context, self.tools_regis, role, system, options, format_,
                                                         self.max_turns, self.abs_max_turns, self.regen_consent_callback)

        gen = Generation(session, self.backend.remove_session, self.context_manager.add_and_maintain, stream,
                         user_save_prefix, think, file_path, video_frames_mod, save_thinking)

        return gen

    async def shut_down(self):
        await log("Shutting Down all services...", "info")
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

        await log("Full System Offline.", "success")

        await asyncio.sleep(1.0)

        print("Done.")