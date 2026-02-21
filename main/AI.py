import asyncio
from typing import Literal, Iterable
import inspect
from .utils import log
from .router import Router
from .tools import ToolRegistry
from .multi_server import MultiServer
from .single_server import SingleServer
from .context_manager import ContextManager
from .tools import Tool
from .configs import ( 
    CoT_PROMPT, 
    CHAT_PROMPT, 
    ROUTER_PROMPT, 
    DEFAULT_PROMPT, 
    STREAM_DISABLED,
    VISION_PROMPT,
    SUMMARIZER_PROMPT,
    CHAOS_PROMPT,
    FILE_NAME_KEY
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

        self.regen = False 

        self.max_turns = max_turns
        self.turns = 0
        self.total_turns = 0
        self.abs_max_turns = absolute_max_turns

        self.router = None

        self.regen_consent_callback = None
        self.platform = None
        self.gen_lock = asyncio.Lock()
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
        if mode is None: 
            mode = self.mode # type:ignore
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
    
    async def _ask_user_consent(self):
        if not self.regen_consent_callback: return False
        true_calls = [True, 1, "1" ,"true", "y", "yes", "consent", "confirm",]
        try:
            result = self.regen_consent_callback() # type:ignore
            if inspect.isawaitable(result): result = await result 
            if type(result) == str: result = result.lower().strip()
            return result in true_calls
        except Exception as e:
            await log(f"An error occurred while calling regeneration consent callback: {e}", "warn")
            return False

    async def _check_regen(self, tools:Iterable[Tool]):
        regen = any(t.needs_regeneration for t in tools)
        if regen:
            if self.total_turns >= self.abs_max_turns:
                await log("Absolute turns limit reached please send a message or reinitiate generation if your UI allows it.", "warn")
                self.turns = 0
                self.total_turns = 0
                self.regen = False
                return False 
            if self.turns < self.max_turns:
                self.turns += 1
                self.total_turns += 1
                await log("Regenerating...", "info")
                self.regen = True
                return True
            else:
                c = await self._ask_user_consent()
                if c:
                    self.total_turns += 1
                    self.turns = 0
                    self.regen =  True
                    return True
                else: 
                    self.regen = False
                    return False
                
    async def cancel_generation(self):
        self.regen = False

        if not self.backend:
            await log("No backend provided!", 'error')
            return
        
        await self.backend.cancel_generation()
            
        await log('Generation cancelled by user', 'info')

    async def run_generation_loop(self, query, stream: None | bool = None, manual_routing=False, think=None, file_path=None):
        while True:
            async for (thinking, response) in self.generate(query, stream, manual_routing, think, file_path): 
                yield thinking, response 
            if not self.regen: break

    async def generate(self, query, stream: None | bool = None, manual_routing=False, think=None, file_path=None, video_frames_mod= 10, ):

        if not self.backend:
            await log("No backend provided!", 'error')
            raise

        async with self.gen_lock:
            context = await self.context_manager.get_context()
            query, role = await self.router.route_query(query, context, manual_routing) if self.router else (query, self.default_role)
            if not role: 
                role = self.default_role
            system = CHAOS_PROMPT if role == "chaos" else None


            role = "chat" if role == "chaos" else role
            if stream is None:
                stream = self.platform not in STREAM_DISABLED

            content_final = ""
            thinking_final = ""
            tools_called = []

            context = await self.context_manager.context_resolver(context)

            if file_path:
                file_path = await self.context_manager.file_path_resolver(file_path)

            summary = self.context_manager.get_summary()
            facts = self.context_manager.get_facts()
            f = f"""
                    The system has also generated a set of facts extracted from the previous turns. These are still lossy facts and maybe inaccurate. 
                    or reference only, use for user profile modeling.
                    <FACTS>
                    {"\n".join(facts).strip()}
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

            async for (thinking_chunk, content_chunk, tools_chunk) in self.backend.generate(role, query, context, stream,
                                                think, file_path, system_prompt_override=system, mod_=video_frames_mod):
                    
                    thinking_final += thinking_chunk or ""
                    content_final += content_chunk or ""
                    
                    if tools_chunk:
                        tools_called.extend(tools_chunk)

                    yield (thinking_chunk or "", content_chunk or "")

            if query and query.strip(): 
                await self.context_manager.append({'role': 'user', 'content': query, FILE_NAME_KEY : file_path})
                
            await self.context_manager.add_and_maintain({'role': 'assistant', 'thinking': thinking_final, 'content': content_final, 'tool_calls': tools_called})

            await self.execute_tools(tools_called)
            await self.context_manager.save()

    async def execute_tools(self, tools):
        results = []  
        tools_objs = []  
        if not tools:  
            return results  
  
        for tool in tools:  
            if not isinstance(tool, dict) or 'function' not in tool:  
                await log(f"Invalid tool format: {tool}", "warn")  
                continue  
  
            await log(f"Executing tool: {tool['function'].get('name')}", "info")      
            tool_name = tool['function'].get('name')  
            tool_idx = tool['function'].get('index')  
            tool_args = tool['function'].get('arguments', {})  
  
            if not tool_name in self.tools_regis.tools.keys():  
                await log(f"{tool_name} is not in the registory. Skipping...", 'warn')  
                continue  
  
            result = await self.tools_regis.execute_tool(tool_name=tool_name, **tool_args)
            if result is None:
                await log(f"An error occured in the tools execution; Tool Name: {tool_name}", 'warn')
                continue

            result, tool_obj = result
            result["index"] = tool_idx  
            results.append(result)  
            tools_objs.append(tool_obj)  
          
        await self.context_manager.append(results) 
        await self._check_regen(tools_objs)   
        return results

    async def shut_down(self):
        await log("Shutting Down all services...", "info")
        for task in list(self.running_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if self.backend:
            await self.backend.shutdown()
        else:
            await log("No backend provided! Skipping backend shutdown.", 'error')

        await self.context_manager.shut_down()
        print("Done.")
