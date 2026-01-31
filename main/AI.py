import asyncio
import aiofiles
import json
import inspect
from .utils import log
from .models import Model
from .router import Router
from .tools import ToolRegistry, Tool
from .context_manager import ContextManager
from .configs import ( 
    CoT_PROMPT, 
    CHAT_PROMPT, 
    ROUTER_PROMPT, 
    DEFAULT_PROMPT, 
    CHAOS_PROMPT, 
    STREAM_DISABLED,
    VISION_PROMPT,
    SUMMARIZER_PROMPT,
    ERROR_TOKEN
)

class AI:
    def __init__(self, model_config_path="main/Models_config.json", context_path="main/saves/context.json", max_turns = 5,absolute_max_turns = 50,):
        self.model_config_path = model_config_path
        self.context_path = context_path
        self.models: dict[str, Model] = {}
        self.system_prompts = {
            "chat": CHAT_PROMPT,
            "router": ROUTER_PROMPT,
            "cot": CoT_PROMPT,
            'vision': VISION_PROMPT,
            'summarizer': SUMMARIZER_PROMPT,
        }
        self.default_role = 'chat'
        self.running_tasks = set()
        self.checking_event = asyncio.Event()

        self.regen = False 

        self.max_turns = max_turns
        self.turns = 0
        self.total_turns = 0
        self.abs_max_turns = absolute_max_turns

        self.router = None
        self.load_models()
        self.active_model = None
        self.generation_task = None
        self.regen_consent_callback = None
        self.platform = None
        self.gen_lock = asyncio.Lock()
        self.context_manager = ContextManager(self.context_path, None)
        self.tools_regis = ToolRegistry(self.context_manager)

    def load_models(self):
        try:
            with open(self.model_config_path, 'r', encoding="utf-8") as f:
                models_data = json.load(f)
                for model_data in models_data:
                    role = model_data.get('role')
                    if role:
                        system = model_data.get("system_prompt")
                        if not (system and system.strip()): model_data["system_prompt"] = self.system_prompts.get(role, DEFAULT_PROMPT)
                        self.models[role] = Model(**model_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            raise Exception("Models loading failed.", e)

    async def init(self, platform: str, summarising_model_role = "summariser", max_turns = 5, absolute_max_turns = 50, regeneration_consent_callback:callable|None = None):
        self.max_turns = max_turns
        self.abs_max_turns = absolute_max_turns
        self.platform = platform
        self.regen_consent_callback = regeneration_consent_callback
        self.router = Router(self.models.get("router"), self.default_role, "chat", "cot")

        await log("Warming up all models...", "info", append=False)

        for model in self.models.values():
            if model.has_tools:
                await model.add_tools(*self.tools_regis.list_tools())

            await model.warm_up()

        self.context_manager.summary_model = self.models.get(summarising_model_role)
        check_task = asyncio.create_task(self.check_models())
        self.running_tasks.add(check_task)

        check_task.add_done_callback(lambda t: self.running_tasks.discard(t))


    async def _ask_user_consent(self):
        if not self.regen_consent_callback: return False
        true_calls = [True, 1, "1" ,"true", "y", "yes", "consent", "confirm",]
        try:
            result = self.regen_consent_callback()
            if inspect.isawaitable(result): result = await result 
            if type(result) == str: result = result.lower().strip()
            return result in true_calls
        except Exception as e:
            await log(f"An error occurred while calling regeneration consent callback: {e}", "warn")
            return False

    async def _check_regen(self, tools):
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

    async def _ping_model_tag(self, url):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/api/tags") as res:
                    return res.status == 200
        except aiohttp.ClientError:
                return False

    async def check_models(self, interval=10):
        cooldowns = {name: 0 for name in self.models}

        while not self.checking_event.is_set(): # i forgot the syntax 
            for name, model in self.models.items():
                if model.state in ["down", "shutting down", "warming up"]:
                    continue

                if cooldowns[name] > 0:
                    cooldowns[name] -= 1
                    continue

                is_alive = await self._ping_model_tag(model.host)

                is_terminated = model.process.poll() is not None if model.process else True 

                if not is_alive:
                    await log(f"CRITICAL: {model.name} API is down. Restarting...", "error")
                    cooldowns[name] = 5
                    await model.warm_up()

                elif is_terminated:
                    model.state = "idle" 

            await asyncio.sleep(interval)

    async def cancel_generation(self):
        model = task = False
        self.regen = False

        if self.active_model:
            self.active_model.cancel()
            model = True
            self.active_model = None

        if self.generation_task:
            if not self.generation_task.done():
                self.generation_task.cancel()
                task = True
                try:
                    await self.generation_task
                except asyncio.CancelledError:
                    pass
                self.generation_task = None

        if model:
            await log("Model loop aborted", "info") 

        if task:
            await log("Streaming task aborted", "info")

        if task and model: await log('Generation cancelled by user', 'info')

    async def run_generation_loop(self, query, stream: None | bool = None, manual_routing=False, think=None, image_path=None):
        while True:
            async for (thinking, response) in self.generate(query, stream, manual_routing, think, image_path): yield thinking, response 
            if not self.regen: break

    async def generate(self, query, stream: None | bool = None, manual_routing=False, think=None, image_path=None):
        async with self.gen_lock:
            context = await self.context_manager.get_context()
            query, model_name = await self.router.route_query(query, context, manual_routing) if self.router else (query, self.default_role)
            if not model_name: 
                model_name = self.default_role

            model = self.models[model_name]
            self.active_model = model

            if stream is None:
                stream = self.platform not in STREAM_DISABLED

            await log(f"Using stream={stream} for {model_name}", "info")

            content_final = ""
            thinking_final = ""
            tools_called = None

            summary = self.context_manager.get_summary()

            if summary: context.insert(0, {"role": "assistant", "content": f"[PERSISTED MEMORY – NOT DIALOGUE]\n[INTERNAL MEMORY — DO NOT REPEAT — NOT PART OF CONVERSATION]\nThe system generated summary of previous messages/turns. **Do NOT** treat this as the part of the conversation. For reference only. \n<SUMMARY>\n{summary}\n</SUMMARY>"}) # role:system is fatal.

            if stream:
                queue = asyncio.Queue()
                async def producer():
                    try:
                        async for (thinking_chunk, content_chunk, tools_chunk) in model.generate(query, context, True,think=think, image_path=image_path):
                            if content_chunk == ERROR_TOKEN:
                                break
                            await queue.put((thinking_chunk, content_chunk, tools_chunk))
                    except asyncio.CancelledError:
                        raise
                    finally:
                        try:
                            queue.put_nowait(None)
                        except asyncio.QueueFull:
                            pass

                task = asyncio.create_task(producer())

                self.generation_task = task

                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    thinking_chunk, content_chunk, tools_chunk = item
                    thinking_final += thinking_chunk or ""
                    content_final += content_chunk or ""
                    if tools_called is None:
                        tools_called = []

                    if tools_chunk:
                        tools_called.extend(tools_chunk)

                    yield (thinking_chunk or "", content_chunk or "")

                await task            
            else:
                await log(f"Non-streaming mode active", "info")
                async for (thinking, content, tools) in model.generate(query, context, False, think=think, image_path=image_path):
                    await log(f"Got non-streaming response chunk", "info")
                    if content == ERROR_TOKEN:
                        break

                    thinking_final = thinking or ""
                    content_final = content or ""
                    tools_called = tools
                    yield (thinking_final, content_final)

            if query and query.strip(): 
                await self.context_manager.append({'role': 'user', 'content': query})

            await self.context_manager.add_and_maintain({'role': 'assistant', 'thinking': thinking_final, 'content': content_final, 'tool_calls': tools_called})

            self.active_model = None
            self.generation_task = None


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

            result, tool_obj = await self.tools_regis.execute_tool(**tool_args, tool_name=tool_name)  
            result["index"] = tool_idx  
            results.append(result)  
            tools_objs.append(tool_obj)  

        await self.context_manager.append(results) 
        await self._check_regen(tools_objs)   
        return results

    async def shut_down(self):
        await log("Shutting Down all services...", "info")
        self.checking_event.set()
        for task in list(self.running_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for model in self.models.values():
            await model.shutdown()

        await self.context_manager.shut_down()
        print("Done.")