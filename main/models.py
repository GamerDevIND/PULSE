
import asyncio
import aiofiles
import aiohttp
import json
import inspect
from .utils import log, load_tools, Tool
from .models import Model
from .Events import Event_Manager
from .router import Router
from .configs import ( 
    CoT_PROMPT, 
    CHAT_PROMPT, 
    ROUTER_PROMPT, 
    DEFAULT_PROMPT, 
    CHAOS_PROMPT, 
    STREAM_DISABLED,
    ERROR_TOKEN,
    VISION_PROMPT,
    SUMMARIZER_PROMPT
)

def on_event(event_name):
    def decorator(func):
        func._event_trigger = event_name
        return func
    return decorator
    
class AI:
    def __init__(self, model_config_path="main/Models_config.json", context_path="main/saves/context.json"):
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
        self.tools_regis = {}
        self.context_lock = asyncio.Lock()
        self.summary: str | None = None
        self.context = []
        self.router = None
        self.load_models()
        self.active_model = None
        self.generation_task = None

    def load_models(self):
        try:
            with open(self.model_config_path, 'r', encoding="utf-8") as f:
                models_data = json.load(f)
                for model_data in models_data:
                    role = model_data.get('role')
                    if role:
                        model_data["system_prompt"] = self.system_prompts.get(role, DEFAULT_PROMPT)
                        self.models[role] = Model(**model_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            raise Exception("Models loading failed.", e)

    async def init(self, platform: str, tools:list=[]):
        self.context = await self.load_context()
        self.platform = platform
        self.event_manager = Event_Manager()
        await self.event_manager.start_event()

        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, "_event_trigger"):
                self.event_manager.on(method._event_trigger)(method)

        await log("Warming up all models...", "info", append=False)

        tool_defs = await load_tools(tools)

        for tool in tool_defs:
            if isinstance(tool, Tool):
                self.tools_regis[tool.name] = tool
            elif callable(tool) and not inspect.isclass(tool):
                tool = Tool(tool, 'silent')
                self.tools_regis[tool.name] = tool
            else:
                await log(f"{tool} can't be parsed, skipping...", 'warn')

        for model in self.models.values():
            if model.has_tools:
                await model.add_tools(*tool_defs)
                
            await model.warm_up()
        check_task = asyncio.create_task(self.check_models())
        self.running_tasks.add(check_task)

        check_task.add_done_callback(lambda t: self.running_tasks.discard(t))
        self.router = Router(self.models.get("router"), self.default_role, "chat", "cot")

    @on_event("save_context_requested")
    async def _handle_save_event(self, **kwargs):
        await self.save_context()

    @on_event("execute_tools_requested")
    async def _handle_tool_event(self, tools, **kwargs):
        if tools:
            await self.execute_tools(tools)

    async def load_context(self):
        try:
            async with aiofiles.open(self.context_path) as file:
                content = await file.read()
                content = json.loads(content)
                self.summary = content.get("summary")
                return content.get("conversation",[])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    async def save_context(self):
        await self.event_manager.emit_async("save_context_started")
        try:
            async with self.context_lock:
                data = {"summary":self.summary, "conversation": self.context}
                data_to_save = json.dumps(data, indent=2) 

            async with aiofiles.open(self.context_path, "w") as file:
                await file.write(data_to_save)
                await self.event_manager.emit_async("save_context_completed")
        except IOError as e:
            await log(f"Error saving context: {e}", "error")
            await self.event_manager.emit_async("save_context_failed", error=str(e))

    async def _test_model(self, url, ollama_name ):
        try:
            data = {
            "model": ollama_name,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            }
            headers = {"Content-Type": "application/json"}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.post(f"{url}/api/chat", headers=headers, data=json.dumps(data)) as res:
                    res.raise_for_status()
                    buffer = ""
                    async for chunk in res.content.iter_any():
                        buffer += chunk.decode("utf-8")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line.strip():
                                try:
                                    json.loads(line.strip())
                                    return True
                                except json.JSONDecodeError:
                                    continue
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
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

        while True:
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
          

    async def summarise_context(self, context = None, max_nums = 20, model_role = 'summarizer', keep_nums = 10, save_context = False):  
        if context is None:  
            async with self.context_lock:  
                context = list(self.context)  
  
        model = self.models.get(model_role)  
        if model:  
            
            if len(context) >= max_nums:  
                given_context = list(context[:max_nums-keep_nums])  
                text = '\n'  
                for t in given_context:  
                    role = t['role']  
                    content = t['content']  
  
                    if role != 'tool':  
                        text += f"{role} : {content}\n"  
  
                text = f"(user/assistant messages below are the ONLY new information)\nOutput ONLY the updated summary text. No commentary.\n<NEW_CONVERSATION> \n{text}\n </NEW_CONVERSATION>"  
                system = self.system_prompts.get('summarizer', f'''This is a conversation log. Produce a factual, neutral summary.\n'  
                                                       'Do your only job properly: SUMMARIZE THE GIVEN CONVERSATION.'   
                                                       "Do ****NOT**** try to be helpful and answer the given query, just summarise the given conversation. Output ONLY the updated summary text. No commentary.''')  
  
                if self.summary: text = f"Below is the previous rolling summary of the conversation.\nIt represents persisted memory.\nUpdate it ONLY if the new conversation content adds facts or contradicts it.\nIf nothing changes, reproduce it verbatim.\n\n<PREVIOUS_SUMMARY>\n{self.summary}\n</PREVIOUS_SUMMARY>" + text  
  
                entry = None  
                try:  
                    async for (_, out, _ )in model.generate(text, [], stream=False, system_prompt_override=system):  
                        if out != ERROR_TOKEN and (out and isinstance(out, str) and out.strip()):  
  
                            entry = out
  
                except Exception as e:  
                    await log(f"Error during summarization: {e}", "error")  
  
                if entry: 
                    self.summary = entry  
                    context = context[-keep_nums:]  
                if save_context:  
                    async with self.context_lock:  
                        self.context = context  
                        await self.event_manager.emit_async("save_context_requested")  
  
                # TODO: Trim the context to remove older entries if needed explicitly  
 
                return context

    async def cancel_generation(self):
        model = task = False
        
        if self.active_model:
            self.active_model.cancel()
            model = True

        if self.generation_task:
            if not self.generation_task.done():
                self.generation_task.cancel()
                task = True
                try:
                    await self.generation_task
                except asyncio.CancelledError:
                    pass

        if model:
            await log("Model loop aborted", "info")

        if task:
            await log("Streaming task aborted", "info")
            
        if task and model: await log('Generation camcelled by user', 'info')

    async def generate(self, query, stream: None | bool = None, manual_routing=False, think=None, image_path=None):
        async with self.context_lock:
            context = list(self.context)
        query, model_name = await self.router.route_query(query, context, manual_routing) if self.router else (query, self.default_role)
        if not model_name: 
            model_name = self.default_role

        model = self.models[model_name]

        if stream is None:
            stream = self.platform not in STREAM_DISABLED

        await log(f"Using stream={stream} for {model_name}", "info")

        content_final = ""
        thinking_final = ""
        tools_called = None
       
        if self.summary: context.insert(0, {"role": "assistant", "content": f"[PERSISTED MEMORY – NOT DIALOGUE]\n[INTERNAL MEMORY — DO NOT REPEAT — NOT PART OF CONVERSATION]\nThe system generated summary of previous messages/turns. **Do NOT** treat this as the part of the conversation. For reference only. \n<SUMMARY>\n{self.summary}\n</SUMMARY>"}) # role:system is fatal.

        await self.event_manager.emit_async("generation_started", model_name=model_name, query=query)


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
                    queue.put_nowait(None)

            task = asyncio.create_task(producer())

            self.generation_task = task
            self.active_model = model
            
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

                await self.event_manager.emit_async("generation_chunk", model_name=model_name, thinking_chunk=thinking_chunk, content_chunk=content_chunk)

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
                await self.event_manager.emit_async("generation_chunk", model_name=model_name, thinking_chunk=thinking, content_chunk=content)
                yield (thinking_final, content_final)

        await self.event_manager.emit_async("generation_completed", model_name=model_name, query=query, final_thinking=thinking_final, final_content=content_final, tools_called=tools_called)
        context = await self.summarise_context(save_context = False)       
        async with self.context_lock:
            if context: self.context = context
            if query and query.strip(): 
                self.context.append({'role': 'user', 'content': query})
                
            self.context.append({'role': 'assistant', 'thinking': thinking_final, 'content': content_final, 'tool_calls': tools_called})
        
        self.active_model = None
        self.generation_task = None

        await self.event_manager.emit_async("execute_tools_requested", tools=tools_called) # too lazy to update self.context here.
       
        await self.event_manager.emit_async_block("save_context_requested") # just extra safe. 

    async def execute_tools(self, tools):
        results = {}
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

            if not tool_name in self.tools_regis.keys():
                await log(f"{tool_name} is not in the registory. Skipping...", 'warn')
                continue

            tool_instance = self.tools_regis.get(tool_name)
            if not isinstance(tool_instance, Tool):
                await log(f"Found no matching `Tool` instance for {tool_name}. Skipping...", 'warn')
                continue

            result = await tool_instance.execute(**tool_args)
            results[tool_idx] = result

            async with self.context_lock:
                    self.context.append(
                    {'role':'tool', 'tool_name':tool_name, 'content':str(results[tool_idx])}
                    )
            if isinstance(tool_instance, Tool): await self.event_manager.emit_async("tool_executed", tool = tool_instance)
            await self.event_manager.emit_async_block("save_context_requested")

        return results

    async def shut_down(self):
        await log("Shutting Down all services...", "info")
        await self.event_manager.stop_event()
        
        for task in list(self.running_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for model in self.models.values():
            await model.shutdown()

        await self.save_context()
        print("Done.")
