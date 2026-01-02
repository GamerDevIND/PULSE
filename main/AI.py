import asyncio
import aiofiles
import aiohttp
import json
from .utils import log, load_tools, Tool
from .models import Model
from .Events import Event_Manager
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
        self.default_model = 'chat'
        self.load_models()
        self.running_tasks = set()
        self.tools_regis = {}
        self.context_lock = asyncio.Lock()
        self.summary: str | None = None
        self.context = []

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
            raise Exception("死ね")

    async def init(self, platform: str, tools:list=[], max_summarize_nums=20, 
                summarize_model_role='chat'):
        self.context = await self.load_context()
        self.platform = platform
        self.event_manager = Event_Manager()
        await self.event_manager.start_event()

        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, "_event_trigger"):
                self.event_manager.on(method._event_trigger)(method)

        await log("Warming up all models...", "info", append=False)
        available_tools = [*tools]
        
        self.available_tools = {str(t.__name__): t for t in available_tools}

        tool_defs = await load_tools(*(self.available_tools.values()))

        for tool in tool_defs: 
           if isinstance(tool, Tool):self.tools_regis[tool.name] = tool

        for model in self.models.values():
            if model.has_tools:
                await model.add_tools(*tool_defs)
            if model.role == "router": model.add_to_avaliable_roles("chat", "cot")
                
            await model.warm_up()
        check_task = asyncio.create_task(self.check_models())
        self.running_tasks.add(check_task)

        check_task.add_done_callback(lambda t: self.running_tasks.discard(t))

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

    async def route_query(self, query: str, manual: bool):
        if not manual:
            router_model = self.models.get("router")
            if not router_model:
                await log("Router model not configured. Using default.", "error")
                return query, self.default_model

            router_resp_parts = []
            async with self.context_lock:
                context = list(self.context)

            async for _, part, _ in router_model.generate(query, context, stream=False):
                if part == ERROR_TOKEN:
                    await log("Router API call failed. Using default.", "error")
                    return query, self.default_model
                if part:
                    router_resp_parts.append(part)

            router_resp_raw = "".join(router_resp_parts).strip()
            selected_role = None

            await log(f"Router raw response: {router_resp_raw}", "info")

            try:
                router_resp = json.loads(router_resp_raw)
                if isinstance(router_resp, dict):
                    selected_role = router_resp.get("role", "").strip().lower()
            except json.JSONDecodeError:
                selected_role = router_resp_raw.lower().strip()

            if selected_role in self.models:
                await log(f"Router selected model '{selected_role}'.", "info")
                await self.event_manager.emit_async("router_selected", query=query, selected_role=selected_role)
                return query, selected_role
            else:
                await log(f"Router selected unknown role or failed to parse ('{selected_role}'). Using default '{self.default_model}'.", "warn")
                await self.event_manager.emit_async("router_selected", query=query, selected_role=self.default_model)
                return query, self.default_model
        else:
            if query.startswith("!chat"):
                query = query.removeprefix("!chat").strip()
                self.models['chat'].system = CHAT_PROMPT
                await self.event_manager.emit_async("router_selected", query=query, selected_role="chat")
                return query, "chat"
            elif query.startswith("!cot"):
                query = query.removeprefix("!cot").strip()
                await self.event_manager.emit_async("router_selected", query=query, selected_role="cot")
                return query, "cot"
            elif query.startswith("!chaos"):
                query = query.removeprefix("!chaos").strip()
                self.models['chat'].system = CHAOS_PROMPT
                await self.event_manager.emit_async("router_selected", query=query, selected_role="chat")
                return query, "chat"
            elif query.startswith("!vision"):
                query = query.removeprefix("!vision").strip()
                await self.event_manager.emit_async("router_selected", query=query, selected_role="vision")
                return query, "vision"
            else:
                await log(f"No or incorrect prefix, using default '{self.default_model}'", "info")
                await self.event_manager.emit_async("router_selected", query=query, selected_role=self.default_model)
                return query, self.default_model

    async def generate(self, query, stream: None | bool = None, manual_routing=False, think=None, image_path=None):
        query, model_name = await self.route_query(query, manual_routing)
        model = self.models[model_name]

        if stream is None:
            stream = self.platform not in STREAM_DISABLED

        await log(f"Using stream={stream} for {model_name}", "info")

        content_final = ""
        thinking_final = ""
        tools_called = None

        async with self.context_lock:
            context = list(self.context)
       
        if self.summary: context.insert(0, {"role": "assistant", "content": f"[PERSISTED MEMORY – NOT DIALOGUE]\n[INTERNAL MEMORY — DO NOT REPEAT — NOT PART OF CONVERSATION]\nThe system generated summary of previous messages/turns. **Do NOT** treat this as the part of the conversation. For reference only. \n<SUMMARY>\n{self.summary}\n</SUMMARY>"}) # role:system is fatal.

        await self.event_manager.emit_async("generation_started", model_name=model_name, query=query)


        if stream:
            queue = asyncio.Queue()
            async def producer():
                async for (thinking_chunk, content_chunk, tools_chunk) in model.generate(query, context, True,think=think, image_path=image_path):
                    if content_chunk == ERROR_TOKEN:
                        break
                    await queue.put((thinking_chunk, content_chunk, tools_chunk))
                await queue.put(None)

            task = asyncio.create_task(producer())
            
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

        await self.event_manager.emit_async("execute_tools_requested", tools=tools_called) # too lazy to update self.context here.
       
        await self.event_manager.emit_async_block("save_context_requested") # just extra safe. 

    async def execute_tools(self, tools):
        results = {}
        type_ = 'silent'
        if not tools:
            return results

        for tool in tools:
            if not isinstance(tool, dict) or 'function' not in tool:
                await log(f"Invalid tool format: {tool}", "warn")
                continue

            await log(f"Executing tool: {tool['function'].get('name')}", "info")    
            tool_name = tool['function'].get('name')
            tool_args = tool['function'].get('arguments', {})

            if tool_name not in self.available_tools:
                await log(f"Tool '{tool_name}' not available. Available tools: {list(self.available_tools.keys())}", "warn")
                results[tool_name] = f"Tool '{tool_name}' not found"
                continue
            tool_obj = self.tools_regis.get(tool_name)
            if tool_obj is None:
                try:
                    results[tool_name] = await self.available_tools[tool_name](**tool_args)
    
                    await self.event_manager.emit_async("tool_executed", tool = tool_obj)
    
                    await log(f"Tool '{tool_name}' executed successfully.", "success")
                except Exception as e:
                    await log(f"Error executing tool '{tool_name}': {e}", "error")
                    results[tool_name] = f"Error executing tool '{tool_name}': {e}"
            elif tool_obj is not None and isinstance(tool_obj, Tool):
                results[tool_name] = await tool_obj.execute(**tool_args)
            else:
                await log(f"Something terrible happened while executing tool: {tool_name}", 'error')
                continue
            
            async with self.context_lock:
                    self.context.append(
                    {'role':'tool', 'tool_name':tool_name, 'content':str(results[tool_name])}
                    )
            if tool_obj is not None: 
                await self.event_manager.emit_async("tool_executed", tool = tool_obj)
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
