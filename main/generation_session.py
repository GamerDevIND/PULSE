from .models.model_instance import LocalModel
from .models.models_profile import RemoteModel
import asyncio
from .tools import ToolRegistry, Tool
from typing import Iterable
from .utils import log, strip_thinking
import json
from copy import deepcopy
import uuid
import datetime
import inspect
from .configs import ERROR_TOKEN, FILE_NAME_KEY, INSTANT_TOOL_EXEC
from .events import EventBus

CREATED = "CREATED"
DONE = "DONE"
FAIL = "FAIL"
TOOLING = "TOOLS"
CANCELLED = "CANCELLED"
GENERATING = "GENERATING"
CANCELLING = "CANCELLING"

class GenerationSession:
    def __init__(self, query:str, context:list[dict], tools_regis:ToolRegistry, model: LocalModel | RemoteModel, system_prompt_override: str | None = None, 
                options: dict | None = None, format_: dict | None = None, max_turns = 10, abs_max_turns = 50, regen_consent_callback= None, tools_override=None, event_bus: None | EventBus = None,) -> None:
        self.query = query
        self.original_context = context
        self.context:list[dict] = []
        self.tools_regis = tools_regis
        self.model = model
        self.state = CREATED
        self.id = str(uuid.uuid4())
        self.created_at = datetime.datetime.now().timestamp()
        self.generation_task = None
        self.sys_override = system_prompt_override
        self.options = options
        self.format = format_
        self.tools_override = tools_override

        self.turns = 0
        self.total_turns = 0
        self.max_turns = max_turns
        self.abs_max_turns = abs_max_turns

        self.state_lock = asyncio.Lock()
        self.context_lock = asyncio.Lock()

        self.regen_consent_callback = regen_consent_callback
        self.regen = False
        self.event_bus = event_bus
    
    async def change_state(self, new, use_lock = True):
        if use_lock:
            async with self.state_lock: 
                self.state = new
        else:
            self.state = new

    async def generate(self,  stream: bool, user_save_prefix= None, think: str | bool | None = False, image_path: None | str = None,  mod_ = 10, save_thinking= True):
        if image_path and self.model.role != "vision": 
            await log("Image / Video path provided but no vision model chosen. Aborting...", 'warn')

            if self.state in [GENERATING, TOOLING]:
                await self.cancel()
                await self.change_state(FAIL)

            yield (ERROR_TOKEN, ERROR_TOKEN, [])
            return
        
        await self.change_state(GENERATING)

        content_final = ""
        thinking_final = ""
        tools_called = []
        query = f"{user_save_prefix + " " if user_save_prefix else ""}{self.query}"

        async with self.context_lock:
            context = self.original_context + self.context

        if stream:
            queue = asyncio.Queue(maxsize=256)
            async def producer():
                try:
                    async for (thinking_chunk, content_chunk, tools_chunk) in self.model.generate(query, context, True,
                                                                                                        think=think, image_path=image_path, mod_ = mod_, 
                                                                                                        system_prompt_override=self.sys_override, options=self.options,
                                                                                                        format_=self.format,  tools_override=self.tools_override):

                        if content_chunk == ERROR_TOKEN:
                            await self.change_state(FAIL)
                            
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

                if tools_chunk:
                    if INSTANT_TOOL_EXEC:
                        await self.execute_tools(tools_chunk)
                    else:
                        tools_called.extend(tools_chunk)

                thinking_final += thinking_chunk or ""
                content_final += content_chunk or ""

                tool_names = [t.get('function', {}).get('name', "") for t in tools_chunk] if tools_chunk else []

                yield (thinking_chunk or "", content_chunk or "", tool_names)
                if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.GENERATION_CHUNK, False, chunk = (thinking_chunk or "", content_chunk or "")) 
            
            try:
                await task  
            except asyncio.CancelledError:
                await self.change_state(CANCELLED)
                raise  

        else:
           async for (thinking_chunk, content_chunk, tools_chunk) in self.model.generate(query, context, False,
                                                                                                        think=think, image_path=image_path, mod_ = mod_, 
                                                                                                        system_prompt_override=self.sys_override, options=self.options,
                                                                                                        format_=self.format,  tools_override=self.tools_override):

                await log(f"Got non-streaming response chunk", "info")
                if content_chunk == ERROR_TOKEN:
                    await self.change_state(FAIL)
                    break

                if tools_chunk:
                    if INSTANT_TOOL_EXEC:
                        await self.execute_tools(tools_chunk)
                    else:
                        tools_called.extend(tools_chunk)

                thinking_final += thinking_chunk or ""
                content_final += content_chunk or ""

                tool_names = [t.get('function', {}).get('name', "") for t in tools_chunk] if tools_chunk else []

                yield (thinking_chunk or "", content_chunk or "", tool_names)
                if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.GENERATION_CHUNK, False, chunk = (thinking_chunk or "", content_chunk or "")) 

        if self.query and self.query.strip(): 
            async with self.context_lock:
                self.context.append({'role': 'user', 'content': query, FILE_NAME_KEY : image_path})
        m = None
        if not thinking_final.strip():
            c, t = strip_thinking(content_final)
            if c:
                content_final = c
            if t:
                thinking_final = t

        if (thinking_final) or (content_final) or (tools_called):
            if save_thinking:
                m = {'role': 'assistant', 'thinking': thinking_final, 'content': content_final, 'tool_calls': tools_called} 
            else: 
                m = {'role': 'assistant','content': content_final, 'tool_calls': tools_called}

        if m:
            async with self.context_lock:
                self.context.append(m)

        self.generation_task = None
        if not INSTANT_TOOL_EXEC: 
            await self.execute_tools(tools_called)
        
        if not self.regen:
            await self.change_state(DONE)
        
    async def _ask_user_consent(self):
        if not self.regen_consent_callback: return False
        true_calls = [True, 1, "1" ,"true", "y", "yes", "consent", "confirm",]
        try:
            if self.event_bus: await self.event_bus.sequence_emit(self.event_bus.PROPOSED_REGEN)
            result = self.regen_consent_callback() # type:ignore
            if inspect.isawaitable(result): result = await result 
            if type(result) == str: result = result.lower().strip()
            return result in true_calls
        except Exception as e:
            await log(f"An error occurred while calling regeneration consent callback: {e}", "warn")
            return False

    async def cancel(self):

        if self.state in [CANCELLED, CANCELLING, DONE]:
            return

        await self.change_state(CANCELLING)
        try:
                
            if self.generation_task:
                if not self.generation_task.done():
                    self.generation_task.cancel()
                    try:
                        await self.generation_task
                    except asyncio.CancelledError:
                        pass
                    self.generation_task = None
        finally:
            await self.change_state(CANCELLED)
            if self.event_bus:
                await self.event_bus.sequence_emit(self.event_bus.GENERATION_CANCELLED, gen_id = self.id)

    async def get_context(self, include_original = False):
        async with self.context_lock:
            return deepcopy((self.original_context + self.context) if include_original else self.context) 
    
    async def set_context(self, context, set_original = False):
        async with self.context_lock:
            if set_original:
                self.original_context = context
            else:
                self.context = context

    async def run_generation_loop(self, stream, user_save_prefix= None, think = False, file_path=None, mod_= 10):
        self.regen = False
        while True:
            async for (thinking, response, tool) in self.generate(stream, user_save_prefix, think, file_path, mod_, True): 
                yield thinking, response, tool
            if not self.regen: break

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

    async def execute_tools(self, tools,):
        results = []  
        tools_objs = []  
        if not tools:
            return results  
        
        await self.change_state(TOOLING)
  
        for tool in tools:  
            if not isinstance(tool, dict) or 'function' not in tool:  
                await log(f"Invalid tool format: {tool}", "warn")  
                continue  
            func = tool['function']
            await log(f"Executing tool: {func.get('name')}", "info")      
            tool_name = func.get('name')  
            call_id = tool.get('id')
            c_id = tool.get('call_id')
            tool_idx = func.get('index')
            if self.event_bus:
                await self.event_bus.sequence_emit(self.event_bus.TOOL_EXECUTING, tool_name= tool_name, session_id = self.id)

            tool_args = func.get('arguments', {}) 
            
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    await log(f"Failed to parse tool arguments for {tool_name}: {tool_args}", "error")
                    continue
            
            if not isinstance(tool_args, dict):
                await log(f"Arguments for {tool_name} are not a dictionary. Skipping...", "warn")
                continue
  
            if not tool_name in self.tools_regis.tools.keys():  
                await log(f"{tool_name} is not in the registory. Skipping...", 'warn')  
                continue  
  
            result = await self.tools_regis.execute_tool(tool_name=tool_name, **tool_args)
            if result is None:
                await log(f"An error occured in the tools execution; Tool Name: {tool_name} Skipping...", 'warn')
                continue

            result, tool_obj = result

            if call_id and not tool_idx:
                result["tool_call_id"] = call_id
                if c_id: result["call_id"] = c_id
            
            if not call_id and tool_idx:
                result["index"] = tool_idx  

            if result.get("role") != "tool":
                await log(f"Invalid tool result role: {result}", "error")
                continue

            results.append(result)  
            tools_objs.append(tool_obj)  

        async with self.context_lock: 
            self.context.extend(results)
          
        await self._check_regen(tools_objs)   
        if self.event_bus and (tools_objs or results):
            await self.event_bus.sequence_emit(self.event_bus.TOOLS_EXECUTED, session_id = self.id)
        return results