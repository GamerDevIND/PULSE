from .model_instance import LocalModel
from .models_profile import RemoteModel
import asyncio
from .tools import ToolRegistry, Tool
from typing import Iterable
from .utils import log
from copy import deepcopy
import uuid
import datetime
import inspect
from configs import ERROR_TOKEN, FILE_NAME_KEY

CREATED = "CREATED"
DONE = "DONE"
FAIL = "FAIL"
TOOLING = "TOOLS"
CANCELLED = "CANCELLED"
GENERATING = "GENERATING"
CANCELLING = "CANCELLING"

class GenerationSession:
    def __init__(self, query:str, context:list[dict], tools_regis:ToolRegistry, model: LocalModel | RemoteModel, system_prompt_override: str | None = None, 
                options: dict | None = None, format_: dict | None = None, max_turns = 10, abs_max_turns = 50, regen_consent_callback= None) -> None:
        self.query = query
        self.context = context
        self.tools_regis = tools_regis
        self.model = model
        self.state = CREATED
        self.id = str(uuid.uuid4())
        self.created_at = datetime.datetime.now().timestamp()
        self.generation_task = None
        self.sys_override = system_prompt_override
        self.opts = options
        self.format = format_

        self.turns = 0
        self.total_turns = 0
        self.max_turns = max_turns
        self.abs_max_turns = abs_max_turns

        self.state_lock = asyncio.Lock()
        self.context_lock = asyncio.Lock()
        self.regen_consent_callback = regen_consent_callback
        self.regen = False
        asyncio.create_task(log(f"Generation Session created at {self.created_at}", 'info'))

    async def change_state(self, new, use_lock = True):
        if use_lock:
            async with self.state_lock: 
                self.state = new
        else:
            self.state = new

    async def generate(self,  stream: bool, user_save_prefix= None, think: str | bool | None = False, image_path: None | str = None,  mod_ = 10, save_thinking= True):
        if image_path and self.model.role != "vision": 
            await log("Image / Video path provided but no vision model chosen. Role changing to Vision.", 'warn')

            if self.state in [GENERATING, TOOLING]:
                await self.cancel()
                await self.change_state(FAIL)

            yield (ERROR_TOKEN, ERROR_TOKEN)
            return

        await self.change_state(GENERATING)

        content_final = ""
        thinking_final = ""
        tools_called = []
        query = f"{user_save_prefix if user_save_prefix else ""}{self.query}",

        if stream:
            queue = asyncio.Queue(maxsize=256)
            async def producer():
                try:
                    async for (thinking_chunk, content_chunk, tools_chunk) in self.model.generate(query, self.context, True,
                                                                                                        think=think, image_path=image_path, mod_ = mod_, 
                                                                                                        system_prompt_override=self.sys_override, ):
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
                    tools_called.extend(tools_chunk)

                thinking_final += thinking_chunk or ""
                content_final += content_chunk or ""

                yield (thinking_chunk or "", content_chunk or "")

            await task            
        else:
            await log(f"Non-streaming mode active", "info")
            async for (thinking, content, tools) in self.model.generate(query, self.context, False, think=think, 
                                                                              image_path=image_path, mod_ = mod_, system_prompt_override=self.sys_override):
                await log(f"Got non-streaming response chunk", "info")
                if content == ERROR_TOKEN:
                    await self.change_state(FAIL)
                    break

                if tools:
                    tools_called.extend(tools)

                thinking_final += thinking or ""
                content_final += content or ""

                yield (thinking or "", content or "")

        if self.query and self.query.strip(): 
            async with self.context_lock:
                self.context.append({'role': 'user', 'content': query, FILE_NAME_KEY : image_path})

            if save_thinking:
                m = {'role': 'assistant', 'thinking': thinking_final, 'content': content_final, 'tool_calls': tools_called} 
            else: 
                m = {'role': 'assistant','content': content_final, 'tool_calls': tools_called}

            async with self.context_lock:
                self.context.append(m)

        self.generation_task = None
        await self.execute_tools(tools_called)

        await self.change_state(DONE)

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

    async def cancel(self):
        await self.change_state(CANCELLING)

        if self.model:
            self.model.cancel()
            if self.generation_task:
                if not self.generation_task.done():
                    self.generation_task.cancel()
                    try:
                        await self.generation_task
                    except asyncio.CancelledError:
                        pass
                    self.generation_task = None

        await self.change_state(CANCELLED)

    async def get_context(self):
        async with self.context_lock:
            return deepcopy(self.context)

    async def set_context(self, context):
        async with self.context_lock:
            self.context = context

    async def run_generation_loop(self, stream, user_save_prefix= None, think = False, file_path=None, mod_= 10):
        while True:
            async for (thinking, response) in self.generate(stream, user_save_prefix, think, file_path, mod_, True): 
                yield thinking, response
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
        await self.change_state(TOOLING)
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
                await log(f"An error occured in the tools execution; Tool Name: {tool_name} Skipping...", 'warn')
                continue

            result, tool_obj = result
            result["index"] = tool_idx  
            results.append(result)  
            tools_objs.append(tool_obj)  

        async with self.context_lock: 
            self.context.extend(results)

        await self._check_regen(tools_objs)   
        return results