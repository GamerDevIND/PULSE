from .utils import convert_funcs, log
import inspect 
import threading
import asyncio

from typing import Literal, Type

def tool(
    *,
    needs_regeneration:bool = False,
    retry: Literal["none", "always",] = "none",
    max_retries: int = 0,
    retry_on: tuple[Type[Exception], ...] | tuple = (),
    timeout_secs:float | Literal['inf'] = 'inf',
):
    def decor(func):
        metadata = {
            "needs_regeneration": needs_regeneration,
            "retry": retry,
            "max_retries": max_retries,
            "retry_on": retry_on or tuple(),
            "timeout": timeout_secs
        }
        func.__meta__ = metadata
        ToolRegistry.register_global(func, metadata)
        return func
    return decor

class Tool:
    def __init__(self, func, meta):
        self.func = func
        self.name = self.func.__name__
        self.schema = convert_funcs(self.func)[0]
        self.needs_regeneration:bool = meta["needs_regeneration"]
        self.max_retry = meta["max_retries"]
        self.retry_on = meta["retry_on"]
        self.retry = meta['retry']
        self.timeout = meta['timeout']

    async def execute(self, **kwargs):
        retries = 0
        error = f"An error occurred executing {self.name}" # fallback 
        retry_on = tuple(self.retry_on) or tuple()

        total_attempts = self.max_retry + 1
        while retries < total_attempts:
            try:

                if self.timeout == 'inf':
                    if inspect.iscoroutinefunction(self.func):
                        r = await self.func(**kwargs)
                    else:
                        r = await asyncio.to_thread(self.func, **kwargs) # type:ignore
                else:
                    if inspect.iscoroutinefunction(self.func):
                        r = await asyncio.wait_for(self.func(**kwargs), self.timeout)
                    else:
                        r  = await asyncio.wait_for(asyncio.to_thread(self.func, **kwargs), self.timeout) # type:ignore

                if inspect.isawaitable(r):
                    r = await r

                return None, r
            except asyncio.TimeoutError as e:
                if self.retry == "none": return True, f"{self.name} timed out"
                retries += 1
                error = f"{self.name} timed out on retry: {retries}: {repr(e)}"
            except Exception as e: 
                if retry_on:
                    if isinstance(e, retry_on):
                        if self.retry == "none": 
                            return True, repr(e)
                        retries += 1
                        error = repr(e)
                    else:
                        return True, repr(e)
                else:
                    return True, repr(e)

        return True, error

class ToolRegistry:
    _tools:dict[str, Tool] = {}
    _lock = threading.RLock()

    @classmethod
    def register_global(cls, func, meta):
        with cls._lock:
            if func.__name__ in cls._tools:
                print(f"Tool {func.__name__} already registered!")
                print(f"Tools listed: {ToolRegistry._tools.keys()}")
                return
            print(f"Registering: {func.__name__}")
            cls._tools[func.__name__] = Tool(func, meta)

    def __init__(self):
        with ToolRegistry._lock:
            self.tools = ToolRegistry._tools.copy()

    def clear(self):
        with ToolRegistry._lock: self.tools.clear()

    def clear_global(self):
        with ToolRegistry._lock: ToolRegistry._tools.clear()

    def add_tool_local(self, func, meta):
        with ToolRegistry._lock:
            if func.__name__ in self.tools:
                print(f"Tool {func.__name__} already registered!")
                print(f"Tools listed: {ToolRegistry._tools.keys()}")
                return
            print(f"Registering: {func.__name__}")
            self.tools[func.__name__] = Tool(func, meta)

    def list_tools(self):
        return list(self.tools.values())

    def list_tools_global(self):
        return list(ToolRegistry._tools.values())

    async def execute_tool(self, tool:Tool | None = None, tool_name:str | None = None, **kwargs):
        if tool_name and tool is None: tool = self.tools.get(tool_name)
        if tool_name and tool: 
            if tool_name != tool.name: 
                await log("Tool and provided tool name do not match. Skipping this item.", "warn")
                return {"role": "tool", "tool_name": tool.name, "content": "Tool and provided tool name do not match."}, tool

        if not tool:
           await log("tool not found / provided. skipping execution", "warn")
           return {"role": "tool", "tool_name": "N/A", "content": "Error: Tool not found. Answer directly or request a valid tool"}, tool

        error, result = await tool.execute(**kwargs)

        if error:
           await log(f"An error occurred while trying to execute Tool: {tool.name}\nMessage:{result}", "error")
           result = f"An error occurred while trying to execute Tool: {tool.name}\nMessage:{result}"

        data = {"role": "tool", "tool_name": tool.name, "content": result}

        return data, tool

    def get(self, name):
        if name not in self.tools: raise Exception(f"{name} not available in the registry.")
        return self.tools[name]

    def __getitem__(self, name):
        if name not in self.tools: raise Exception(f"{name} not available in the registry.")
        return self.tools[name]

from .plugins import *
