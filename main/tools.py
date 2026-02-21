from .utils import convert_funcs, log
from .configs import ERROR_TOKEN
import inspect 

from typing import Literal, Type

def tool(
    *,
    needs_regeneration:bool = False,
    retry: Literal["none", "always", ] = "none",
    max_retries: int = 0,
    retry_on: tuple[Type[Exception], ...] | None = None,
    visible_to_user: bool = True,
):
    def decor(func):
        metadata = {
            "needs_regeneration": needs_regeneration,
            "retry": retry,
            "max_retries": max_retries,
            "retry_on": retry_on or (),
            "visible_to_user": visible_to_user,
        }
        func.__meta__ = metadata
        ToolRegistry.register(func, metadata)
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
        self.visible_to_user = meta["visible_to_user"]

    async def execute(self, **args):
        retries = 0
        error = f"An error occurred executing {self.name}" # fallback 
        retry_on = tuple(self.retry_on) or ()
        if len(retry_on) < 1:
            try:
                await log(f"{self.name} has 0 retries exceptions set, updating to Exception retry fallback", "warn")
            finally: pass
            retry_on = retry_on or (Exception,)

        total_attempts = self.max_retry + 1
        while retries < total_attempts:
            try:
                if inspect.iscoroutinefunction(self.func):
                    r = await self.func(**args)
                else:
                    r = self.func(**args)
                if inspect.isawaitable(r):
                    r = await r
                return None, r

            except retry_on as e:
                if self.retry == "none": return ERROR_TOKEN, repr(e)
                retries += 1
                error = repr(e)
            except Exception as e: return ERROR_TOKEN, repr(e)
        return ERROR_TOKEN, error

class ToolRegistry:
    _tools = {}

    @classmethod
    def register(cls, func, meta):
        cls._tools[func.__name__] = Tool(func, meta)

    def __init__(self):
        self.tools = ToolRegistry._tools.copy()
 

    def clear(self):
        self.tools.clear()
     
    def clear_global(self):
        ToolRegistry._tools.clear()

    def list_tools(self):
        return list(self.tools.values())
    
    def list_tools_global(self):
        return list(ToolRegistry._tools.values())

    async def execute_tool(self,tool:Tool | None = None, tool_name:str | None = None, **args):
        if tool_name and tool is None: tool = self.tools.get(tool_name)
        if tool_name and tool: 
            if tool_name != tool.name: 
                await log("Tool and provided tool name do not match. Skipping this item.", "warn")
                return

        if not tool:
           await log("tool not found / provided. skipping execution", "warn")
           return

        error, result = await tool.execute(**args)

        if error == ERROR_TOKEN:
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