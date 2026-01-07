from datetime import datetime
import os
import aiofiles
import inspect
from typing import Any, get_origin, get_args, Union, List, Dict, Literal

async def log(message: str, level, append = True, stdout = True):
    log_dir = os.path.join( "main", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime(" %H:%M:%S - %d / %m  / %Y ")
    emoji = {
        "info": "ℹ️",
        "warn": "⚠️",
        "error": "🟥",
        "success": "✅"
    }.get(level, "")
    text = f"{emoji} [{level}] {message} - [{timestamp}]\n"
    log_file = os.path.join(log_dir, 'log.log')
    async with aiofiles.open(log_file, "w" if not  append else "a") as f:
        if stdout: print(text)
        await f.write(text)



async def schemaify(python_type: Any) -> dict: # Before you ask: yes, this is AI generated
    origin = get_origin(python_type)

    if python_type is str:  
        return {"type": "string"}  
    elif python_type is int:  
        return {"type": "integer"}  
    elif python_type is float:  
        return {"type": "number"}  
    elif python_type is bool:  
        return {"type": "boolean"}  
    elif python_type is type(None):  
        return {"type": "null"}  

    elif origin is Literal:  
        return {"enum": list(get_args(python_type))}  
    
    elif origin is Union:  
        args = get_args(python_type)  
        if len(args) == 2 and type(None) in args:   
            non_none_type = next(arg for arg in args if arg is not type(None))  
            return await schemaify(non_none_type)  
        else:  
            return {"anyOf": [await schemaify(arg) for arg in args]}  
    
    elif origin in (list, List):  
        args = get_args(python_type)  
        if args:  
            return {  
                "type": "array",  
                "items": await schemaify(args[0])  
            }  
        else:  
            return {"type": "array"}
   
    elif origin in (dict, Dict):  
        args = get_args(python_type)  
        if len(args) == 2:  
            return {  
                "type": "object",  
                "additionalProperties": await schemaify(args[1])  
            }  
        else:  
            return {"type": "object"}
    
    elif origin is tuple:  
        args = get_args(python_type)  
        if len(args) == 1:  
            return {  
                "type": "array",  
                "items": await schemaify(args[0])  
            }  
        elif len(args) > 1:  
            return {  
                "type": "array",  
                "prefixItems": [await schemaify(a) for a in args],  
                "items": False  
            }  
        else:  
            return {"type": "array"}  
    
    elif inspect.isclass(python_type):  
        return {"type": "object"}  
    else:
        return {"type": "string"}

async def convert_funcs(*funcs):
    jsons = []

    for func in funcs:
        name = func.__name__
        desc = inspect.getdoc(func) or "No description provided."
        sig = inspect.signature(func)

        properties = {}  
        required = []  

        for arg_name, param in sig.parameters.items():
            annotation = param.annotation if param.annotation != inspect._empty else str
            properties[arg_name] = await schemaify(annotation)

            if "description" not in properties[arg_name]:
                properties[arg_name]["description"] = f"The {arg_name} parameter."

            if param.default == inspect._empty:
                required.append(arg_name)

        jsons.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        })
    return jsons

async def load_tools(*tools_type_iterable):
    tools = []
    for tool, type_ in tools_type_iterable:
        tool_created = await Tool.create(tool, type_)
        tools.append(tool_created)

    return tools
    
class Tool:
    def __init__(self, func, type_):
        self.func = func
        self.name = self.func.__name__
        self.type = type_
        self.schema = None

    @classmethod
    async def create(cls, func, type_):
        instance = cls(func, type_)
        instance.schema = (await convert_funcs(instance.func))[0]
        return instance

    async def execute(self, **args):
        try:
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(**args)
            else:
                result = self.func(**args)
        except Exception as e:
            result = f"An Error occurred while calling {self.name}: {str(e)}"

            await log(f"An Error occurred while calling {self.name}: {str(e)}", 'error')

        return result
