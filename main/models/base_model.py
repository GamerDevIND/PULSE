import asyncio
from main.resource_manager import SessionManager, ResourceManager
from main.tools import Tool
import inspect
from typing import Literal, Type
from main.utils import log

class Model:
    def __init__(self, role, host, name: str, model_name: str, api_key:str | None, init_state, **kwargs) -> None:
        '''
        `resource_manager` attribute defaults to `SessionManager`
        '''
        self.role = role
        self.name = name
        self.model_name = model_name
        self.host = host
        self.warmed_up = False
        self.state_lock = asyncio.Lock()
        self.tools = []
        self.state = init_state
        self.api_key = api_key
        self.port:int | None = None
        self.details_cache = None
        self.resource_manager: ResourceManager | SessionManager = SessionManager(model_name)

    async def __aenter__(self):
        await self.warm_up()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    async def warm_up(
        self,
        use_mmap=False,
        warmup_image_path="main/test.jpg",
        use_custom_keep_alive_timeout=True,
        custom_keep_alive_timeout: str = "-1",
        has_video_processing=False,
        warmup_video_path="main/test.mp4"): 
        raise NotImplementedError
    
    async def generate(self, query: str, context: list[dict], stream: bool, think: str | bool | None = False, image_path: None | str = None, 
                   mod_ = 1, system_prompt_override: str | None = None):

        raise NotImplementedError

    async def shutdown(self):
        raise NotImplementedError
    
    async def clear_details_cache(self):
        async with self.state_lock: self.details_cache = None

    async def change_state(self, new, use_lock = True):
        if use_lock:
            async with self.state_lock: 
                self.state = new
        else:
            self.state = new

    async def _get_model_details_helper(self, res:dict, default_capability = "completion", auto_add_completion = True):
        capabilities:list = res.get("capabilities", [default_capability])
        inp = res.get("input_modalities", [])
        if "image" in inp:
            capabilities.append("vision")
        if auto_add_completion and ('text' in inp and not 'completion' in capabilities):
            capabilities.append("completion")

        modified_at = res.get("modified_at", "")
        details = res.get("details", {})
        parent_model = details.get("parent_model", "")
        family = details.get("family", "")
        parameter_size = details.get("parameter_size", "")
        quantization_level = details.get("quantization_level", "")
        model_info = res.get("model_info", {})
        architecture = model_info.get("general.architecture", "")
        parameter_count = model_info.get("general.parameter_count", -1)

        out = {
            "capabilities": capabilities,
            "modified_at": modified_at,
            "parent_model": parent_model,
            "family": family,
            "parameter_size": parameter_size,
            "quantization_level" : quantization_level,
            "architecture": architecture,
            "parameter_count": parameter_count,
        }
        async with self.state_lock : self.details_cache = out
        return out
    
    async def add_tools(self, *tool_dicts):
        for tool in tool_dicts:
            if isinstance(tool, Tool):
                self.tools.append(tool.schema)
            elif isinstance(tool, dict):
                self.tools.append(tool)
            elif callable(tool) and not inspect.isclass(tool):
                needs_regeneration:bool = False
                retry: Literal["none", "always", ] = "none"
                max_retries: int = 0
                retry_on: tuple[Type[Exception], ...] | None = None
                visible_to_user: bool = True
                metadata = {
                    "needs_regeneration": needs_regeneration,
                    "retry": retry,
                    "max_retries": max_retries,
                    "retry_on": retry_on or (),
                    "visible_to_user": visible_to_user,
                }
                tool = Tool(tool, metadata)
                self.tools.append(tool.schema)
            else:
                await log(f"No valid type dectected for: {tool}", 'warn')