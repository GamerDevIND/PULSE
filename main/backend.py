from .model_instance import LocalModel
from .models_profile import RemoteModel
import asyncio
from .generation_session import GenerationSession, DONE, FAIL
from .tools import ToolRegistry
from .utils import log

class Backend:
    def __init__(self, models_list_path, system_prompts, default_system_prompt) -> None:
        self.models_list_path = models_list_path
        self.system_prompts = system_prompts
        self.default_system_prompt = default_system_prompt
        self.models:dict[str, LocalModel | RemoteModel] = {}
        self.sessions:dict[str, GenerationSession] = {}
        self.running_tasks = set()

    async def cancel_generation(self, session_id:str):
        session = self.sessions[session_id]
        await session.cancel()

    def get_session(self, session_id:str):
        session = self.sessions[session_id]
        return session 

    def get_model(self, role):
        model_obj = self.models.get(role)
        return model_obj
    
    def _init(self):
        t = asyncio.create_task(self._check_sessions())
        self.running_tasks.add(t)

    async def remove_session(self, session_id):
        await self.cancel_generation(session_id)
        del self.sessions[session_id]

    async def _check_sessions(self):
        while True:
            for s in list(self.sessions.values()):
                if s.state in [DONE, FAIL]:
                    await self.cancel_generation(s.id)
                    del self.sessions[s.id]

            await asyncio.sleep(5.5)

    async def create_session(self, query:str, context:list[dict], tools_regis:ToolRegistry, role, system_prompt_override: str | None = None, 
                options: dict | None = None, format_: dict | None = None, max_turns = 10, abs_max_turns = 50, regen_consent_callback= None):
        
        model_obj = self.models.get(role)

        if not model_obj:
            await log(f"{role} not found in the model registry!","error")
            raise
        
        session = GenerationSession(query, context, tools_regis, model_obj, 
                system_prompt_override, options, format_, max_turns, abs_max_turns, regen_consent_callback)
        
        session_id = session.id
        
        self.sessions[session_id] = session
        return session_id, session

    async def close_sessions(self):
        for s in self.sessions.values():
            await s.cancel()
        
        for task in list(self.running_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass