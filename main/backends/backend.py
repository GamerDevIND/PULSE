from main.models.model_instance import LocalModel, LocalEmbedder
from main.models.models_profile import RemoteModel, RemoteEmbedder
from main.models.openrouter_model import OpenRouterEmbedder, OpenRouterModel
import asyncio
import os
from main.events import EventBus
from main.configs import ENV_READ_PREFIX
from main.generation_session import GenerationSession, DONE, FAIL
from main.tools import ToolRegistry
from main.utils import log
from main.configs import ERROR_TOKEN, EMBEDDING_MODEL_ROLE
import inspect
import json
import aiofiles

class Backend:
    def __init__(self, models_list_path, system_prompts, default_system_prompt, event_bus: None | EventBus = None) -> None:
        self.models_list_path = models_list_path
        self.system_prompts = system_prompts
        self.default_system_prompt = default_system_prompt
        self.models:dict[str, LocalModel | LocalEmbedder | RemoteModel | RemoteEmbedder | OpenRouterEmbedder | OpenRouterModel] = {}
        self.sessions:dict[str, GenerationSession] = {}
        self.running_tasks = set()
        self.event_bus = event_bus
        

    def _create_model_data(self, models_data, model, embedder, override_port = False, overwritten_port=11343, auto_resolve_ports = True, require_key = False):
        for model_data in models_data:
                    role = model_data.get('role')
                    if override_port:
                        port = model_data.get('port', overwritten_port)
                    
                        if port != overwritten_port:
                            if auto_resolve_ports:
                                model_data['port'] = overwritten_port
                            else:
                                raise Exception(f"Port mismatch for {role} (SingleServer Mode!)")

                    key = model_data.get('api_key')
                    if key and key.startswith(ENV_READ_PREFIX): 
                        key = os.environ.get(key[len(ENV_READ_PREFIX):])
                        model_data['api_key'] = key
                    if require_key and not key: raise ValueError(f'No API key provided for {model_data["model_name"]}')

                    if role:
                        if role != EMBEDDING_MODEL_ROLE:
                            system = model_data.get("system_prompt")
                            if not (system and system.strip()):
                                model_data["system_prompt"] = self.system_prompts.get(role, self.default_system_prompt)
                            
                            self.models[role] = model(**model_data)
                        else:
                            self.models[role] = embedder(EMBEDDING_MODEL_ROLE, model_data.get("name", 'Embedder'), model_data['model_name'], model_data.get('port'), key)
                        if override_port:
                            self.models[role].update_port(overwritten_port)
                    else:
                        raise KeyError

    def _load(self, model, embedder, override_port = False, overwritten_port=11343, auto_resolve_ports = True, require_key = False):
        try:
            with open(self.models_list_path, 'r', encoding="utf-8") as f:
                models_data = json.load(f)
                self._create_model_data(models_data, model, embedder, override_port, overwritten_port, auto_resolve_ports, require_key)
    
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            raise Exception("Models loading failed.", e)

    async def _async_load(self, model, embedder, override_port = False, overwritten_port=11343, auto_resolve_ports = True, require_key = False):
        try:
            async with aiofiles.open(self.models_list_path, 'r', encoding="utf-8") as f:
                c = await f.read()
                models_data =  json.loads(c)
                if not isinstance(models_data, list): raise ValueError("The config file must contain a list of the configs.")
                self._create_model_data(models_data, model, embedder, override_port, overwritten_port, auto_resolve_ports, require_key)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            raise Exception("Models loading failed.", e)

    async def cancel_generation(self, session_id:str):
        session = self.sessions.get(session_id)
        if not session: 
            return
        await session.cancel()

    def get_session(self, session_id:str):
        session = self.sessions[session_id]
        return session 

    def get_model(self, role):
        model_obj = self.models.get(role)
        return model_obj
    
    async def _init(self, *tools_list):
        for model in self.models.values():
            if isinstance(model, (LocalModel, RemoteModel, OpenRouterModel)):
                if model.has_tools:
                    await model.add_tools(*tools_list)
                if not model.warmed_up:
                    await model.warm_up()
                else:
                    await log(f"{model.name} ({model.model_name}) is already warmed, skipping... This maybe abnormal, please ensure the initilising logic.", 'warn')
        t = asyncio.create_task(self._check_sessions())
        self.running_tasks.add(t)
        t.add_done_callback(self.running_tasks.discard)

    async def remove_session(self, session_id):
        await self.cancel_generation(session_id)
        if session_id in self.sessions: 
            del self.sessions[session_id]

    async def _check_sessions(self):
        while True:
            to_remove = [s_id for s_id, s in self.sessions.items() if s.state in [DONE, FAIL]]
        
            for s_id in to_remove:
                session = self.sessions.get(s_id)
                if session:
                    await session.cancel()
                    del self.sessions[s_id]
                    await log(f"Session {s_id} cleaned up.", "info", stdout=False)

            await asyncio.sleep(5.5)

    async def create_session(self, query:str, context:list[dict], tools_regis:ToolRegistry, role, system_prompt_override: str | None = None, 
                options: dict | None = None, format_: dict | None = None, max_turns = 10, abs_max_turns = 50, regen_consent_callback= None, temp_remove_tool_name = None):
        
        if role == EMBEDDING_MODEL_ROLE:
            await log('Cannot create a session with an embedding model!', 'error')
            raise

        model_obj = self.models.get(role)

        if not model_obj:
            await log(f"{role} not found in the model registry!","error")
            raise

        active_tools = None

        if isinstance(model_obj, (OpenRouterModel, LocalModel, RemoteModel)):
            active_tools = list(model_obj.tools)

        if temp_remove_tool_name and active_tools:
            active_tools = [t for t in active_tools if t.get('function', {}).get('name') != temp_remove_tool_name]
            await log(f"Temporarily disabled {temp_remove_tool_name} for this turn.", "info")
        
        session = GenerationSession(query, context, tools_regis, model_obj, # type:ignore
                system_prompt_override, options, format_, max_turns, abs_max_turns, regen_consent_callback, active_tools, self.event_bus)
        
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


class Generation:
    def __init__(self, session:GenerationSession, remove_callback, append_callback, stream, user_save_prefix, think, 
                 file_path, video_frames_mod, save_thinking, event_bus: None | EventBus = None) -> None:
        
        '''
        Note: `remove_callback` MUST cancel the given session through `session_id` before removal.
        '''

        self.session = session
        self.session_id = session.id
        self.remove_callback = remove_callback
        self.append_callback = append_callback
        self.stream_ = stream
        self.user_save_prefix = user_save_prefix
        self.think = think
        self.file_path = file_path
        self.video_frames_mod = video_frames_mod
        self.save_thinking = save_thinking
        self.tools_override =  None
        self.event_bus = event_bus
    
    async def stream(self, ):
        if self.event_bus:
            await self.event_bus.sequence_emit(self.event_bus.GENERATION_STARTED, gen_id = self.session_id)
        try:
            async for (thinking, content) in self.session.generate(self.stream_, self.user_save_prefix, self.think, 
                                                                   self.file_path, self.video_frames_mod, self.save_thinking):
                        
                if content == ERROR_TOKEN:
                    return
                    
                yield (thinking or "", content or "")

        finally:
            c = await self.session.get_context()
            r = self.remove_callback(self.session_id)
            if inspect.isawaitable(r):
                await r
                
            r = self.append_callback(c)
            if inspect.isawaitable(r):
                await r
        
        if self.event_bus:
            await self.event_bus.sequence_emit(self.event_bus.GENERATION_STOPPED , gen_id = self.session_id)

    async def terminate(self):
        if self.event_bus:
            await self.event_bus.sequence_emit(self.event_bus.GENERATION_STOPPED,  gen_id = self.session_id)
        r = self.remove_callback(self.session_id)
        if inspect.isawaitable(r):
            await r

        if self.event_bus:
            await self.event_bus.sequence_emit(self.event_bus.GENERATION_STOPPED, gen_id = self.session_id)