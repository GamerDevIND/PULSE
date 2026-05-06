from .utils import Logger
import inspect
import asyncio
import threading
from datetime import datetime

class EventBus:
    GENERATION_CHUNK = "generation chunk"
    GENERATION_CANCELLED = "generation cancelled"
    TOOL_EXECUTING = "tool executing"
    TOOLS_EXECUTED = "tools executed"
    GENERATION_STARTED = "generation started"
    GENERATION_STOPPED = "generation stopped"
    GARBAGE_COLLECTOR = 'garbage collector'
    GARBAGE_COLLECTED  = 'garbage collected'
    SUMMARISING = "summarising"
    SUMMARISING_FAILED = "summarisation failed"
    SUMMARISED = 'summarised'
    INITIALISING = "initialising"
    INITIALISED = "initialised"
    PROPOSED_MEMORY = 'proposed memory'
    PROPOSED_REGEN = "propose regen"
    CANCELLING_SESSION = 'cancelling session'
    RETRIEVING_MEMORY = "retrieving memory"
    CREATING_SESSION = "creating session"
    ROUTING = "routing"
    ROUTING_ROLE = "routing role"
    SESSION_CREATED = 'session created'
    SHUTTING_DOWN = 'shutting down'
    SHUTDOWN = 'shut down'

    MODELS_LOADING = "loading models"
    MODELS_LOADED = "loaded models"
    INFO = "info"
    WARN = "warn"
    WARNING = "warn"
    ERROR = "error"
    SUCCESS = "success"

    def  __init__(self) -> None:
        self.listeners:dict[str, set] = {}
        self._lock_sync = threading.RLock()
        self._lock_async = asyncio.Lock()
        self.wild_listeners = set()

    def add_listener(self, event_name:str, listener):
        Logger.log_sync(f"Adding {listener.__name__} to event '{event_name}'", 'info')
        with self._lock_sync:
            if event_name == '*':
                if not listener.__name__ in self.wild_listeners:
                    self.wild_listeners.add(listener)
            else:
                if not self.listeners.get(event_name):
                    self.listeners[event_name] = {listener}
                else:
                    self.listeners[event_name].add(listener)

    def remove_listener(self, event_name:str, listener):
        Logger.log_sync(f"Adding {listener.__name__} to event '{event_name}'", 'info')
        with self._lock_sync:
            if event_name == '*':
                self.wild_listeners.discard(listener)
            else:
                l = self.listeners.get(event_name)
                if not l:
                    print(f"{listener.__name__} doesn't exist for event '{event_name}'", 'warn')
                    return
                
                self.listeners[event_name].discard(listener)
                if len(self.listeners[event_name]) < 1:
                    del self.listeners[event_name]

    async def sequence_emit(self, event_name, should_log = True, **event):
        if should_log: await Logger.log_async(f"Event '{event_name}' emitted with parameter(s): {event}; Timestamp: {datetime.now().isoformat()}", 'info')

        async with self._lock_async:
            listeners = list(self.wild_listeners)

        for l in listeners:
            try:
                ev = event.copy()
                ev['event_name'] = event_name
                f = l(**ev)
                if inspect.isawaitable(f):
                    await f
            except Exception as e:
                await Logger.log_async(f"'{event_name}' tried to call function: '{l.__name__}' with {ev}. Error: {repr(e)}. Skipping...", 'warn')

        async with self._lock_async:
            listeners = self.listeners.get(event_name, set())
            
        for listener in list(listeners):
            try:
                f = listener(**event)
                if inspect.isawaitable(f):
                    await f
            except Exception as e:
                await Logger.log_async(f"'{event_name}' tried to call function: '{listener.__name__}' with {event}. Error: {repr(e)}. Skipping...", 'warn')

    async def parallel_emit(self, event_name, should_log= True, **event):
        if should_log: await Logger.log_async(f"Event '{event_name}' emitted with parameter(s): {event}; Timestamp: {datetime.now().isoformat()}", 'info')

        tasks = []

        async with self._lock_async:
            wild = list(self.wild_listeners)
            specific = list(self.listeners.get(event_name, set()))

        if wild:
            wild_payload = {**event, 'event_name': event_name}
            for listener in wild:
                if inspect.iscoroutinefunction(listener):
                    tasks.append(listener(**wild_payload))
                else:
                    tasks.append(asyncio.to_thread(listener, **wild_payload))

        for listener in specific:
            if inspect.iscoroutinefunction(listener):
                tasks.append(listener(**event))
            else:
                tasks.append(asyncio.to_thread(listener, **event))

        if not tasks: return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, asyncio.CancelledError): raise r
            if isinstance(r, Exception):
                await Logger.log_async(f"'{event_name}' error: {repr(r)}. Skipping...", 'warn')