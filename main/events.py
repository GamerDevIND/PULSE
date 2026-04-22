from .utils import log
import inspect
import asyncio
import threading

class EventBus:
    GENERATION_CHUNK = "generation chunk"
    GENERATION_CANCELLED = "generation cancelled"
    TOOL_EXECUTING = "tool executing"
    TOOLS_EXECUTED = "tools executed"
    GENERATION_STARTED = "generation started"
    GENERATION_STOPPED = "generation stopped"
    GARBAGE_COLLECTOR = 'garbage collector'
    GARBAGE_COLLECTED  = 'garbage collected'
    SURMMARISING = "summarising"
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

    def  __init__(self) -> None:
        self.listeners:dict[str, set] = {}
        self._lock = threading.RLock()

    def add_listener(self, event_name:str, listener):
        print(f"Adding {listener.__name__} to event '{event_name}'")
        with self._lock:
            if not self.listeners.get(event_name):
                self.listeners[event_name] = {listener}
            else:
                self.listeners[event_name].add(listener)

    def remove_listener(self, event_name:str, listener):
        print(f"Removing {listener.__name__} from event '{event_name}'")
        with self._lock:
            l = self.listeners.get(event_name)
            if not l:
                print(f"{listener.__name__} doesn't exist for event '{event_name}'", 'warn')
                return

            self.listeners[event_name].remove(listener)
            if len(self.listeners[event_name]) < 1:
                del self.listeners[event_name]

    async def sequence_emit(self, event_name,should_log = True, **event):
        if should_log: await log(f"Event '{event_name}' emitted with parameter(s): {event}", 'info')
        with self._lock:
            listeners = self.listeners.get(event_name, set())
        for listener in list(listeners):
            try:
                f = listener(**event)
                if inspect.isawaitable(f):
                    await f
            except Exception as e:
                await log(f"'{event_name}' tried to call function: '{listener.__name__}' with {event}. Error: {repr(e)}. Skipping...", 'warn')

    async def parallel_emit(self, event_name,should_log = True, **event):
        if should_log: await log(f"Event '{event_name}' emitted with parameter(s): {event}", 'info')
        with self._lock:
            listeners = self.listeners.get(event_name, set())
        l = []
        for listener in list(listeners):
            if inspect.iscoroutinefunction(listener):
                l.append(listener(**event))
            else:
                t = asyncio.to_thread(listener, **event)
                l.append(t)

        results = await asyncio.gather(*l, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                await log(f"'{event_name}' error: {repr(r)}. Skipping...", 'warn')