from .utils import log
import inspect
import asyncio

class EventBus:
    def  __init__(self) -> None:
        self.listeners:dict[str, set] = {}

    def add_listener(self, event_name:str, listener):
        print(f"Adding {listener.__name__} to event '{event_name}'")
        if not self.listeners.get(event_name):
            self.listeners[event_name] = {listener}
        else:
            self.listeners[event_name].add(listener)

    def remove_listener(self, event_name:str, listener):
        print(f"Removing {listener.__name__} from event '{event_name}'")
        l = self.listeners.get(event_name)
        if not l:
            print(f"{listener.__name__} doesn't exist for event '{event_name}'", 'warn')
            return
        
        self.listeners[event_name].remove(listener)
        if len(self.listeners[event_name]) < 1:
            del self.listeners[event_name]

    async def sequence_emit(self, event_name,should_log = True, **event):
        if should_log: await log(f"Event '{event_name}' emitted with parameter(s): {event}", 'info')
        listeners = self.listeners.get(event_name, set())
        for listener in list(listeners):
            try:
                f = listener(**event)
                if inspect.isawaitable(f):
                    await f
            except Exception as e:
                await log(f"'{event_name}' tried to call function: '{listener.__name__}' with {event}. Error: {repr(e)}. Skipping...", 'warn')

    async def parrallel_emit(self, event_name,should_log = True, **event):
        if should_log: await log(f"Event '{event_name}' emitted with parameter(s): {event}", 'info')
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