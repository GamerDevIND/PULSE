import asyncio
from .utils import log

class Event_Manager:
    def __init__(self):
        self._events = {}
        self.queue = asyncio.Queue()
        self.current_task = None

    def on(self, event_name):
        def decorator(func):
            if event_name not in self._events:
                self._events[event_name] = []
            self._events[event_name].append(func)
            return func
        return decorator

    async def emit(self, event_name, *args, **kwargs):
        await self.queue.put((event_name, args, kwargs))

    async def emit_and_wait(self, event_name, *args, **kwargs):
        if event_name in self._events:
            tasks = []
            for callback in self._events[event_name]:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(asyncio.create_task(callback(*args, **kwargs)))
                else:
                    callback(*args, **kwargs)
            
            if tasks:
                await asyncio.gather(*tasks)
        else:
            await self.queue.put((event_name, args, kwargs))

    async def process(self):
        while True:
            event_name = "Unknown"
            try:
                item = await self.queue.get()

                if item is None: break

                event_name, args, kwargs = item
                if event_name in self._events:
                    for callback in self._events[event_name]:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(*args, **kwargs))
                        else:
                            callback(*args, **kwargs)
            except Exception as e:
                await log(f"Error processing event '{event_name}': {str(e)}", "error")

    async def start_event(self):
        if self.current_task is None or self.current_task.done():
            self.current_task = asyncio.create_task(self.process())
            await log("Event manager started.", "info")

    async def stop_event(self):
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
            self.current_task = None
            await log("Event manager stopped.", "info")
