import asyncio
from .utils import log

class Event_Manager:
    def __init__(self):
        self._events = {}
        self.queue = asyncio.Queue()
        self.current_task = None
        self._tasks = set()
        self._lock = asyncio.Lock()

    def on(self, event_name):
        def decorator(func):
            self._events.setdefault(event_name, []).append(func)
            return func
        return decorator

    async def emit_async(self, event_name, *args, **kwargs):
        await self.queue.put((event_name, args, kwargs))

    async def emit_async_block(self, event_name, *args, **kwargs):
        callbacks = self._events.get(event_name)
        if not callbacks:
            return  # deterministic no-op

        tasks = []
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(
                        self._safe_call(callback, event_name, *args, **kwargs)
                    )
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                    tasks.append(task)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                await log(
                    f"Error in event '{event_name}' ({getattr(callback, '__name__', 'unknown')}): {e}",
                    "error"
                )

        if tasks:
            await asyncio.gather(*tasks)

    async def _safe_call(self, callback, event_name, *args, **kwargs):
        try:
            await callback(*args, **kwargs)
        except Exception as e:
            await log(
                f"Async error in event '{event_name}' ({getattr(callback, '__name__', 'unknown')}): {e}",
                "error"
            )

    async def process(self):
        try:
            while True:
                event_name, args, kwargs = await self.queue.get()
                callbacks = self._events.get(event_name, [])

                for callback in callbacks:
                    if asyncio.iscoroutinefunction(callback):
                        task = asyncio.create_task(
                            self._safe_call(callback, event_name, *args, **kwargs)
                        )
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)
                    else:
                        try:
                            callback(*args, **kwargs)
                        except Exception as e:
                            await log(
                                f"Error in event '{event_name}' ({getattr(callback, '__name__', 'unknown')}): {e}",
                                "error"
                            )

                self.queue.task_done()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            await log(f"Event loop crashed: {e}", "critical")

    async def start_event(self):
        async with self._lock:
            if self.current_task is None or self.current_task.done():
                self.current_task = asyncio.create_task(self.process())
                await log("Event manager started.", "info")

    async def stop_event(self):
        async with self._lock:
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()
                try:
                    await self.current_task
                except asyncio.CancelledError:
                    pass
                self.current_task = None

            # cancel outstanding callback tasks
            for task in list(self._tasks):
                task.cancel()
            self._tasks.clear()

            await log("Event manager stopped.", "info")