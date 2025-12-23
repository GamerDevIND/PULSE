import asyncio
import signal
import sys
import os
import colorama

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from main.AI import AI
from main.utils import log

colorama.init()

async def main():
    ai = AI("main/Models_config_test.json")
    await ai.init("cli")

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    @ai.event_manager.on("generation_chunk")
    async def handle_chunk(content_chunk, thinking_chunk, **kwargs):
        if thinking_chunk:
            print(f"{colorama.Fore.LIGHTBLACK_EX}{thinking_chunk}{colorama.Style.RESET_ALL}", end="", flush=True)
        if content_chunk:
            print(content_chunk, end="", flush=True)

    @ai.event_manager.on("tool_executed")
    async def handle_tool(tool_name, result, **kwargs):
        pass

    @ai.event_manager.on("save_context_completed")
    async def handle_save(**kwargs):
        pass

    async def shutdown():
        if not shutdown_event.is_set():
            shutdown_event.set()
            print("\nShutting down AI services...")
            await ai.shut_down()

    def signal_handler():
        asyncio.create_task(shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: asyncio.create_task(shutdown()))

    while not shutdown_event.is_set():
        try:
            req = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            req = "/bye"

        if req.strip() == "/bye":
            await shutdown()
            break

        image_path = None
        if req.startswith("!vision"):
            req = req.removeprefix("!vision").strip()
            parts = req.split("|", 1)
            if len(parts) == 2:
                image_path, req = map(str.strip, parts)
            else:
                await log("No image path. Continuing with text only.", "warn")

        try:
            async for (thinking, response) in ai.generate(req, manual_routing=False, image_path=image_path):
                pass # do whatever you want with the chunks 
            
            print()
            
        except Exception as e:
            await log(f"Main loop error: {e}", "error")
            if not shutdown_event.is_set():
                await shutdown()
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
