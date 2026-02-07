import asyncio
import signal
import sys
import os
import colorama
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from main.AI import AI
from main.utils import log

colorama.init()

async def main():
    ai = AI("main/Models_config_test.json")
    await ai.init("cli")

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

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
                if thinking:
                    print(f"{colorama.Fore.LIGHTBLACK_EX}{thinking}{colorama.Style.RESET_ALL}", end="", flush=True)
                if response:
                    print(response, end="", flush=True)
            
            print()
         
        except KeyboardInterrupt:
            await ai.cancel_generation()
            print(f"{colorama.Fore.RED}Generation cancelled{colorama.Style.RESET_ALL}")
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
