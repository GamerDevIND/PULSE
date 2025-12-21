import asyncio
import signal
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from main.AI import AI
from main.utils import log

async def main():
    ai = AI("main/Models_config_test.json")
    await ai.init("cli")

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    async def shutdown():
        if not shutdown_event.is_set():
            shutdown_event.set()
            await ai.shut_down()

    def signal_handler():
        print("\nReceived shutdown signal. Initiating graceful shutdown...")
        asyncio.create_task(shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: asyncio.create_task(shutdown()))

    while not shutdown_event.is_set():
        try:
            req = await loop.run_in_executor(None, input, ">>> ")
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
                await log("No image path provided. Continuing with text only.", "warn")

        if image_path and not any(m.has_vision for m in ai.models.values()):
            await log("No vision models available. Skipping image input.", "warn")
            image_path = None

        try:
            async for (_, res) in ai.generate(req, manual_routing=False, image_path=image_path):
                print(res, end="", flush=True)
            print()
            await log("Generation Completed", "success")
        except Exception as e:
            await log(f"Main loop error: {e}", "error")
            break


if __name__ == "__main__":
    asyncio.run(main())
