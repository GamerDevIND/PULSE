import asyncio
import signal
import sys
import colorama
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from main.AI import AI
from main.utils import log, estimate_tokens

colorama.init()

async def main():
    ai = AI("main/openrouter_models_configs.json", mode='openrouter')
    await ai.init("cli")

    gen = None

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    async def shutdown():
        if not shutdown_event.is_set():
            shutdown_event.set()
            print("\nShutting down AI services...")
            await ai.shut_down()

    sigint_state = {"count": 0, "last": 0.0}
    SIGINT_WINDOW = 3.0

    @ai.event("initialising")
    def p():
        print("Initialising...")

    @ai.event("initialised")
    def r():
        print("initialised")

    @ai.event("shutting down")
    async def a():
        await shutdown()
    ai.event("shut down")
    def b():
        print("Shut down successful")

    ai.event("cancelling session")
    def cancel(**_):
        print(f"Session: was cancelled")

    @ai.event("retrieving memory")
    def rag(query, **_):
        print(f"Retrieving memory for {query}")

    @ai.event('garbage collector')
    def gc(path, **_):
        print(f"Garbage collector cleaning: {path}")
    
    @ai.event("routing role")
    def route(role, **_):
        print(f"Routing to {role}")

    @ai.event("summarising")
    def sumarise():
        print("Summarising conversation...")

    @ai.event("tool executing")
    def tool(tool_name, **_):
        print(f"Executing tool :{tool_name}")
        
    ai.event("tools executed")
    def tools(**_):
        print("Tools executed")

    async def _on_sigint():
        now = loop.time()
        if sigint_state["count"] == 0 or (now - sigint_state["last"]) > SIGINT_WINDOW:
            sigint_state["count"] = 1
            sigint_state["last"] = now
            if gen:
                try:
                    await gen.terminate()
                except Exception:
                    pass
            print("\nPress Ctrl+C again within 3 seconds to exit.")

            async def _reset():
                await asyncio.sleep(SIGINT_WINDOW)
                sigint_state["count"] = 0

            asyncio.create_task(_reset())
        else:
            await shutdown()

    def _sigint_handler():
        asyncio.create_task(_on_sigint())

    def _sigterm_handler():
        asyncio.create_task(shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            if sig is signal.SIGINT:
                loop.add_signal_handler(sig, _sigint_handler)
            else:
                loop.add_signal_handler(sig, _sigterm_handler)
        except NotImplementedError:
            if sig is signal.SIGINT:
                signal.signal(sig, lambda *_: asyncio.create_task(_on_sigint()))
            else:
                signal.signal(sig, lambda *_: asyncio.create_task(shutdown()))
    
    if not sys.stdin.isatty():
        await log("No TTY detected — running headless; waiting for signals.", "info")
        await shutdown_event.wait()
        if not shutdown_event.is_set():
            await shutdown()
        return

    while not shutdown_event.is_set():
        try:
            req = input(">>> ")
        except EOFError:
            req = "/bye"
        except KeyboardInterrupt:
            print()
            continue

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
            gen = await ai.create_generation(req, manual_routing=False, file_path=image_path)

            if not gen:
                return
            
            async for (thinking, response) in gen.stream():
                if thinking:
                    print(f"{colorama.Fore.LIGHTBLACK_EX}{thinking}{colorama.Style.RESET_ALL}", end="", flush=True)
                if response:
                    print(response, end="", flush=True)

            print()
            words = await ai.context_manager.get_conversation(ai.last_cid)
            words = [c["content"] for c in await words.get_context()]
            words = estimate_tokens(" ".join(words))
            print(f"Estimated total tokens: {round(words)}")
            print()

        except KeyboardInterrupt:
            if gen:
                await gen.terminate()
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