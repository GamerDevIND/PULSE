import asyncio
import aiofiles
import aiohttp
import json
import signal
from utils import log
from models import Model
from configs import ( 
    CoT_PROMPT, 
    CHAT_PROMPT, 
    ROUTER_PROMPT, 
    DEFAULT_PROMPT, 
    CHAOS_PROMPT, 
    STREAM_DISABLED,
    ERROR_TOKEN,
    VISION_PROMPT
)

class AI:
    def __init__(self, model_config_path="main/Models_config.json", context_path="main/saves/context.json"):
        self.model_config_path = model_config_path
        self.context_path = context_path
        self.models: dict[str, Model] = {}
        self.system_prompts = {
            "chat": CHAT_PROMPT,
            "router": ROUTER_PROMPT,
            "cot": CoT_PROMPT,
            'vision':VISION_PROMPT
        }
        self.default_model = 'chat'
        self.load_models()
        self.running_tasks = set()

    def load_models(self):
        try:
            with open(self.model_config_path, 'r', encoding="utf-8") as f:
                models_data = json.load(f)
                for model_data in models_data:
                    role = model_data.get('role')
                    if role:
                        model_data["system_prompt"] = self.system_prompts.get(role, DEFAULT_PROMPT)
                        self.models[role] = Model(**model_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"🟥 Error loading models: {e}")
            exit(1)

    async def init(self, platform: str):
        self.context = await self.load_context()
        self.platform = platform

        await log("Warming up all models...", "info", append=False)

        for model in self.models.values():
            await model.warm_up()

        check_task = asyncio.create_task(self.check_models())
        self.running_tasks.add(check_task)
        check_task.add_done_callback(self.running_tasks.discard)

    async def load_context(self):
        try:
            async with aiofiles.open(self.context_path) as file:
                content = await file.read()
                return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"conversations": []}

    async def save_context(self):
        try:
            async with aiofiles.open(self.context_path, "w") as file:
                await file.write(json.dumps(self.context, indent=2))
        except IOError as e:
            await log(f"🟥 Error saving context: {e}", "error")

    async def _ping_model_tag(self, url):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{url}/api/tags") as res:
                    return res.status == 200
        except aiohttp.ClientError:
                return False

    async def check_models(self, interval=10):
        while True:
            for _, model in self.models.items():
                is_terminated = model.process and model.process.returncode is not None

                is_unresponsive = not await self._ping_model_tag(model.host)

                if is_terminated or is_unresponsive:
                    await log(f"⚠️ {model.name} crashed/unresponsive (Terminated: {is_terminated}, Unresponsive: {is_unresponsive}). Restarting...", "warn")
                    await model.warm_up() 
            await asyncio.sleep(interval)


    async def route_query(self, query: str, manual: bool):
        if not manual:
            router_model = self.models.get("router")
            if not router_model:
                await log("Router model not configured. Using default.", "error")
                return query, self.default_model

            router_resp_parts = []
            async for part in router_model.generate(query, self.context['conversations'], stream=False):
                if part == ERROR_TOKEN:
                    await log("Router API call failed. Using default.", "error")
                    return query, self.default_model
                router_resp_parts.append(part)

            router_resp_raw = "".join(router_resp_parts).strip()
            selected_role = None

            try:
                router_resp = json.loads(router_resp_raw)
                if isinstance(router_resp, dict):
                    selected_role = router_resp.get("model", router_resp.get("role", "")).strip().lower()
            except json.JSONDecodeError:
                selected_role = router_resp_raw.lower()

            if selected_role in self.models:
                await log(f"Router selected model '{selected_role}'.", "info")
                return query, selected_role
            else:
                await log(f"Router selected unknown role or failed to parse ('{selected_role}'). Using default '{self.default_model}'.", "warn")
                return query, self.default_model
        else:
            if query.startswith("!chat"):
                query = query.removeprefix("!chat").strip()
                self.models['chat'].system = CHAT_PROMPT
                return query, "chat"
            elif query.startswith("!cot"):
                query = query.removeprefix("!cot").strip()
                return query, "cot"
            elif query.startswith("!chaos"):
                query = query.removeprefix("!chaos").strip()
                self.models['chat'].system = CHAOS_PROMPT
                return query, "chat"
            elif query.startswith("!vision"):
                query = query.removeprefix("!vision").strip()
                return query, "vision"
            else:
                await log(f"No or incorrect prefix, using default '{self.default_model}'", "info")
                return query, self.default_model

    async def generate(self, query, stream: None | bool = None, manual_routing = False, seperate_thinking = False, image_path= None):
        query, model_name = await self.route_query(query, manual_routing)

        response = "" 
        save = True
        if stream is None:
            stream = self.platform not in STREAM_DISABLED

        model = self.models[model_name]

        async for res in model.generate(query, self.context['conversations'], stream, image_path=image_path):
            if res == ERROR_TOKEN:
                save = False
                break

            response += res
            if stream:
                yield res

        if not stream and save:
            if model.has_CoT and seperate_thinking:
                if response.startswith("<think>") and "</think>" in response:
                    thinking, response_part = response.split("</think>", 1)
                    thinking = thinking.removeprefix("<think>").strip()
                    response_part = response_part.strip()

                    yield (thinking, response_part) 
                else:
                    yield response
            else:
                yield response

        if save:
            self.context["conversations"].extend([{'role': 'user', 'content': query}, {'role': 'assistant', 'content': response}])
            await self.save_context()

    async def shut_down(self):
        await log("Shutting Down all services...", "info")
        for task in self.running_tasks:
            task.cancel()
        shutdown_tasks = [model.shutdown() for model in self.models.values()]
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        await self.save_context()
        print("Done.")

async def main():
    ai = AI("main/Models_config_test.json")
    await ai.init("cli")

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler():
        print("\nReceived shutdown signal. Initiating graceful shutdown...")
        asyncio.create_task(ai.shut_down())
        shutdown_event.set()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    if hasattr(signal, "SIGHUP"):
        loop.add_signal_handler(signal.SIGHUP, signal_handler)

    while not shutdown_event.is_set():
        try:
            req = await loop.run_in_executor(None, input, ">>> ")
        except EOFError:
            req = "/bye"
        except (KeyboardInterrupt, signal.default_int_handler):
            req = "/bye"

        if shutdown_event.is_set():
            break

        if req.strip() == "/bye":
            await ai.shut_down()
            break ; exit()

        image_path = None
        if req.startswith("!vision"):
            req = req.removeprefix("!vision").strip()
            parts = req.split("|", 1)
            if len(parts) == 2:
                image_path, req = parts
                image_path = image_path.strip()
                req = req.strip()
            else:
                image_path = None
                req = parts[0].strip()
                await log("No image path provided. Continuing with text only.", "warn")

        if not any(m.has_vision for m in ai.models.values()):
            await log("No vision models available. Skipping image input.", "warn")
            image_path = None

        try:
            async for part in ai.generate(req, manual_routing=True, image_path=image_path):
                print(part, end="", flush=True)
            print()
            await log("Generation Completed", "success")
        except Exception as e:
            await log(f"Main loop error: {e}", "error")
            break

if __name__ == "__main__":
    asyncio.run(main())