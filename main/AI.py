import asyncio
import aiofiles
import json
from utils import log
from models import Model
from configs import ( 
    CoT_PROMPT, 
    CHAT_PROMPT, 
    ROUTER_PROMPT, 
    DEFAULT_PROMPT, 
    CHAOS_PROMPT, 
    STREAM_DISABLED,
    ERROR_TOKEN
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
        }
        self.default_model = 'chat'
        self.load_models()

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
            await asyncio.sleep(0.02) 

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

    async def check_models(self, interval=10):
        while True:
            for role, model in self.models.items():
                if model.process and (model.process.poll() is not None):
                    await log(f"⚠️ {model.name} crashed. Restarting...", "warn")
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
        shutdown_tasks = [model.shutdown() for model in self.models.values()]
        await asyncio.gather(*shutdown_tasks)
        await self.save_context()
        print("Done.")

async def main():
    ai = AI("main/Models_config_test.json")
    await ai.init("cli")
    while True:
        req = input(">>> ")
        if req == "/bye":
            await ai.shut_down()
            break

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
                await log("No image path provided. Continuing with text only.", 'warn')

        if not any(m.has_vision for m in ai.models.values()):
            await log("No vision models available. Skipping image input.", 'warn')
            image_path = None

        try:
            async for part in ai.generate(req,manual_routing=True, image_path = image_path):
                print(part, end="", flush=True)
            print()
            await log("Generation Completed", 'success')
        except Exception as e:
            await log(f"Main loop error: {e}", "error")
            break

if __name__ == "__main__":
    asyncio.run(main())
