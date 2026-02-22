from .utils import log
from .configs import ERROR_TOKEN
import json
from .models import DOWN
from .model_instance import LocalModel
from .models_profile import RemoteModel

class Router:
    def __init__(self, model: RemoteModel | LocalModel | None, fallback_role, *available_roles, manual_prefix="!", auto_warmup=False):
          self.model = model
          self.fallback_role = fallback_role
          self.available_roles = list(map(lambda x: str(x).lower(), available_roles))
          self.auto_warmup = auto_warmup
          self.manual_prefix = manual_prefix
          self._update_format()

    def add_to_available_roles(self, *roles):
        roles = list(map(lambda x: str(x).lower(), roles))
        self.available_roles.extend(roles)
        self.available_roles = list(set(self.available_roles)) 
        self._update_format()

    def _update_format(self): 
        self.format = {
            "type": "object",
            "properties": {
            "role": {"type": "string", "enum": self.available_roles, "description": "Selected role or model the query to be routed. remember: cot is for chain of thought/reasoning heavy model."}
            },
            "required": ["role"]
          }

    async def route_query(self, query: str, context, manual: bool | None):
        if not manual:
            if not self.model:
                await log("Router model not configured. Using default.", "error")
                return query, self.fallback_role

            if not self.model.warmed_up or self.model.state == DOWN:
                await log("Router model is not warmed up or down", 'warn')
                if self.auto_warmup:
                    await log("Warming up router model", 'info')
                    await self.model.warm_up()
                else:
                    await log("Please warm up the router model first", 'info')
                    return query, self.fallback_role

            router_resp_parts = []
            if len(self.available_roles) < 1:
                await log("No role provided for router", "error") 
                raise Exception("No role provided for router")

            async for _, part, _ in self.model.generate(query=query, context=context, stream=False, format = self.format):
                if part == ERROR_TOKEN:
                    await log("Router API call failed. Using default.", "error")
                    return query, self.fallback_role
                if part:
                    router_resp_parts.append(part)

            router_resp_raw = "".join(router_resp_parts).strip()
            selected_role = None

            await log(f"Router raw response: {router_resp_raw}", "info")

            try:
                router_resp = json.loads(router_resp_raw)
                if isinstance(router_resp, dict):
                    selected_role = router_resp.get("role", "").strip().lower()
            except json.JSONDecodeError:
                await log("Router json failed", "warn")
                selected_role = router_resp_raw.lower().strip()

            if selected_role in self.available_roles:
                await log(f"Router selected model '{selected_role}'.", "info")
                return query, selected_role
            else:
                await log(f"Router selected unknown role or failed to parse ('{selected_role}'). Using default '{self.fallback_role}'.", "warn")

                return query, self.fallback_role
        else:
            return await self.parse_query(query)

    async def parse_query(self, query): # i might add automatic role parsing later
        if query.startswith(f"{self.manual_prefix}chat"):
            query = query.removeprefix(f"{self.manual_prefix}chat").strip()
            return query, "chat"
        elif query.startswith(f"{self.manual_prefix}cot"):
            query = query.removeprefix(f"{self.manual_prefix}cot").strip()
            return query, "cot"
        elif query.startswith(f"{self.manual_prefix}chaos"):
            query = query.removeprefix(f"{self.manual_prefix}chaos").strip()
            return query, "chaos"
        elif query.startswith(f"{self.manual_prefix}vision"):
            query = query.removeprefix(f"{self.manual_prefix}vision").strip()

            return query, "vision"
        else:
            await log(f"No or incorrect prefix, using default '{self.fallback_role}'", "info")

            return query, self.fallback_role