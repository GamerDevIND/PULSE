from .utils import log
from .configs import ERROR_TOKEN
import json

class Router:
    def __init__(self, model, fallback_role, *available_roles, manual_prefix="!"):
          self.model = model
          self.fallback_role = fallback_role
          self.available_roles = set(available_roles)
          self.manual_prefix = manual_prefix

    async def route_query(self, query: str, context, manual: bool | None):
        if not manual:
            if not self.model:
                await log("Router model not configured. Using default.", "error")
                return query, self.fallback_role

            router_resp_parts = []

            async for _, part, _ in self.model.generate(query, context, stream=False):
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
