from .utils import Logger
from .configs import ERROR_TOKEN
import json
from .models.ollama_models import DOWN
from .models.model_instance import LocalModel
from .models.models_profile import RemoteModel
from .models.openrouter_model import OpenRouterModel

class Router:
    def __init__(self, model: RemoteModel | LocalModel | OpenRouterModel | None, fallback_role, *available_roles, manual_prefix="!", auto_warmup=False):
          if not isinstance(model, (RemoteModel, LocalModel, OpenRouterModel)):
              raise TypeError
          
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

    async def route_query(self, query: str | None, context, manual: bool | None, include_chaos = True):
        if not manual:
            if not self.model:
                await Logger.log_async("Router model not configured. Using default.", "error")
                return query, self.fallback_role
            
            if not self.model.warmed_up or self.model.state == DOWN:
                await Logger.log_async("Router model is not warmed up or down", 'warn')
                if self.auto_warmup:
                    await Logger.log_async("Warming up router model", 'info')
                    await self.model.warm_up()
                else:
                    await Logger.log_async("Please warm up the router model first", 'info')
                    return query, self.fallback_role

            router_resp_parts = []
            if len(self.available_roles) < 1:
                await Logger.log_async("No role provided for router", "error") 
                raise Exception("No role provided for router")

            try:
                async for _, part, _ in self.model.generate(query=query, context=context, stream=False, format_ = self.format, tools_override=[]):
                    if part == ERROR_TOKEN:
                        await Logger.log_async("Router API call failed. Falling back to default role.", "error")
                        return query, self.fallback_role
                    if part:
                        router_resp_parts.append(part)
            except Exception as e:
                await Logger.log_async(f"Exception during routing: {e}", "error")
                return query, self.fallback_role

            router_resp_raw = "".join([str(p) for p in router_resp_parts if p is not None]).strip()
            selected_role = None

            await Logger.log_async(f"Router raw response: {router_resp_raw}", "info")

            try:
                router_resp = json.loads(router_resp_raw)
                if isinstance(router_resp, dict):
                    selected_role = router_resp.get("role", "").strip().lower()
            except json.JSONDecodeError:
                await Logger.log_async("Router json failed", "warn")
                selected_role = router_resp_raw.lower().strip()

            if selected_role in self.available_roles:
                await Logger.log_async(f"Router selected model '{selected_role}'.", "info")
                return query, selected_role
            else:
                await Logger.log_async(f"Router selected unknown role or failed to parse ('{selected_role}'). Using default '{self.fallback_role}'.", "warn")
       
                return query, self.fallback_role
        else:
            if query:
                return await self.parse_query(query, include_chaos)
            else:
                await Logger.log_async("No query provided, defaulting...", 'warn')
                return query, self.fallback_role

    async def parse_query(self, query:str, include_chaos = True): # i might add automatic role parsing later

        roles = list(map(lambda x: x.strip() if x else "", self.available_roles))
        if not 'chaos' in roles and include_chaos:
            roles.append('chaos')

        for role in roles:
            prefix = f"{self.manual_prefix}{role} "
            if query and query.strip().startswith(prefix):
                query = query.removeprefix(prefix)
                return query.strip(), role
            
        await Logger.log_async(f"No or incorrect prefix, using default '{self.fallback_role}'", "info")
              
        return query, self.fallback_role
