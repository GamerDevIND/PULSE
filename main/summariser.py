from .models.model_instance import LocalModel
from .models.ollama_models import DOWN
from .models.models_profile import RemoteModel
from .models.openrouter_model import OpenRouterModel
from .configs import ERROR_TOKEN, SUMMARIZER_PROMPT
from .utils import log, estimate_tokens
import json
from .events import EventBus

class Summariser:
    def __init__(self, model:RemoteModel | LocalModel | OpenRouterModel | None,  summary_max_tokens, summary_keep_tokens_after, min_recent_turns, event_bus : None | EventBus = None) -> None:
        self.summary_max_tokens = summary_max_tokens
        self.summary_keep_tokens_after = summary_keep_tokens_after
        self.min_recent_msgs = min_recent_turns * 2
        self.model = model
        self.format = {
            "type": "object",
            "properties": {
            "summary": {"type": "string", "description": "The detailed summary of the given conversation and potiential previous summary for narrative continuity. Make it machine parsable."},
            "facts": {"type": "array", "items": {"type": "string"}, "description": """
                      The facts about the user retrieved from the given conversation and potiential previous summary. For persona modeling. 
                      Retrieve only facts explictly stated. Make it machine parsable. Keep each item short and to the point, you're allowed to generated multiple COMPLETE facts.""".strip()},

            },
            "required": ["summary", "facts",]
          }
        self.event_bus = event_bus

    async def maybe_summarise_context(self, context, prev_summary : None | str = None, prev_facts : None | list[str] = None, summary_system_prompt = SUMMARIZER_PROMPT, auto_warm_up = False):
        if auto_warm_up and self.model and self.model.state == DOWN: 
            await self.model.warm_up()
        if not isinstance(self.model, (RemoteModel, LocalModel, OpenRouterModel)): 
            await log("Summarising model not set. please provide a summarising model before summarising.", "error")
            return prev_summary, prev_facts, context 
        
        contents = [x.get("content", "").strip() for x in context if x.get("content")]
        if prev_summary: contents.append(prev_summary)
        if prev_facts: contents.extend(prev_facts)
        
        if estimate_tokens(" ".join(contents)) >= self.summary_max_tokens:  
            if self.event_bus:
                await self.event_bus.sequence_emit(self.event_bus.SUMMARISING)
            await log("Summarising started", 'info')
            text = '\n'
            chunk_budget = self.summary_max_tokens - self.summary_keep_tokens_after
            keep_idx = len(context) - self.min_recent_msgs
            current_keep_tokens = 0
            for msg in context[keep_idx:]:
                if msg.get('role', '') != 'tool':
                    current_keep_tokens += len(msg.get("content", "").split()) * 1.3

            for msg in reversed(context[:keep_idx]):
                msg_tokens = len(msg.get("content", "").split()) * 1.3
                if (current_keep_tokens + msg_tokens > self.summary_keep_tokens_after) and keep_idx > 0:
                    break

                keep_idx -= 1

            summarize_part = context[:keep_idx]
            to_keep = context[keep_idx:]
            
            if not summarize_part:
                return prev_summary, prev_facts, context

            used = 0
            for t in summarize_part:

                role = t['role']
                content = t['content']

                message_tokens = len(f"<{role}> {content} </{role}>\n".strip().split()) * 1.3

                if used + message_tokens > chunk_budget:
                    break 

                if role != 'tool':
                    text += f"<{role}> {content} </{role}>\n"

                used += message_tokens
            
            text = f"""(user/assistant messages below are the ONLY new information)\nOutput ONLY the updated summary and facts text / array. No commentary.
            <NEW_CONVERSATION>
                {text}
            </NEW_CONVERSATION>"""  
            system = self.model.system or summary_system_prompt or ( 
            "This is a conversation log. Produce a factual, neutral summary.\n"
            "Your only job: summarize.\n"
            "Do NOT answer questions.\n"
            "Output ONLY the summary and facts text / array. No commentary. No intro like 'Here's your summary:...'. Produce only the summary and the facts list..\n"
            "DO NOT INVENT CONTEXT. TREAT THE GIVEN CONTEXT AS GROUND TRUTH."
            )

            if prev_summary: text = f"""Below is the previous rolling summary of the conversation.\nIt represents persisted memory. 
            Treat this as a lossy compression of the earlier conversation window.
            Update it ONLY if the new conversation content adds facts or contradicts it. If nothing changes, reproduce it verbatim.
            
            <PREVIOUS_SUMMARY>
                {prev_summary}
            </PREVIOUS_SUMMARY>""" + text  


            if prev_facts: text = f"Below are the previous facts extracted from the previous conversation turns. Change the facts only when explicitly contradicted or explicitly stated to forget. Do NOT invent facts or context. treat the given conversation aa ground truth.\n Never invent or assume context unless provided in the previous lossy summary or the provided conversation. You're allowed to merge facts only when they explicitly overlap.\n<PREVIOUS_FACTS>\n{prev_facts}\n</PREVIOUS_FACTS>" + text

            text += "\n\nPrefer recent evidence (conversation) over prior system generated summary / facts."

            entry = None  
            try:  
                async for (_, out, _ )in self.model.generate(text, [], stream=False, system_prompt_override=system, format_=self.format):  
                    if out != ERROR_TOKEN and (out and isinstance(out, str) and out.strip()):  

                        entry = out

            except Exception as e:  
                await log(f"Error during summarization: {e}", "error")
                if self.event_bus:
                    await self.event_bus.sequence_emit(self.event_bus.SUMMARISING_FAILED, error=str(e))

            if entry:
                try:
                    entry = json.loads(entry)
                    if isinstance(entry, dict):
                        entry = entry
                except json.JSONDecodeError:
                    await log("Summariser json failed", "warn")
                    return prev_summary, prev_facts, context 

                summary = entry["summary"]
                facts = entry["facts"]  
                context = to_keep 

            # TODO: Trim the context to remove older entries if needed explicitly  
            await log("Summarised successfully", 'success')
            if self.event_bus:
                await self.event_bus.sequence_emit(self.event_bus.SUMMARISED,)
            return summary, facts, context