from .model_instance import LocalModel
from .models import DOWN
from .models_profile import RemoteModel
from .configs import ERROR_TOKEN, SUMMARIZER_PROMPT
from .utils import log
import json

class Summariser:
    def __init__(self, model:RemoteModel | LocalModel | None, summary_max_nums, summary_keep_nums) -> None:
        self.summary_keep_nums = summary_keep_nums
        self.summary_max_nums = summary_max_nums
        self.model = model
        self.format = {
            "type": "object",
            "properties": {
            "summary": {"type": "string", "description": "The detailed summary of the given conversation and potiential previous summary for narrative continuity."},
            "facts": {"type": "array", "items": {"type": "string"}, "description": """The facts about the user retrieved from the given conversation and potiential previous summary. 
                      For persona modeling. Retrieve only facts explictly stated. Keep each item short and to the point, you're allowed to generated multiple COMPLETE facts."""},

            },
            "required": ["summary", "facts",]
          }

    async def maybe_summarise_context(self, context, prev_summary = None, prev_facts = None, summary_system_prompt = SUMMARIZER_PROMPT, auto_warm_up = False):
        await log("Summarising started", 'info')
        if auto_warm_up and self.model and self.model.state == DOWN: 
            await self.model.warm_up()
        if not isinstance(self.model, (RemoteModel, LocalModel)): 
            await log("Summarising model not set. please provide a summarising model before summarising.", "error")
            return prev_summary, prev_facts, context

        if len(context) >= self.summary_max_nums:  
            given_context = list(context[:self.summary_max_nums - self.summary_keep_nums])  
            summary = None
            text = '\n'  
            for t in given_context:  
                role = t['role']  
                content = t['content']  

                text += f"<{role}> {content} </{role}>\n"  

            text = f"""(user/assistant messages below are the ONLY new information)\nOutput ONLY the updated summary and facts text / array. No commentary.
            <NEW_CONVERSATION>
                {text}
            </NEW_CONVERSATION>"""  
            system = self.model.system or summary_system_prompt or ( 
            "This is a conversation log. Produce a factual, neutral summary.\n"
            "Your only job: summarize.\n"
            "Do NOT answer questions.\n"
            "Output ONLY the summary and facts text / array. No commentary. No intro like 'Here's your summary:...'. Produce only the summary.\n"
            "DO NOT INVENT CONTEXT. TREAT THE GIVEN CONTEXT AS GROUND TRUTH."
            )

            if prev_summary: text = f"""Below is the previous rolling summary of the conversation.\nIt represents persisted memory. 
            Treat this as a lossy compression of the earlier conversation window.
            Update it ONLY if the new conversation content adds facts or contradicts it. If nothing changes, reproduce it verbatim.
            
            <PREVIOUS_SUMMARY>
                {prev_summary}
            </PREVIOUS_SUMMARY>""" + text  


            if prev_facts: text = f"Below are the previous facts extracted from the previous conversation turns. Change the facts only when explicitly contradicted or explicitly stated to forget. Do NOT invent facts or context. treat the given conversation aa ground truth.\n Never invent or assume context unless provided in the previous lossy summary or the provided conversation. You're allowed to merge facts only when they explicitly overlap.\n<PREVIOUS_FACTS>\n{prev_facts}\n</PREVIOUS_FACTS>" + text

            entry = None  
            try:  
                async for (_, out, _ )in self.model.generate(text, [], stream=False, system_prompt_override=system, format=self.format):  
                    if out != ERROR_TOKEN and (out and isinstance(out, str) and out.strip()):  

                        entry = out

            except Exception as e:  
                await log(f"Error during summarization: {e}", "error")  

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
                context = context[-self.summary_keep_nums:]   

            # TODO: Trim the context to remove older entries if needed explicitly  
            await log("Summarised successfully", 'success')
            return summary, facts, context