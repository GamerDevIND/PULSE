from .models.model_instance import LocalModel
from .models.ollama_models import DOWN
from .models.models_profile import RemoteModel
from .models.openrouter_model import OpenRouterModel
from .configs import ERROR_TOKEN, SUMMARIZER_PROMPT
from .utils import Logger, estimate_tokens
import traceback
import re
import json
from .events import EventBus

TURNS_TO_MSG_MULTIPLIER = 2.5

class Summariser:
    def __init__(self, model:RemoteModel | LocalModel | OpenRouterModel | None,  summary_max_tokens, summary_keep_tokens_after, min_recent_turns, trim_turn_num, event_bus : None | EventBus = None) -> None:
        self.summary_max_tokens = summary_max_tokens
        self.summary_keep_tokens_after = summary_keep_tokens_after
        self.min_recent_msgs = round(min_recent_turns * TURNS_TO_MSG_MULTIPLIER)
        self.model = model
        self.trim_turn_num = round(trim_turn_num * TURNS_TO_MSG_MULTIPLIER)
        self.format = {
            "type": "object",
            "properties": {
            "summary": {"type": "string", "description": "The detailed summary of the given conversation and potiential previous summary for narrative continuity. \n"
            "Make it machine parsable. Preserve the specific / explicit details and references "},
            "facts": {"type": "array", "items": {"type": "string"}, "description": """
                      The facts about the user retrieved from the given conversation and potiential previous summary. For persona modeling. Preserve specific details if needed.
                      Preserve specific references and explicitly stated facts if proven needed for future conversation flow. Never self narrate. 
                      Retrieve only facts explictly stated. Make it machine parsable. NEVER reflect your system in these responses, follow the system, don't narrate.
                      Keep each item short and to the point, you're allowed to generated multiple COMPLETE facts.""".strip()},
            "key_decisions": {"type": "array", "items": {"type": "string"}, "description": '''
                      The key decisions taken during the conversation discussion, which might be relevant for later references, or be relevant for shaping the conversation flow.
                      Preserve long term decisions over short term choices. Make sure to extract the key decisions during the conversation flow. Keep finalized choices, architectural directions, or preferences that override defaults. 
                      Focus on 'the what' and 'the why'. (e.g. [do not treat this as the part of the conversation]: 'User opted for SQLite to maintain local-first portability'). Avoid transient choices.
                      If a new decision EXPLICTLY contradicts or updates a previous one, replace the old entry to maintain a single source of truth. Prioritize technical constraints and user-defined 'Hard Rules'.
                      Strictly deduplicate. If the current decision is a refinement of a previous one, merge them into a single concise bullet point. Avoid decision bloat (keeping copies of the same decisions),
                      Internally distinguish between 'Explicit' (User said: 'Use X') and 'Inferred' (User followed a path implying Y). Prioritize Explicit decisions. IF no decison is taken, return an empty array. 
                    '''.strip()},
            "open_threads": {"type": "array", "items": {"type": "string"}, "description": '''
                      List unresolved problems, pending tasks, or 'to-be-decided' items.  Focus on roadblocks or questions the user hasn't answered yet. (e.g., 'Need to determine the encryption method for local DB'). 
                      Remove an item once is solved (expictly stated by user, or tool). Preserve ongoing open and unresolved discussion, issues or threads. 
                      Priotise long term threads over short term one time. List ONLY currently active unresolved problems. When a thread is resolved in the new conversation, it MUST be removed. 
                      Do not carry over completed tasks. IF no threads remain, return an empty array.
                            '''.strip()}

            },
            "required": ["summary", "facts", "key_decisions", 'open_threads', ]
        }
        
        self.event_bus = event_bus

    def build_convo_text(self, text, convo:list[dict], chunk_budget):
        text = text

        used = 0
        for t in convo:

            role = t['role']
            content = t['content']

            message_tokens = estimate_tokens(f"<{role}> {content} </{role}>\n".strip())

            if used + message_tokens > chunk_budget:
                break 

            if role != 'tool':
                text += f"<{role}> {content} </{role}>\n"
            elif role == 'tool':
                try:
                    tc = json.dumps(content)
                except json.JSONDecodeError:
                    tc = ""
                    
                if tc:
                    name = t['tool_name']
                    text += f"<{role} (Name: {name})> {content} </{role}>\n"

            used += message_tokens

        return text

    async def maybe_summarise_context(self, context, prev_meta = {}, summary_system_prompt = SUMMARIZER_PROMPT, auto_warm_up = False):
        if auto_warm_up and self.model and self.model.state == DOWN: 
            await self.model.warm_up()
        
        if prev_meta is None:
            prev_meta = {}

        prev_facts = prev_meta.get('facts', prev_meta.get("prev_facts"))
        prev_summary = prev_meta.get('summary', prev_meta.get("prev_summary"))
        prev_open_threads = prev_meta.get('open_threads', prev_meta.get("prev_open_threads"))
        prev_key_decisions = prev_meta.get('key_decisions', prev_meta.get("prev_key_decisions"))

        if not isinstance(self.model, (RemoteModel, LocalModel, OpenRouterModel)): 
            await Logger.log_async("Summarising model not set. please provide a summarising model before summarising.", "error")
            meta = {
                'facts': prev_facts,
                'summary': prev_summary,
                'open_threads': prev_open_threads,
                'key_decisions': prev_key_decisions
            }

            return meta, context
        
        contents = [x.get("content", "").strip() for x in context if x.get("content")]
        if prev_summary: contents.append(prev_summary)
        if prev_facts: contents.extend(prev_facts)
        if prev_open_threads: contents.extend(prev_open_threads)
        if prev_key_decisions: contents.extend(prev_key_decisions)

        est = estimate_tokens(" ".join(contents))

        if est >= self.summary_max_tokens:  
            text = '\n'
            chunk_budget = self.summary_max_tokens - self.summary_keep_tokens_after
            keep_idx = len(context) - self.min_recent_msgs
            keep_idx = max(0, keep_idx)
            current_keep_tokens = 0
            for msg in context[keep_idx:]:
                if msg.get('role', '') != 'tool':
                    current_keep_tokens += estimate_tokens(msg.get("content", "").strip())

            for msg in reversed(context[:keep_idx]):
                msg_tokens = estimate_tokens(msg.get("content", "").strip())
                if (current_keep_tokens + msg_tokens > self.summary_keep_tokens_after) and keep_idx > 0:
                    break

                keep_idx -= 1

            summarize_part = context[:keep_idx]
            to_keep = context[keep_idx:]
            
            if not summarize_part:
                meta = {
                'facts': prev_facts,
                'summary': prev_summary,
                'open_threads': prev_open_threads,
                'key_decisions': prev_key_decisions
                }

                return meta, context

            text = self.build_convo_text(text, summarize_part, chunk_budget)

            if self.event_bus:
                await self.event_bus.sequence_emit(self.event_bus.SUMMARISING)
                
            await Logger.log_async("Summarising started", 'info')

            est = round(estimate_tokens(text))

            await Logger.log_async(f"Estimated tokens of the convo: {est}",'info')

            if est >= self.summary_max_tokens * 3:
                await Logger.log_async(f"Conversation too big, trimming the oldest {self.trim_turn_num} messages", 'warn')
                if self.event_bus: await self.event_bus.parallel_emit(self.event_bus.WARN, msg = f"Conversation too big, trimming the oldest {self.trim_turn_num} messages")

                summarize_part = summarize_part[self.trim_turn_num:]

                if not summarize_part:
                    meta = {
                    'facts': prev_facts,
                    'summary': prev_summary,
                    'open_threads': prev_open_threads,
                    'key_decisions': prev_key_decisions
                    }

                    return meta, context

                text = self.build_convo_text(text, summarize_part, chunk_budget)

                to_keep = to_keep[self.trim_turn_num:]

            text = f"""(user/assistant messages below are the ONLY new information)\nOutput ONLY the updated summary and facts text / array. No commentary.
            <NEW_CONVERSATION>
                {text}
            </NEW_CONVERSATION>"""  
            system = self.model.system or summary_system_prompt or ( 
            "This is a conversation Logger.log_async. Produce a factual, neutral summary.\n"
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


            if prev_facts: text = f"Below are the previous facts extracted from the previous conversation turns. Change the facts only when explicitly contradicted or explicitly stated to forget. Do NOT invent facts or context. treat the given conversation aa ground truth.\n Never invent or assume context unless provided in the previous lossy summary or the provided conversation. You're allowed to merge facts only when they explicitly overlap.\n<PREVIOUS_FACTS>\n{"\n".join(prev_facts)}\n</PREVIOUS_FACTS>".strip() + text
            
            if prev_open_threads:
                text = f"Below are the previous open threads extracted from the previous conversation turns. Change the open threads only when explicitly contradicted or explicitly stated to forget. Do NOT invent threads or context. treat the given conversation aa ground truth.\n Never invent or assume context unless provided in the previous lossy summary or the provided conversation. You're allowed to merge threads only when they explicitly overlap\n<THREADS>\n{"\n".join(prev_open_threads)}\n<THREADS>".strip() + text

            if prev_key_decisions:
                text = f"Below are the previous key decisions extracted from the previous conversation turns. Change the key decisions only when explicitly contradicted or explicitly stated to forget. Do NOT invent key decisions or context. treat the given conversation aa ground truth.\n Never invent or assume context unless provided in the previous lossy summary or the provided conversation. You're allowed to merge decisions only when they explicitly overlap\n<DECISIONS>\n{"\n".join(prev_key_decisions)}\n<DECISIONS>".strip() + text


            text += ("\n\nPrefer recent evidence (conversation) over prior system generated summary / facts / open threads / key decisions. \n"
            "If nothing changes, preserve it verbatim, don't say 'Orginal summary / facts / open threads / key decisions unchanged.' produce it as is (1:1) verbatim.")

            entry = None  
            try:
                async for (_, out, _ )in self.model.generate(text, [], stream=False, system_prompt_override=system, format_=self.format):  
                    if out != ERROR_TOKEN and (out and isinstance(out, str) and out.strip()):
                        entry = out

            except Exception as e:  
                await Logger.log_async(f"Error during summarization: {e}; {traceback.format_exc()}", "error")
                if self.event_bus:
                    await self.event_bus.sequence_emit(self.event_bus.SUMMARISING_FAILED, error=str(e))

            summary = prev_summary
            facts = prev_facts
            key_decisions = prev_key_decisions or []
            open_threads = prev_open_threads or []
            if entry:
                entry = re.sub(r"```json\n?|```", "", entry).strip().replace("\n\n\n\n", "\n")
                entry_obj = None

                def extract_json_payload(raw_text):
                    start = raw_text.find('{')
                    if start == -1:
                        return None
                    stack = []
                    for idx, ch in enumerate(raw_text[start:], start):
                        if ch == '{':
                            stack.append('{')
                        elif ch == '}':
                            if stack:
                                stack.pop()
                                if not stack:
                                    return raw_text[start:idx + 1]
                    return None

                def safe_json_load(raw_text):
                    try:
                        return json.loads(raw_text)
                    except json.JSONDecodeError:
                        payload = extract_json_payload(raw_text)
                        if not payload:
                            return None
                        try:
                            return json.loads(payload)
                        except json.JSONDecodeError:
                            normalized = re.sub(r",\s*([\]}])", r"\1", payload)
                            try:
                                return json.loads(normalized)
                            except json.JSONDecodeError:
                                return None

                entry_obj = safe_json_load(entry)
                if not isinstance(entry_obj, dict):
                    await Logger.log_async("Summariser json failed", "warn")
                    await Logger.log_async(f"Raw output: {entry}", "warn")
                    if self.event_bus:
                        await self.event_bus.sequence_emit(self.event_bus.SUMMARISING_FAILED, error="Summariser json failed")
                    meta = {
                        'facts': prev_facts,
                        'summary': prev_summary,
                        'open_threads': prev_open_threads,
                        'key_decisions': prev_key_decisions
                    }

                    t = " ".join(list(map(lambda x: x.get('content', ''), context)))

                    if estimate_tokens(t) >= self.summary_max_tokens * 3:
                        context = context[self.trim_turn_num:]
                    return meta, context

                summary = str(entry_obj.get("summary", prev_summary or ""))
                summary_words_len = len(summary.split())
                if summary_words_len < 15:
                    await Logger.log_async(f"Summary too short: {summary}", "warn")
                    if self.event_bus:
                        await self.event_bus.sequence_emit(self.event_bus.SUMMARISING_FAILED, msg=f"Summary too short: {summary}")
                    meta = {
                        'facts': prev_facts,
                        'summary': prev_summary,
                        'open_threads': prev_open_threads,
                        'key_decisions': prev_key_decisions

                    }

                    if estimate_tokens(" ".join([m.get('content', '') for m in context])) >= self.summary_max_tokens * 3:
                        context = context[self.trim_turn_num:]

                    return meta, context

                facts = entry_obj.get("facts", []) or []
                facts = list(set(facts))
                key_decisions = list(set(entry_obj.get('key_decisions', []) or []))
                open_threads = list(set(entry_obj.get('open_threads', []) or []))

                context = to_keep

            await Logger.log_async("Summarisation successfully ran", 'success')
            if self.event_bus:
                await self.event_bus.sequence_emit(self.event_bus.SUMMARISED,)
            
            meta = {
                'facts': facts if facts and len(facts) >= 3 else ((prev_facts) + (facts or []) if prev_facts else []),
                'summary': summary,
                'open_threads': open_threads,
                'key_decisions': key_decisions
            }

            return meta, context