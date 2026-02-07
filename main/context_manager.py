from .utils import log
from .configs import ERROR_TOKEN
import json 
import asyncio
import aiofiles 
from .models import DOWN
from .model_instance import LocalModel
from .models_profile import RemoteModel

class ContextManager:
    def __init__(self, context_path, summary_model: LocalModel | RemoteModel | None, summary_max_nums = 20, summary_keep_nums = 10,):
        self.context_path = context_path
        self.summary_model = summary_model
        self.summary_max_nums = summary_max_nums
        self.summary_keep_nums = summary_keep_nums
         
        self.summary = None
        self.context = []

        self.lock = asyncio.Lock()

        self.queue = asyncio.Queue()
        self.summary_task = None
        
    async def append(self, data:dict | list[dict] | tuple[dict]): 
        if type(data) == dict:
            await self.queue.put(data)
        elif type(data) in [list, tuple]:
            for d in data: await self.queue.put(d)
        else: await log("unknown data type, skipping item.", "warn")
 
    async def add_and_maintain(self, data:dict | list[dict] | tuple[dict]):
        await self.append(data)
             
        await self.flush_queue()
        await self.maybe_summarise_context(None, True)
        await self.save()

    async def flush_queue(self):
        items = []
         
        while not self.queue.empty():
            item = await self.queue.get()
            items.append(item)
        async with self.lock:
            self.context.extend(items)

    async def get_context(self):
        async with self.lock:
            return list(self.context)

    def get_summary(self):
        return str(self.summary) if self.summary else None
 
    async def shut_down(self):        
        await self.flush_queue()
        await self.save()
        await log("Context saved and context manager shut down.", "info")

    async def maybe_summarise_context(self, context=None, save_context= False, summary_system_prompt = None, auto_warm_up = False):
        if context is None:  
            await self.flush_queue()
            async with self.lock: 
                context = list(self.context)  
        if auto_warm_up and self.summary_model and self.summary_model.state == DOWN: await self.summary_model.warm_up()
        if not isinstance(self.summary_model, (RemoteModel, LocalModel)): 
            await log("Summarising model not set. please provide a summarising model before summarising.", "error")
            raise Exception("Summarising model not set. please provide a summarising model before summarising.")
        if len(context) >= self.summary_max_nums:  
            given_context = list(context[:self.summary_max_nums - self.summary_keep_nums])  
            text = '\n'  
            for t in given_context:  
                role = t['role']  
                content = t['content']  

                if role != 'tool':  
                    text += f"{role} : {content}\n"  

            text = f"(user/assistant messages below are the ONLY new information)\nOutput ONLY the updated summary text. No commentary.\n<NEW_CONVERSATION> \n{text}\n </NEW_CONVERSATION>"  
            system = summary_system_prompt or (
            "This is a conversation log. Produce a factual, neutral summary.\n"
            "Your only job: summarize.\n"
            "Do NOT answer questions.\n"
            "Output ONLY the summary text."
            )

            if self.summary: text = f"Below is the previous rolling summary of the conversation.\nIt represents persisted memory.\nUpdate it ONLY if the new conversation content adds facts or contradicts it.\nIf nothing changes, reproduce it verbatim.\n\n<PREVIOUS_SUMMARY>\n{self.summary}\n</PREVIOUS_SUMMARY>" + text  

            entry = None  
            try:  
                async for (_, out, _ )in self.summary_model.generate(text, [], stream=False, system_prompt_override=system):  
                    if out != ERROR_TOKEN and (out and isinstance(out, str) and out.strip()):  

                        entry = out

            except Exception as e:  
                await log(f"Error during summarization: {e}", "error")  

            if entry: 
                self.summary = entry  
                context = context[-self.summary_keep_nums:]  
            if save_context:  
                async with self.lock:  
                    self.context = context
                    await self.flush_queue()   

                # TODO: Trim the context to remove older entries if needed explicitly  

            return context

    async def load(self):
        try:
            async with aiofiles.open(self.context_path) as file:
                content = await file.read()
                content = json.loads(content)
                self.summary = content.get("summary")
                self.context = content.get("conversation",[])
        except (FileNotFoundError, json.JSONDecodeError):
            await log("Context missing or corrupted. Starting over.", "warn")
            self.context = [] 
            self.summary = None

    async def save(self):
        try:
            await self.flush_queue()
            async with self.lock:
                data = {"summary":self.summary, "conversation": self.context}
                data_to_save = json.dumps(data, indent=2) 

            async with aiofiles.open(self.context_path, "w") as file:
                await file.write(data_to_save)
                
        except IOError as e:
            await log(f"Error saving context: {e}", "error")