from .chunking import chunk
from .embedding import embed
from .index import generate_index, write_index, read_index, search
from .reading import read
import os
import re
import asyncio
from main.models.model_instance import LocalEmbedder
from main.models.models_profile import RemoteEmbedder
from main.models.openrouter_model import OpenRouterEmbedder
from main.configs import RAG_MIN_SCORE
from main.utils import log

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class RAG_manager:
    def __init__(self, index_file, embedder : None | LocalEmbedder | RemoteEmbedder | OpenRouterEmbedder, embedder_auto_warm_up = True) -> None:
        self.index_file = index_file
        self.embedder = embedder
        self.index = []
        self.embedder_auto_warm_up = embedder_auto_warm_up
        self.cache = {}

    async def load(self):
        if os.path.exists(self.index_file):
            self.index = await read_index(self.index_file)
        else:
            self.index = []
    
    async def save(self):
        await write_index(self.index, self.index_file)

    async def retrieve(self, query:str, top_k:int=3, min_score = RAG_MIN_SCORE):
        if (query.lower().strip(), min_score, top_k) in self.cache:
            return self.cache[(query.lower().strip(), min_score, top_k)]
        
        if not self.embedder:
            raise ValueError
        
        await log(f'Retrieving: {query}', 'info')
        
        embedded_query = await embed(self.embedder, [query], self.embedder_auto_warm_up)
        r = await search(embedded_query[0], top_k, self.index_file, self.index)

        r = filter(lambda x: x[0] >= min_score, r)
        r = list(sorted(r, key=lambda x: x[0], reverse=True))
        r = [{"text": _[1], "score":_[0], "metadata": _[2]} for _ in r]
        self.cache[(query.lower().strip(), min_score, top_k)] = r
        return r
    
    async def index_document(self, file_path:str, metadata=None, auto_save = True):
        await log(f'Indexing document: {file_path}', 'info')
        contents = await read(file_path)
        await self.index_text(contents, metadata=metadata, auto_save=auto_save)

    async def index_text(self, text, metadata=None, auto_save = True):
        if not self.embedder:
            raise ValueError
        
        await log(f'Indexing text...', 'info')
        
        chunks = list(dict().fromkeys(chunk(text)).keys())
        embeds = await embed(self.embedder, chunks, self.embedder_auto_warm_up)
        idx = generate_index(chunks, embeds, [metadata])
        for item in idx:
            if len(await self.retrieve(item['text'], min_score=0.87)) < 1:
                self.index.append(item)

        seen = set()
        new_index = []

        for entry in self.index:
            key = normalize(entry["text"])
            if key not in seen:
                seen.add(key)
                new_index.append(entry)

        self.index = new_index
        self.cache.clear()
        if auto_save:
            await self.save()

    async def clear_index(self, auto_delete = True):
        self.index.clear()
        if auto_delete:
            await self.delete_index_db()
    
    async def delete_index_db(self):
        if os.path.exists(self.index_file):
            await asyncio.to_thread(os.remove, self.index_file)