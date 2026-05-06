import lancedb
import pyarrow
from datetime import datetime
import asyncio
from main.models.model_instance import LocalEmbedder
from main.models.models_profile import RemoteEmbedder
from main.models.openrouter_model import OpenRouterEmbedder
from main.configs import RAG_MIN_SCORE
from main.utils import Logger
from .chunking import chunk
from .reading import read
from hashlib import sha256
from .embedding import embed
from lancedb.rerankers import MRRReranker # you can use your own
import shutil
import os
import json
import re

def sanitize_id(oid: str) -> str:
    if not re.fullmatch(r"[a-fA-fa-f0-9]{64}", oid):
        raise ValueError("Invalid ID format")
    return oid

class RAG_manager:
    def __init__(self, embedder: None | LocalEmbedder | RemoteEmbedder | OpenRouterEmbedder, embedder_auto_warm_up = True, db_path="./rag_db", 
                 table_name="memories", embed_chunk_size = 24):
        self.embedder = embedder
        self.db_path = db_path
        self.table_name = table_name
        self.embedder_auto_warm_up = embedder_auto_warm_up
        self.embed_chunk_size = embed_chunk_size
        self.reranker = MRRReranker()
        self.cache = {}
        self.db = None
        self.table = None

    async def connect(self):
        if not self.db:
            self.db = await asyncio.to_thread(lancedb.connect, self.db_path)

    async def load(self):
        await self.connect()

        if self.db is None: 
            await Logger.log_async("DB not found!", 'error')
            raise ValueError

        if self.table: return
        
        table_names = await asyncio.to_thread(self.db.table_names)
        if self.table_name not in table_names:
            schema = pyarrow.schema([
                pyarrow.field("id", pyarrow.string()),
                pyarrow.field("text", pyarrow.string()),
                pyarrow.field("vector", pyarrow.list_(pyarrow.float32())),
                pyarrow.field("metadata", pyarrow.string()),
                pyarrow.field("timestamp", pyarrow.string()),
            ])
            self.table = await asyncio.to_thread(self.db.create_table, self.table_name, schema=schema)
            await asyncio.to_thread(self.table.create_scalar_index, "id")
        else:
            self.table = await asyncio.to_thread(self.db.open_table, self.table_name)

    async def save(self):
        if not self.table: await self.load()
        if not self.table: raise ValueError
        await asyncio.to_thread(self.table.optimize)

    async def index_document(self, file_path:str, metadata):
        await Logger.log_async(f'Indexing document: {file_path}', 'info')
        contents = await read(file_path)
        await self.index_text(contents, metadata=metadata)

    async def index_text(self, text: str, metadata: dict | None = None):
        if not self.embedder:
            raise ValueError("No embedder")

        await self.load()

        if not self.table: raise ValueError

        chunks = chunk(text)
        if not chunks:
            return
            
        chunks = list(dict.fromkeys(c.strip() for c in chunks if c.strip()))

        vectors = await embed(self.embedder, chunks, self.embedder_auto_warm_up, self.embed_chunk_size)

        records = []
        for chunk_text, vec in zip(chunks, vectors):
            cid = sha256(chunk_text.encode()).hexdigest()
            records.append({
                "id": cid,
                "text": chunk_text,
                "vector": vec,
                "metadata": json.dumps(metadata or {}),
                "timestamp": datetime.now().isoformat()
            })
        
        if records:

            await asyncio.to_thread(lambda *_: self.table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(records), ) # type:ignore

            await asyncio.to_thread(self.table.create_fts_index, "text", replace=True)
            self.cache.clear()

        await Logger.log_async(f"Indexed {len(records)} new chunks.", "info")

    async def retrieve(self, query: str, top_k: int = 5, min_score: float = RAG_MIN_SCORE):
        await self.load()
        if not self.table:
            return []
        
        if (query.lower().strip(), min_score, top_k) in self.cache:
            return self.cache[(query.lower().strip(), min_score, top_k)] 

        if not self.embedder:
            raise ValueError("No embedder configured")
        
        if len(self.cache) > 1024:
            self.cache.clear()

        def _helper_search(table:lancedb.Table, query:str, query_vec, reranker, top_k:int):
            return table.search(query_type='hybrid').vector(query_vec).text(query).rerank(reranker).limit(top_k).to_list()

        query_vec = (await embed(self.embedder, [query], self.embedder_auto_warm_up))[0]

        results = await asyncio.to_thread(_helper_search, self.table, query, query_vec, self.reranker, top_k)

        res = []

        for r in results:
            if "_relevance_score" in r:
                score = r["_relevance_score"]
            elif '_score' in r:
                score = r['_score']
            else:
                score = (1 - r.get("_distance", 1.0))

            if score >= min_score:
                res.append({
                    "text": r["text"],

                    "score": score,

                    "metadata": json.loads(r.get("metadata", "{}"))
                })

        res = sorted(res, key=lambda x: x['score'], reverse=True)

        self.cache[(query.lower().strip(), min_score, top_k)]  = res

        return res
    
    async def clear_index(self):
        if not self.db: await self.load()

        if not self.db: raise FileNotFoundError

        await asyncio.to_thread(self.db.drop_table, self.table_name)
        self.table = None

    async def delete_index_db(self, delete_full_db):
        if not self.db: await self.load()

        if not self.db: raise FileNotFoundError

        await asyncio.to_thread(self.db.drop_all_tables)

        if delete_full_db and os.path.exists(self.db_path):
            await asyncio.to_thread(shutil.rmtree, self.db_path)

        self.db = None
    
    async def search_metadata(self, metadata: str, limit: int = 10):
        await self.load()

        if not self.table: 
            return []
        

        res = await asyncio.to_thread(
            lambda: self.table.search().where(metadata).limit(limit).to_list() # type: ignore
            
        )
        return self._format_results(res)
    
    async def search_by_metadata_value(self, key: str, value: str, limit: int = 10):
        await self.load()
        if not self.table: 
            return []

        filter_query = f"metadata LIKE '%\"{key}\": \"{value}\"%'"
        
        res = await asyncio.to_thread(
            lambda: self.table.search().where(filter_query).limit(limit).to_list() # type: ignore
        )
        return self._format_results(res)
    
    async def filtered_search(self, query: str | None = None, metadata_filter: str | None = None, limit: int = 10):
        await self.load()
        if not self.table: 
            return []

        search_obj = self.table.search(query)
        
        if metadata_filter:
            search_obj = search_obj.where(metadata_filter)
            
        res = await asyncio.to_thread(lambda: search_obj.limit(limit).to_list())
        return self._format_results(res)
        
    async def text_search(self, query: str, limit: int = 10):
        await self.load()

        if not self.table: 
            return []

        res = await asyncio.to_thread(
            lambda: self.table.search(query).limit(limit).to_list() # type: ignore
        )
        return self._format_results(res)
    
    def _format_results(self, raw_results):
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": json.loads(r.get("metadata", r"{}")),
                "score": r.get("_score", None) 
            }
            for r in raw_results
        ]

    async def list_memories(self, limit = 10, offset = 0):
        await self.load()
        if not self.table:
            return []
        
        res = await asyncio.to_thread(lambda *_: self.table.search().offset(offset).limit(limit).to_list(),) # type: ignore

        return [
        {
            "id": r["id"],
            "text": r["text"],
            "metadata": json.loads(r.get("metadata", "{}"))
        }
        for r in res
    ]

    async def delete(self, mem_id:str):
        await self.load()
        if not self.table:
            return False
        
        try:
            await asyncio.to_thread(self.table.delete, f"id = '{sanitize_id(mem_id)}'")
            await asyncio.to_thread(self.table.create_fts_index, "text", replace=True)
            self.cache.clear()
            await Logger.log_async(f"Deleted memory {mem_id}", "info")
            return True
        except Exception as e:
            await Logger.log_async(f"Failed to delete memory {mem_id}: {e}", "error")
            return False
    
    async def update(self, mem_id: str, new_text: str):

        await self.load()
        if not self.table:
            return False
        
        if not self.embedder:
            return False
        
        new_text = new_text.strip()
        
        if len(new_text) < 10:
            return False
        
        try:
            new_vector = (await embed(self.embedder, [new_text], self.embedder_auto_warm_up, self.embed_chunk_size))[0]

            await asyncio.to_thread(
                self.table.update,
                where=f"id = '{sanitize_id(mem_id)}'",
                values={
                    "text": new_text,
                    "vector": new_vector,
                    "timestamp": datetime.now().isoformat()
                }
            )

            await asyncio.to_thread(self.table.create_fts_index, "text", replace=True)

            self.cache.clear()

            await Logger.log_async(f"Updated memory {mem_id}", "info")
            return True
        except Exception as e:
            await Logger.log_async(f"Failed to update memory {mem_id}: {e}", "error")
            return False