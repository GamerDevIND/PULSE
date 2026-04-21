import asyncio
from .manager import RAG_manager
from main.models.model_instance import LocalEmbedder
import sys
import io
from main.configs import EMBEDDING_MODEL_ROLE
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

async def main(): 
    model = LocalEmbedder(EMBEDDING_MODEL_ROLE, "Embedder", 'embeddinggemma', 11434)
    file = r"G:\PULSE\JARVIS-Test\main\RAG\test.json"
    try:
        await model.warm_up()
        print('warmed up')
        
        manager = RAG_manager(file, model, True)
        # await manager.index_document(r"D:\Study Notes\NCERT books\English\jeff101.pdf")
        await manager.load()
        print('loaded index')

        query = input('>>> ')
        print('retrieving...')
        r = await manager.retrieve(query)
        print(f"retrieved: {len(r)} results")

        for i, (score, text, _) in enumerate(r):
            print(f"{i}. {score}:\n{text}\n\n\n")
            
    except Exception as e:
        print(f"Extraction/Search failed: {e}")
    finally:
        await model.shutdown()

asyncio.run(main())