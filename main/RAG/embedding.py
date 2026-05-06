from main.models.model_instance import LocalEmbedder
from main.models.models_profile import RemoteEmbedder
from main.models.openrouter_model import OpenRouterEmbedder
from main.configs import ERROR_TOKEN
from main.utils import Logger

async def embed(embedder:LocalEmbedder | RemoteEmbedder | OpenRouterEmbedder, chunks:list[str], auto_warm_up = True, chunk_size = 16):
    if auto_warm_up and not embedder.warmed_up:
        await embedder.warm_up()

    if isinstance(chunks, str):
        chunks = [chunks]
    elif len(chunks) <= 1 and isinstance(chunks, (tuple, list)):
        chunks = [*chunks]


    embeddings = [] 

    for i in range(0, len(chunks), chunk_size):
        chunk = chunks[i : i + chunk_size]
        embeds = await embedder.embed(chunk)
        if embeds == ERROR_TOKEN:
            await Logger.log_async('An error occured during embedding!', 'error')
            raise RuntimeError
        elif embeds is None:
            await Logger.log_async("Embeddings not found ('None')", 'error')
            raise RuntimeError
        
        embeddings.extend(embeds)

    return embeddings