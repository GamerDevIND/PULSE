from main.models.model_instance import LocalEmbedder
from main.models.models_profile import RemoteEmbedder
from main.models.openrouter_model import OpenRouterEmbedder
from main.configs import ERROR_TOKEN
from main.utils import log

async def embed(embedder:LocalEmbedder|RemoteEmbedder | OpenRouterEmbedder, chunks:list[str], auto_warm_up = True):
    
    if auto_warm_up and not embedder.warmed_up:
        await embedder.warm_up()

    embeds = await embedder.embed(chunks)

    if embeds == ERROR_TOKEN:
        await log('An error occured during embedding!', 'error')
        raise RuntimeError
    elif embeds is None:
        await log("Embeddings not found ('None')", 'error')
        raise RuntimeError

    return embeds