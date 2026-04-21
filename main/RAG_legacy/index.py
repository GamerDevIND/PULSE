import json
import numpy
import aiofiles
from copy import deepcopy

def index(chunk:str, vector:list[float] | list[list[float]], metadata= None):
    r= {
        'text':chunk,
        'vector':vector
    }
    if metadata:
        r['metadata'] = metadata

    return r

def generate_index(chunks:list[str], vectors: list[list[float]], metadata:None | list = None):
    metadata = deepcopy(metadata)
    if metadata is None:
        metadata = [None] * len(chunks)
    elif metadata and len(metadata) <= len(chunks):
        metadata.extend(None for _ in range(len(chunks) - len(metadata)))
    else:
        raise RuntimeError

    if len(chunks) != len(vectors):
        raise RuntimeError

    l = list(zip(chunks, vectors, metadata))
    indices = []
    for c, v, m in l:
        indices.append(index(c, v, m))

    return indices

async def write_index(indices, filename="./index.json"):
    clean_index = []
    for entry in indices:
        item = entry.copy()
        if isinstance(item['vector'], numpy.ndarray):
            item['vector'] = item['vector'].tolist()
        clean_index.append(item)

    async with aiofiles.open(filename, 'w') as f:
        s = json.dumps(clean_index, indent=2)
        await f.write(s)

async def read_index(filename="./index.json"):
    entries = []

    async with aiofiles.open(filename, 'r') as f:
        c = await f.read()
        c = c.strip()
        data = json.loads(c)
        for item in data:
            vec = numpy.array(item["vector"])
            norm = numpy.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                entries.append({
                    "text": item["text"],
                    "vector": vec,
                    'metadata': item.get('metadata')
                })

    return entries

def cosine_similarity(a, b):
    return numpy.dot(a, b)

async def search(embedded_query, top_k=3, filename="./index.json", entries = None):
    query_vec = numpy.array(embedded_query)
    norm = numpy.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    if entries is None:
        entries =  await read_index(filename)

    results = []

    for entry in entries:
        score = cosine_similarity(query_vec, entry["vector"])
        results.append((score, entry["text"], entry['metadata']))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]