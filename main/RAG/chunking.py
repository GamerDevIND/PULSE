import re

def total_tokens(_chunk: list):   
    return len(" ".join(_chunk).split()) * 1.3

def chunk(text:str, limit=700, overlap_size= 1) -> list[str]:

    text = re.sub(r'Reprint \d{4}-\d{2}', '', text)

    if total_tokens(text.split()) <= limit:
        return [text.strip()]

    chunks = []
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    if len(paragraphs) <= 1:
        sentences = re.split(r'(?<!\b(?:Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc)\.)(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            words = text.split()
            step = max(1, int(limit / 1.3))
            chunks = [
                " ".join(words[i:i+step])
                for i in range(0, len(words), step)
            ]
            return chunks
        
        unit = sentences
    else:
        unit = paragraphs

    buffer = []

    for u in unit:
        if total_tokens(u.split()) > limit:
            if buffer:
                chunks.append('\n\n'.join(buffer))
                buffer = []
            chunks.extend(chunk(u, limit, overlap_size))
        elif total_tokens(buffer + [u]) <= limit:
            buffer.append(u)
        else:
            chunks.append('\n\n'.join(buffer))
            overlap = buffer[-overlap_size:] if len(buffer) >= overlap_size else []
            buffer = overlap + [u]

    if buffer:
        chunks.append('\n\n'.join(buffer))

    return chunks