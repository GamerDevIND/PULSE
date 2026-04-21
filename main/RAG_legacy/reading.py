import aiofiles
import os
import pymupdf
import asyncio
from main.utils import log

async def read_txt(file_path:str):
    async with aiofiles.open(file_path, "r", encoding = "utf-8") as f:
        return await f.read()


async def read_pdf(file_path:str):
    def read_docs():
        with pymupdf.open(file_path) as doc:
            contents = []

            for page in doc:
                t = page.get_text('text')
                contents.append(str(t))

            text = "\n\n".join(contents)
            return text
    text = await asyncio.to_thread(read_docs)

    if not text.strip():
        await log('Text not found, returning empty.', 'warn')

    return text



async def read(file_path:str):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in ['.txt', '.py', '.md', '.json', '.js', '.html', '.css', '.java', '.cpp']:
        return await read_txt(file_path)
    elif ext in ['.pdf']:
        return await read_pdf(file_path)
    else:
        raise ValueError