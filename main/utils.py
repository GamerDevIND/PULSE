from datetime import datetime
import asyncio
import aiofiles

async def log(message: str, level, log_file: str = "logs/log.log", append = True):
    timestamp = datetime.now().strftime("%H:%M:%S - %d / %m  / %Y ")
    emoji = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "🟥",
        "success": "✅"
    }.get(level, "")
    text = f"{emoji} [{level}] {message} - [{timestamp}]\n"
    async with aiofiles.open(log_file, "a" if append else "w") as f:
        print(text)
        await f.write(text)