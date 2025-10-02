from datetime import datetime
import os
import aiofiles

async def log(message: str, level, append = True):
    log_dir = os.path.join( "main", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H:%M:%S - %d / %m  / %Y ")
    emoji = {
        "info": "ℹ️",
        "warn": "⚠️",
        "error": "🟥",
        "success": "✅"
    }.get(level, "")
    text = f"{emoji} [{level}] {message} - [{timestamp}]\n"
    log_file = os.path.join(log_dir, 'log.log')
    async with aiofiles.open(log_file, "w" if not  append else "a") as f:
        print(text)
        await f.write(text)
