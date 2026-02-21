import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

def run_command(args):
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {' '.join(args)}: {e}")
        sys.exit(1)

async def setup():
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    ollama_installed = input("Do you have Ollama installed? (y/n): ").lower() == 'y'
    if not ollama_installed:
        if sys.platform == "linux":
            install = input("Linux detected. Install Ollama now? (y/n): ").lower() == 'y'
            if install:
                subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
        else:
            print("Please download Ollama for your OS: https://ollama.com/download")
            return
        
    config_path = Path("main/Models_config.json")
    if config_path.exists():
        download = input(f"Found {config_path}. Download models now? (y/n): ").lower() == 'y'
        if download:
            with open(config_path) as f:
                models = json.load(f)
            

            serve_proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            
            try:
                for model in models:
                    name = model.get("ollama_name")
                    if name:
                        print(f"Pulling {name}...")
                        subprocess.run(["ollama", "pull", name], check=True)
            finally:
                serve_proc.terminate()
                print("Setup Complete.")
    else:
        print(f"Warning: {config_path} not found. Skipping model downloads.")

if __name__ == "__main__":
    asyncio.run(setup())