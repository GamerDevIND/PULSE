from subprocess import run, Popen
import json
import time

commands_1 = "pip install -r requirements.txt "
commands_2 = ""
ollama = input("Do you have Ollama installed? (Y/N) ").lower()
if ollama == "n":
    os = input("Which OS are you using? (W for Windows / L for Linux / M for MacOS) ").lower()
    if os == "l":
        if input("Do you want to install Ollama now? (Y/N) ").lower() == "y":
            commands_1 += "&& curl -fsSL https://ollama.com/install.sh | sh"
        else: 
            print("Ollama is required")
            exit()
    else:
        print("Please install Ollama first: https://ollama.com/download")
        exit()

models = input("Do you have the models config JSON file configured? (Y/N) ").lower()
if models == "y":
    if input("Would you like to download the configured models? (Y/N) ").lower() == 'y':
        with open("main/Models_config.json") as f:
            model_file:list[dict] = json.load(f)
            for model in model_file:
                name = model.get("ollama_name")
                if name is not None:
                    commands_2 += f"ollama pull {name} &&"
            
            commands_2 += "echo done"
else:            
    print("Please install the desired models")
    exit()

run(commands_1, shell=True, check=True)
s = Popen(['ollama', 'serve'], )
time.sleep(0.1)
run(commands_2, shell=True, check=True)
s.terminate()