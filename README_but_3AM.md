# P.U.L.S.E.

Personal Unified Logic System Entity (v1.3)

This is the chaotic version.

For the cleaner and more corporate read: README.md

# WHAT THE FUCK ON EARTH IS THIS PIECE OF SHIT?:

PULSE a.k.a. Personal Unified Logic System Entity is a replication of JARVIS (I just realised this implementation is closer to FRIDAY but I'll stick with JARVIS, bear with me) from Iron Man (MCU), but instead of arc reactor it runs on:

- Coffee

- [Ollama](https://ollama.com/)

- Local machine

- [Python](https://www.python.org/)

- Open source LLMs

- My slowly deteriorating mental stability


PULSE IS:

- A talking piece of fuck shit

- A multimodel architecture for running multiple local LLMs

- A unhinged system of LLMs which claims to be your friend, assistant and roaster


---

Oh, by the way I added tool calling (function calling).
Also The single server mode is still experimental. Expect unstable behaviour

---

# TECH STUFF (because sadly this shit needs to work):

- [Python](https://www.python.org/) (the snake, which bites)

- [aiohttp](https://pypi.org/project/aiohttp/), [aiofiles](https://pypi.org/project/aiofiles/) etc (see requirements.txt): because unfortunately nobody wants to pause their main loop to get a response. (SPOILER: it's for multimodal and async I/O for http requests)
- Multimodal support: served by Ollama. on their very own port to actually have multiple models (Yes, this is why I'm using aiohttp instead of ollama) and async file I/O for non-blocking file I/O in the main loop.

- Per model configuration: system prompt and other configurations for each model is available for some reason

- Swappable models: as this is running locally nobody gives a fuck what you're doing you can do anything and everything that includes you can use any and every model you want.

- Experimental features: random existential crisis with emotions. I don't know how it works.

---

# HOW TO ACTUALLY USE THIS PIECE OF SHIT:

1. Download and install [Ollama](https://ollama.com/download)


2. Download your models. ANY MODEL. LIKE ANY MODEL AVAILABLE. My suggestions are:



- [Smollm2](https://ollama.com/library/smollm2)

- [Zephyr](https://ollama.com/library/zephyr)

- [DeepSeek-R1](https://ollama.com/library/deepseek-r1)
- [MoonDream](https://ollama.com/library/moondream)


3. Install and setup [Python](https://www.python.org/downloads/)


4. Install the requirements:

```bash
pip install -r requirements.txt
```


5. Write some interface code and the backend might be on AI.py. Though there are some basic UI examples in the `UI/` folder.


6. Run and test the code


7. Enjoy

--- OR ---

Use the new `setup.py` for faster setup, it's still under development so expect some bugs, though I hope it'll work fine in Linux.

---

# BASIC CONFIGURATION SETUP GUIDE (if you care):

1. You can define multiple models like this in a JSON file:


```json
[
    {
        "role": "chat", 
        "name": "Zephyr", 
        "ollama_name": "zephyr", 
        "has_tools": false, 
        "has_CoT": false, 
        "has_vision": false,
        "port":11434, 
        "system_prompt": ""
    },
    {
        "role": "cot",
        "name":"Deepseek-R1",
        "ollama_name": "deepseek-r1:7b",
        "has_tools": true,
        "has_CoT":true,
        "has_vision": false,
        "port": 13345,
    },
    {
        "role": "router",
        "name": "Router",
        "ollama_name":"smollm2:135m",
        "has_tools": true,
        "has_CoT": false,
        "has_vision": false,
        "port": 11435,
    },
    {
        "role": "vision",
        "name": "MoonDream",
        "ollama_name": "moondream",
        "has_tools": false,
        "has_CoT": false,
        "has_vision": true,
        "port": 11543
    },
    ...
]
```

Then the AI class will handle them like little smart minions slaves

2. System prompts are also available. Just configure it.

---

### ***WARNING:*** On windows, make sure to open / launch the ollama desktop app before running the program, I hope this works fine on Linux and Mac.

# WHAT THE FUCK CHANGED?

- **LanceDB is here:** I nuked the shitty JSON-based RAG because it was slower than my grandma. It now uses a real-ass vector database.
- **Hybrid Search:** It searches via vectors AND keywords at the same time (hopefully). It’s like using both sides of the brain, and both of them actually work for once.
- **Reranking:** I added a reranker to rank the results, because the first things the AI finds are usually garbage.
- **Memory Saving:** Added a confirmation gate for `propose_memory`. The AI can’t just gaslight itself into a hallucination loop without you saying "Okay." (maybe)
- **CRUD for Memories:** You can now list, update, and delete the shit the AI remembers. Total control, finally.
- - **Experimental WebUI and multi conversation support**: I've added an experimental WebUI using [`Quart`](https://pypi.org/project/Quart/) and multiple conversation support. BECAUSE THE CLI IS TOO BORING!!!!
  
---

**[OpenRouter](https://openrouter.ai/) and [Ollama Cloud](https://ollama.com/search?c=cloud) support has been added for weak hardware by using cloud models instead of complete local inference with an optional `api_key` field in the configs. User can also provide `.env` variables through a configurable prefix found in `configs.py`. Because of course I did, inference on a CPU is painful.** 
For example the entry might look like

```json
{
    "role": "cot",
    "name": "LFM2.5-1.2B-Thinking ",
    "model_name": "liquid/lfm-2.5-1.2b-thinking:free",
    "has_tools": true,
    "has_CoT": true,
    "has_vision": false,
    "api_key": "$OPENROUTER_KEY"
}
```
You can also put a seperate port field in the entry but that'll be ignored as Openrouter doesn't support custom ports in its API.

You can change `$` to whatever prefix you choose in `configs.py` and you may also put the key directly in the field removing the prefix completely, though this may cause a security hazard and a disaster if leaked.
A new mode has also been added. OpenRouter models can be used by setting the `mode` to `'openrouter'` before launching the application. For now, you can only use openrouter OR ollama, one at a time, this limitation is expected to be removed in future updates (I hope so).

---

# Future plans:

- Voice I/O

- Vision (Image added. Video added. Audio is under consideration. Please stand by.)

- Agentic mode (because why the hell not / WIP please don't contact emergency services those screams AREN'T MINE.)

- Perception (hoping to be added on v2.x, I have no idea how I'll pull that off but eh let's see.)

- Caffeine

- Sleep

- Staring at walls

- My 3 brain cells can't think of anything else


# Why Local?:

I'm not interested in giving my data and money to OpenAI and Google

### For documentation check ARCHITECTURE.md (which will be provided soon) if your will to understand this chaos hasn't been satisfied.

---

# End note:

If this README confuses you don't worry so does life. Congratulations you have successfully made this far. Here's some coffee for you: ☕. Thanks a lot for wasting your time with me. I really appreciate it!