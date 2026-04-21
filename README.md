# P.U.L.S.E.

Personal Unified Logic System Entity (v1.3.5 - Backend)

This is the polite version of the README.

For the director's cut with more swearing and less corporate read: README_but_3AM.md

# What's this project?
PULSE, or Personal Unified Logic System Entity is an attempt to create a local, opensource, and sovereign LLM semi-agentic runtime, running on:

- [Ollama](https://ollama.com/)

- Local machine

- [Python](https://www.python.org/)

- Open source LLMs


PULSE IS:

- An attempt to orchestrate and manage multiple concurrent LLMs locally for better UX with the tradeoff of increased resource consumption

- A multimodel architecture for running multiple local LLMs


It launches the configured models into their separate instances and manages them to avoid constant unload and reload latency.


Some features I'm told to highlight:

- LLM based routing with manual overrides. It utilizes another small LLM as an intent classification and correct role selection to choose the model the query is directed to. It also supports manual override for power users.

- It supports two modes. `SingleServer` (mode = "single") and `MultiServer` (mode = "multi"). Multi server mode launches a separate Ollama server for each model instance. Single server mode is an attempt to manage multiple concurrent models in with single Ollama server.

- I've also implemented a basic decorator based plugin system for tool calling (function calling) for better semi-agentic feel.


***Warning:*** Single server mode is still an experimental feature. Unstable behaviour is expected.

---

# Changes in v1.3.5

- **Vector Database Integration:** Switched from naive JSON storage to [LanceDB](https://lancedb.com/).
- **Hybrid Search:** Implemented a dual-path retrieval system (Vector + Full-Text Search).
- **Reranking:** Integrated a reranker for significantly higher precision in context retrieval.
- **Atomic Operations:** Uses `merge_insert` (Upsert) logic for deduplication and efficient indexing.
- **Enhanced Tooling:** - `propose_memory`: Model suggests a fact to save; user must confirm (prevents self-poisoning).
    - `update_memory` / `delete_memory` / `list_memories`: Full CRUD support for long-term storage, user exclusive, needs to be handled by UI backend for better UX.
- **Fixed Generation Cancellation:** Better handling of async task cleanup.
- **Experimental WebUI and multi conversation support**: I've added an experimental WebUI using [`Quart`](https://pypi.org/project/Quart/) and multiple conversation support.

- Event Bus prototype 

Known major limitations:
- No tests (yes, I know)
- Manual FTS indexing can be CPU-heavy on very large datasets.
- Expect bugs from mobile coding

---

# Tech stuff

- [Python](https://www.python.org/)

- [aiohttp](https://pypi.org/project/aiohttp/), [aiofiles](https://pypi.org/project/aiofiles/) etc (see requirements.txt): because it's a bad idea for responsiveness to pause the main loop to get a response. (SPOILER: it's for multimodal and async I/O for http requests and files.)
- Multimodal support: served by Ollama. on their very own port to actually have multiple models (Yes, this is why I'm using aiohttp instead of ollama) and async file I/O for non-blocking file I/O in the main loop.

- **[LanceDB](https://lancedb.com/) & PyArrow:** For high-performance vector storage and schema-enforced data handling.

- Per model configuration: system prompt and other configurations for each model is available for control and customisation.

- Swappable models: as this is running locally nobody gives a fuck what you're doing you can do anything and everything that includes you can use any and every model you want.

---

# How to use?:

1. Download and install [Ollama](https://ollama.com/download)


2. Download your models. ANY MODEL. LIKE ANY MODEL AVAILABLE. My suggestions are:



* [Smollm2](https://ollama.com/library/smollm2)

* [Zephyr](https://ollama.com/library/zephyr)

* [DeepSeek-R1](https://ollama.com/library/deepseek-r1)
* [MoonDream](https://ollama.com/library/moondream)


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

# Basic configuration setup guide:

1. You can define multiple models like this in a JSON file (preferably `Models_config.json`):


```json
[
    {
        "role": "chat", 
        "name": "Zephyr", 
        "model_name": "zephyr", 
        "has_tools": false, 
        "has_CoT": false, 
        "has_vision": false,
        "port":11434
       
    },
    {
        "role": "cot",
        "name":"Deepseek-R1",
        "model_name": "deepseek-r1:7b",
        "has_tools": true,
        "has_CoT":true,
        "has_vision": false,
        "port": 13345
        
    },
    {
        "role": "router",
        "name": "Router",
        "model_name":"smollm2:135m",
        "has_tools": true,
        "has_CoT": false,
        "has_vision": false,
        "port": 11435
        
    },
    {
        "role": "vision",
        "name": "MoonDream",
        "model_name": "moondream",
        "has_tools": false,
        "has_CoT": false,
        "has_vision": true,
        "port": 11543
       
    },
    ...
]
```

Then the AI class will handle them like little smart minions.

2. System prompts are also available. Just configure them.

---

### ***WARNING:*** On windows, make sure to open / launch the ollama desktop app before running the program, I hope this works fine on Linux and Mac.

# What changed?

[OpenRouter](https://openrouter.ai/) and [Ollama Cloud](https://ollama.com/search?c=cloud) support has been added for weak hardware by using cloud models instead of complete local inference with an optional `api_key` field in the configs. User can also provide `.env` variables through a configurable prefix found in `configs.py`. 
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

You can change `$` to whatever prefix you choose in `configs.py` and you may also put the key directly in the field removing the prefix completely, though this may cause a security hazard.
A new mode has also been added. OpenRouter models can be used by setting the `mode` to `'openrouter'` before launching the application. For now, you can only use openrouter OR ollama, one at a time, this limitation is expected to be removed in future updates.

---

# Future plans:

- Voice I/O

- Vision (Image added. Video added. Audio is under consideration. Please stand by.)

- Agentic mode (because why the hell not / WIP *expected* full drop: v1.5.x)

- Perception (hoping to be added on v2.x, I have no idea how I'll pull that off but eh let's see.)

- Caffeine

- Sleep

- Staring at walls


# Why Local?:

I'm not interested in giving my data and money to OpenAI and Google

### Documentation will be available in ARCHITECTURE.md soon if you're willing to know the internals.

---

# End note:

If this README confuses you don't worry so does life. Congratulations you have successfully made this far. Here's some coffee for you: ☕. Thanks a lot for wasting your time with me. I really appreciate it!
