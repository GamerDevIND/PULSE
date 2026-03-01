# P.U.L.S.E.

Personal Unified Logic System Entity (V1.1)

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

And Single Server mode is still experimental. Expect unpredictable behaviour.

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


5. Write some interface code and the backend might be on AI.py


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
        "system_prompt": ""
    },
    {
        "role": "router",
        "name": "Router",
        "ollama_name":"smollm2:135m",
        "has_tools": true,
        "has_CoT": false,
        "has_vision": false,
        "port": 11435,
        "system_prompt": ""
    },
    {
        "role": "vision",
        "name": "MoonDream",
        "ollama_name": "moondream",
        "has_tools": false,
        "has_CoT": false,
        "has_vision": true,
        "port": 11543,
        "system_prompt" : ""
    },
    ...
]
```

Then the AI class will handle them like little smart minions slaves

2. System prompts are also available. Just configure it.


# Future plans:

- Voice I/O

- Vision (Image added. Video added. Audio is under consideration. Please stand by.)

- Agentic mode (because why the hell not / WIP please don't contact emergency services those screams AREN'T MINE.)

- Perception (hoping to be added on v2.x)

- Caffeine

- Sleep

- My 3 brain cells can't think of anything else


# Why Local?:

I'm not interested in giving my data and money to OpenAI and Google

### **You don't know how LLMs (transformers) work? Here's a quick overview:**

A transformer has:

1. **ENCODER**: IT ENCODES THE DATA (IMGS TEXT AUDIO AND EVERY SINGLE DATA TYPE KNOWN TO MANKIND) INTO LIST OF NUMBERS, why? bcuz somewhere in the past, computer said to number: "I love you <3" and rest is history, THE ENCODER FOR SOME APPARANT REASON ALSO LEARNS LIKE (not really) THE ACTUAL MODEL (how? idk), why? only to "learn" the relation between words.. like thats the model's work


2. **DECODER**: ITS THE TWIN BROTHER OF ENCODER BUT DOES THE EXACT OPPOSITE, IT CONVERTS THE ARRAY OF NUMBERS PUKED BY THE MODEL AND CONVERTS THEM INTO THE DESIRED DATA TYPE (IMGS TEXT AUDIO AND EVERY SINGLE DATA TYPE KNOWN TO MANKIND), FOR SOME REASON IT ALSO "learns" PATTERN, away from the encoder i assume, (and if they share the same patterns why dont use the same learnt vocab for both)


3. **THE ACTUAL MODEL**: THIS PIECE OF SHIT HAS SOME COOL MATHS GOING ON, it eats the numbers from the encoder, does the digestion process and pukes the digested shit out to the decoder, and the digestive system consists of:
i) *ATTENTION BLOCK* (the black magic): it allows the tensors of other words communicate to each other like its a family function and like the relatives they allow, ah yes you'll become an enginner becuz some uncle's brother's son's step son is a doctor who lives on mars
ii) *MLPs* : the chill guy, it just learns patterns just like other models thats it 

these repeats for the rest of eternity, until the output layer is reached



the model takes the formed "thing" and predicts what should come next using probablity, why? its fun. how? temperature, unlike my friends's crush, its the parameter which controls the empathy of the model, this gives the option for the backward words to have a chance and appends it until the brakes aren't pressed  

----- QUESTIONS I MANAGED TO ANSWER -----

1. Why do the twins not share the same vocab? Ans) they sometimes do. Only when the input and output formats are the same like txt2txt but if the formats are different it's not possible (at least not in this universe) to share the same vocab. As the decoding needs a different algorithm to decode the output like in text2img



--- POINTS I MISSED ---

1. **TOKENISATION**: breaks the input into smaller words (ex: i eat dirt => [i, eat, dirt]) or subwords (ex: cinematic universe -> [cinema, tic, uni, verse] . SPOILER: thisishowllmscanreadthistexts


2. **Positional embedding**: as llms are just math equations throwing pseudo-random predictions (and ironically replacing humans) it lacks the basic understanding of positions thus we need to yet another matrix just for the sake of GPS and sanity. Otherwise to a llm "I eat grass" is same as "eat i grass"


3. **Etc**: it's not over yet i just lack the knowledge and too lazy to search it

4. **Current LLMs**: Most modern LLMs are decoder only, meaning the "encoder" is just the embedding layer and the model just predicts the next token based on previously generated ones.

---

# End note:

If this README confuses you don't worry so does life. Congratulations you have successfully made this far. Here's some coffee for you: ☕. Thanks a lot for wasting your time with me. I really appreciate it!

