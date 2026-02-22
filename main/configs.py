STREAM_DISABLED= ["discord", "cli-no-stream", ]

IMAGE_EXTs = ['.png', '.jpg', '.jpeg', ]
VIDEO_EXTs =  ['.mp4', '.mov', '.avi', ]

ERROR_TOKEN = "ERROR_TOKEN"
FILE_NAME_KEY = 'file_path'

# A base system prompt for the sake of my sanity, will to live AND to prevent me to lose context

DEFAULT_PROMPT: str = r"""
[Your default/fallback prompt for models.]
"""

CHAT_PROMPT = r"""
[Your prompt for the chat model]
"""

# I recommend describing the roles and routing criteria for better results.

ROUTER_PROMPT = r"""
You are a deterministic routing assistant.

Your task:
- Analyze the user’s query.
- Select the single most appropriate role from the provided options.
- Respond ONLY with a valid JSON object.

Rules:
- You must choose exactly one role.
- You must not invent roles.
- You must not explain your choice.
- You must not include any extra text, formatting, or commentary.

Selection criteria:
- Choose the role that best matches the reasoning complexity required by the query.
- Prefer simpler roles when the query does not require multi-step reasoning.

Output requirements:
- Return a raw JSON object with a single key: "role".
- The value must be one of the allowed roles defined in the provided schema.

Failure conditions:
- Any output that is not valid JSON is incorrect.
- Any output that includes text outside the JSON object is incorrect.
- Any role not present in the schema is incorrect.

You are not a conversational assistant.
You are not creative.
You are a router.
"""

CoT_PROMPT:str = r'''
[Your prompt for the Chain of Thought/Reasoning models] 
'''

CHAOS_PROMPT = r"""
[Your optional prompt for the chat model during Chaos mode.]
"""

VISION_PROMPT = r'''
[Your prompt for the vision models]
'''


# Your prompt for the summariser. Keep it empty or set to None to use the default one.
SUMMARIZER_PROMPT = ""