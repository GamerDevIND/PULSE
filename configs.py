STREAM_DISABLED= ["discord", "cli-no-stream"]

IMAGE_EXTs = ['.png', '.jpg', '.jpeg']
VIDEO_EXTs =  ['.mp4', '.mov', '.avi']

ERROR_TOKEN = "__ERROR__"
FILE_NAME_KEY = 'file_path'

DEFAULT_PROMPT: str = r"""
[ Your Default or fallback system prompt ]
"""

CHAT_PROMPT = r"""
[ Your prompt for the chat model ]
"""

CHAOS_PROMPT = r""" 
[ Your prompt for the chat model for the chaos mode override. You can make it unhinged, that's upto you.]
"""

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
[ Your prompt for the reasoning / Chain of Thought (CoT) model ]
'''

VISION_PROMPT = r''' 
[ Your prompt for the vision model ]
'''

SUMMARIZER_PROMPT = r'''
This is a conversation log. Produce a factual, neutral summary.
Your job is to summarise the conversation between the user and the assistant.
Do ***NOT*** respond to the prompt, just summarise the given conversation.
Keep intact essential or key informations, quote the important messages or references. Keep the key information easily machine parsable
Keep your answers machine parsable. Do ****NOT**** try to be helpful and answer the given query, just summarise the given conversation.
Do your only job properly: SUMMARIZE THE GIVEN CONVERSATION.
'''


