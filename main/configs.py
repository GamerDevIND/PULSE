STREAM_DISABLED= ["discod", "cli-no-stream"]

IMAGE_EXTs = ['.png', '.jpg', '.jpeg']
VIDEO_EXTs =  ['.mp4', '.mov', '.avi']

ERROR_TOKEN = "__ERROR__"

DEFAULT_PROMPT: str = r"""
[ Your Default or fallback system prompt ]
"""

CHAT_PROMPT = r"""
[ Your prompt for the chat model ]
"""


ROUTER_PROMPT = r"""
[ Your prompt for the router. You may need to tweak the roting code logic to make it work ]
"""


CoT_PROMPT:str = r'''
[ Your prompt for the reasoning / Chain of Thought (CoT) model ]
'''

CHAOS_PROMPT = r"""
[ Optional: Alter ego of your assistant. MAKE IT CHAOTIC! ]

"""

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
