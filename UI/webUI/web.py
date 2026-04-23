import pathlib
import sys
import json
root_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

import asyncio
from quart import Quart, render_template, jsonify, request, Response, stream_with_context
import datetime

from main.AI import AI
from main.utils import log, estimate_tokens
from main.configs import USERNAME

app = Quart(__name__)
ai = AI("main/Models_config.json", mode='multi', use_RAG=False)
last = ""

def get_greeting():
    now = datetime.datetime.now()
    h = now.hour
    if 12 > h >= 4:
        greeting = "Good Morning"
    elif 12 <= h < 18:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"
    
    return greeting

shutdown_event = asyncio.Event()

async def shutdown():
    if not shutdown_event.is_set():
        shutdown_event.set()
        await ai.shut_down()

@app.after_serving
async def shutdown_app():
    await shutdown()

@app.before_serving
async def startup():
    asyncio.create_task(ai.init("web"))

@app.route('/status')
async def get_status():
    global last
    async with ai.lock:
        status = ai.status
    if last != status['status']:
        await log(f"System Status: {status}", 'info')
        last = status['status']
    return jsonify(status)

@app.route('/')
async def index():
    current_greeting = get_greeting()
    return await render_template("new.html", greeting = current_greeting, name = USERNAME, 
                                 chats = await ai.context_manager.list_conversations())

@app.route('/chat/<cid>')
async def chat_view(cid):
    try:
        chat = await ai.context_manager.get_conversation(cid)
        chat_data = await chat.get_context()
        chat_summary = await chat.get_summary()
        chat_facts = await chat.get_facts()
        chats = await ai.context_manager.list_conversations()
        words = [c["content"] for c in await chat.get_context()]
        words = " ".join(words)
        est = round(estimate_tokens(words))
        
        return await render_template("new.html", 
                                     greeting="",
                                     name=USERNAME, 
                                     chats=chats, 
                                     current_chat=chat,
                                     history=chat_data,
                                     summary = chat_summary,
                                     facts = chat_facts,
                                     estimated_tokens = est)
    except KeyError:
        return "Chat not found", 404
    
@app.route('/chat/new', methods=['POST'])
async def new_chat():
    data = await request.get_json()
    query = data.get('message')
    
    c = await ai.context_manager.new_conversation()
    
    gen = await ai.create_generation(query, cid=c.id, use_memory=False)

    if not gen:
        return "Generation creation failed", 500
    
    async def generate():
        yield json.dumps({"chat_id": c.id, "content": "", "thinking": ""}) + "\n"
        
        async for thinking, content in gen.stream():
            out = json.dumps({"thinking": thinking, "content": content}) + "\n"
            yield out

    return Response(stream_with_context(generate)(), mimetype='application/x-ndjson')

@app.route('/chat/<cid>/rename', methods=['POST'])
async def rename_chat(cid):
    data = await request.get_json()
    new_name = data.get('name', 'New Chat')
    await ai.context_manager.rename_convo(cid, name=new_name)
    return jsonify({"status": "success"})

@app.route('/chat/<cid>/delete', methods=['DELETE'])
async def delete_chat(cid):
    await ai.context_manager.delete_conversation(cid)
    return jsonify({"status": "deleted"})

@app.route('/chat/<cid>', methods=['POST'])
async def send_message(cid):
    data = await request.get_json()
    query = data.get('message')
    
    gen = await ai.create_generation(query, cid=cid, use_memory=False)

    if not gen:
        return jsonify({"error": "Generation failed"}), 500
    
    async def generate():
        try:
            async for thinking, content in gen.stream():
                out = json.dumps({"thinking": thinking, "content": content}) + "\n"
                yield out
                
        except asyncio.CancelledError:

            await ai.cancel_generation(gen.session_id)
            await log(f"Stream cancelled for {cid}", "warn")

    return Response(generate(), mimetype='application/x-ndjson')
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False,)
