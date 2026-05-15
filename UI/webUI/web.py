import pathlib
import sys
import json
root_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

import asyncio
from quart import Quart, render_template, jsonify, request, Response, stream_with_context
import datetime
import random
import traceback

from main.AI import AI
from main.utils import Logger, estimate_tokens
from main.configs import USERNAME, DEFAULT_PROMPT, CHAOS_PROMPT, RAG_MIN_SCORE

app = Quart(__name__)
ai = AI("main/Models_config.json", mode='multi', use_RAG=False)
notification_queue = asyncio.Queue()
models_status_cooldown_secs = 0.8
SSE_timeout = 30

def get_greeting():
    
    def formal():
        now = datetime.datetime.now()
        h = now.hour
        if 12 > h >= 4:
            greeting = "Good Morning"
        elif 12 <= h < 18:
            greeting = "Good Afternoon"
        else:
            greeting = "Good Evening"
        
        return greeting

    greetings:list[str] = []
    greetings.append(formal())
    greetings.append("Good to see you")
    greetings.append("Welcome")
    if USERNAME and USERNAME.strip(): greetings = list(map(lambda x : x + f", {USERNAME}.", greetings))
    greetings.append("What's on your mind today?")
    greetings.append("What are we doing today?")
    greetings.append("Where should we start?")
    greeting = random.choice(greetings)
    return greeting

shutdown_event = asyncio.Event()
startup_task = None

async def shutdown():
        try:
            await ai.shut_down()
        except Exception as e:
            await Logger.log_async(f"Error during shutdown: {e}; {traceback.format_exc()}", 'error')

@app.after_serving
async def shutdown_app():
    await shutdown()

@app.before_serving
async def startup():
    startup_task = asyncio.create_task(ai.init("web"))

@app.route('/')
async def index():
    current_greeting = get_greeting()
    return await render_template("chat.html", greeting = current_greeting, name = USERNAME, 
                                 chats = await ai.context_manager.list_conversations(),
                                 estimated_tokens = 0,
                                 mode = ai.mode,
                                 models_status = ai.backend.get_models_state() if ai.backend else [])

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
        c = {
            "greeting": "",
            "name": USERNAME, 
            "chats":chats, 
            "current_chat": chat,
            "history": chat_data,
            "summary": chat_summary,
            "facts": chat_facts,
            "estimated_tokens": est,
            "mode": ai.mode,
            "models_status": ai.backend.get_models_state() if ai.backend else []
        }
        return await render_template("chat.html", **c)
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
        yield json.dumps({"chat_id": c.id, "content": "", "thinking": "", "tools": []}) + "\n"
        
        async for thinking, content, tools in gen.stream():
            out = json.dumps({"thinking": thinking, "content": content, "tools": tools}) + "\n"
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
            async for thinking, content, tools in gen.stream():
                out = json.dumps({"thinking": thinking, "content": content, "tools": tools}) + "\n"
                yield out
                
        except asyncio.CancelledError:

            await ai.cancel_generation(gen.session_id)
            await Logger.log_async(f"Stream cancelled for {cid}", "warn")

    return Response(generate(), mimetype='application/x-ndjson')

@app.route("/settings", strict_slashes=False)
async def settings():
    system_prompts = ai.system_prompts.copy()
    system_prompts.update({'default': DEFAULT_PROMPT, 'chaos': CHAOS_PROMPT})
    config_state = {
        "username": USERNAME,
        "mode": ai.mode,
        "use_rag": getattr(ai, 'use_RAG', False),
        "models_status": ai.backend.get_models_state() if ai.backend else [],
        'system_prompts': ai.system_prompts,
        "rag_min_score": RAG_MIN_SCORE
    }

    return await render_template('full_details.html', **config_state)

@app.route('/api/models-status')
async def models_status_api():
    async def event_generator():
        def get():
            return ai.backend.get_models_state() if ai.backend else []
        
        while True:
            try:
                try:
                    data = await asyncio.wait_for(asyncio.to_thread(get), timeout=SSE_timeout)
                    yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(models_status_cooldown_secs)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
            except asyncio.CancelledError:
                break

    return Response(event_generator(), mimetype="text/event-stream")

@ai.event('*')
async def all_events(**kwargs):
    event_name = kwargs.get('event_name')
    msg = kwargs.get('msg')
    
    type_map = {
        ai.event_bus.ERROR: "error",
        ai.event_bus.SUMMARISING_FAILED: 'error',
        ai.event_bus.WARN: "warn",
        ai.event_bus.SUCCESS: "success",
        ai.event_bus.INFO: "info",
        ai.event_bus.MODELS_LOADED: "success",
        ai.event_bus.INITIALISED: "success"
    }

    if (not event_name) or event_name == ai.event_bus.GENERATION_CHUNK:
        return

    message = msg if msg is not None else event_name.replace("_", " ").capitalize().strip()
    ev = {
        "message": message,
        "type": type_map.get(event_name, 'info'),
        "event": event_name
    }
    await notification_queue.put(ev)

@app.route('/events/notifications')
async def stream_notifications():
    async def event_generator():
        while True:
            try:
                try:
                    data = await asyncio.wait_for(notification_queue.get(), timeout=SSE_timeout)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        
            except asyncio.CancelledError:
                break
            
    return Response(event_generator(), mimetype="text/event-stream")

@app.route("/temp_chat", methods=['POST'])
async def temp_chat_endpoint():
    data = await request.get_json()
    query = data.get('message')
    
    c = await ai.context_manager.create_temp_chat()
    cid = c.id
    
    gen = await ai.create_generation(query, cid=cid, use_memory=False)
    if not gen:
        return jsonify({"error": "Generation failed"}), 500

    async def generate():
        try:
            yield json.dumps({"chat_id": cid, "content": "", "thinking": "", "tools": []}) + "\n"
            
            async for thinking, content, tools in gen.stream():
                yield json.dumps({"thinking": thinking, "content": content, "tools": tools}) + "\n"
                
        except asyncio.CancelledError as e:
            await ai.context_manager.delete_conversation(cid)

    return Response(stream_with_context(generate)(), mimetype='application/x-ndjson')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)