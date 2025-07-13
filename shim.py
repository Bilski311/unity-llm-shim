# shim.py ─ minimal bridge between LLMUnity and OpenAI
import os, json
from openai import OpenAI
from flask import Flask, request, jsonify, Response, stream_with_context

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

app = Flask(__name__)

# --- tiny helpers -----------------------------------------------------------
def to_openai_messages(req: dict) -> list[dict]:
    """
    Translate the LLMUnity payload into OpenAI's 'messages' list.
    Expected keys in the request body:
        prompt   : current user prompt  (string)
        history  : optional chat history [{'role': 'user'|'assistant', 'content': str}, …]
    """
    hist = req.get("history", [])
    prompt = req.get("prompt") or req.get("query") or req.get("text")
    if prompt is None:
        raise ValueError("No prompt in request")
    return hist + [{"role": "user", "content": prompt}]

# --- main endpoint ----------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    stream = bool(data.get("stream", False))

    messages = to_openai_messages(data)

    # If LLMUnity asked for a streaming reply, proxy the stream token-by-token
    if stream:
        def gen():
            for chunk in client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=data.get("temperature", 0.7),
                    stream=True):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return Response(stream_with_context(gen()), mimetype="text/plain")

    # Non-streaming (one-shot) mode
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=data.get("temperature", 0.7),
    )
    msg = resp.choices[0].message
    # Return in the flat format LLMUnity expects
    return jsonify({
        "role": msg.role,
        "content": msg.content,
        "finish_reason": resp.choices[0].finish_reason,
    })

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 11434)))
