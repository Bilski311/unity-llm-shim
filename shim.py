import json, time
from flask import Flask, request, jsonify, Response, stream_with_context
from openai import OpenAI
import os
from flask_cors import CORS

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
app = Flask(__name__)
CORS(app,
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

def sse_llama_stream(messages, sampling):
    for chunk in client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
            **sampling):

        tok = chunk.choices[0].delta.content
        if tok:
            yield 'data: ' + json.dumps({
                "index": 0,
                "content": tok,
                "stop": False
            }, ensure_ascii=False) + '\n\n'

    yield 'data: ' + json.dumps({
        "index": 0,
        "content": "",
        "stop": True
    }) + '\n\n'


@app.route("/completion", methods=["POST"])
def chat():
    req = request.get_json(force=True)
    stream = bool(req.get("stream", False))

    sampling = {
        "temperature": req.get("temperature", 0.7),
        "top_p":       req.get("top_p", 1.0),
        "max_tokens":  req.get("n_predict", 256) if req.get("n_predict", -1) > 0 else None,
    }
    messages = req.get("messages") or [{"role":"user", "content": req["prompt"]}]

    if stream:
        headers = {
            "Content-Type":  "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops nginx buffering, if any
        }
        return Response(stream_with_context(sse_llama_stream(messages, sampling)),
                            headers=headers)

    # ── non-stream mode ────────────────────────────────────────────────────
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=False,
        **sampling)
    full = resp.choices[0].message.content
    return jsonify({"content": full, "stop_reason": resp.choices[0].finish_reason})


# ---------------------------------------------------------------------------
# extra endpoints Unity probes
@app.route("/template", methods=["POST"])
def template():
    # tell Unity to use plain ChatML (`<|im_start|>role\ncontent`)
    return jsonify({"template": "chatml"})

@app.route("/tokenize", methods=["POST"])
def tokenize():
    txt = request.get_json(force=True).get("content", "")
    # Unity only needs the length; fake token IDs are fine
    return jsonify({"tokens": list(range(len(txt.split()))) })

@app.route("/detokenize", methods=["POST"])
def detokenize():
    toks = request.get_json(force=True).get("tokens", [])
    return jsonify({"content": " ".join(str(t) for t in toks)})

@app.route("/embeddings", methods=["POST"])
def embeddings():
    return jsonify({"embedding": [0.0] * 768})

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 11434)))