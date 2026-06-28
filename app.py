"""
Anime companion — local web app
===============================

A browser chat you can talk to about any anime. It infers how far you've watched
from what you say and never spoils anything past that point.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install -r requirements.txt
    python app.py
    # open http://localhost:5000

Progress per show is remembered in data/progress.json.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from anime_companion import Anime, AnimeCompanion, Progress, search_anime

app = Flask(__name__)
companion = AnimeCompanion()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = DATA_DIR / "progress.json"
_lock = threading.Lock()


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_progress(store: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(store, indent=2))


def _get_progress(mal_id: int) -> Progress:
    rec = _load_progress().get(str(mal_id), {})
    return Progress(rec.get("episode", 0), rec.get("explicitly_stated", False),
                    rec.get("note", ""))


def _set_progress(mal_id: int, p: Progress) -> None:
    with _lock:
        store = _load_progress()
        store[str(mal_id)] = {"episode": p.episode,
                              "explicitly_stated": p.explicitly_stated, "note": p.note}
        _save_progress(store)


@app.route("/")
def index():
    return render_template("index.html", has_key=bool(os.environ.get("ANTHROPIC_API_KEY")))


@app.route("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"results": []})
    try:
        results = [a.to_dict() for a in search_anime(q)]
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"results": results})


@app.route("/api/progress")
def api_progress():
    mal_id = int(request.args.get("mal_id", 0))
    p = _get_progress(mal_id)
    return jsonify({"episode": p.episode, "explicitly_stated": p.explicitly_stated,
                    "note": p.note})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True)
    anime = _anime_from_payload(body["anime"])
    history = body.get("history", [])
    message = body["message"]

    # 1) Infer how far they've watched (locked in before answering).
    prior = _get_progress(anime.mal_id)
    progress = companion.infer_progress(anime, history, message, prior)
    _set_progress(anime.mal_id, progress)

    def gen():
        # First event: the inferred progress, so the UI can show "you're up to…".
        yield _sse({"type": "progress", "episode": progress.episode,
                    "total": anime.episodes, "label": progress.label(anime),
                    "note": progress.note})
        try:
            for chunk in companion.chat_stream(anime, progress, history, message):
                yield _sse({"type": "delta", "text": chunk})
        except Exception as exc:
            yield _sse({"type": "error", "text": f"({exc})"})
        yield _sse({"type": "done"})

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _anime_from_payload(d: dict) -> Anime:
    return Anime(
        mal_id=d.get("mal_id"), title=d.get("title") or d.get("display_title") or "Unknown",
        title_english=d.get("title_english"), episodes=d.get("episodes"),
        year=d.get("year"), synopsis=d.get("synopsis", ""), image=d.get("image"),
        type=d.get("type"),
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY is not set — the chat won't work until you set it.")
        print("   export ANTHROPIC_API_KEY=sk-ant-...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
