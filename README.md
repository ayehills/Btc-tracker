# 🌸 Anime Companion — spoiler-safe

A local browser chat you can talk to about **any anime**. It acts like a friend
who has seen the whole show — but it will **never spoil anything past where you
are**, and you don’t have to tell it which episode you’re on. It **infers your
progress from what you say** (scenes you mention, arcs you reference, "I just
finished the Marineford fight") and hard-gates every reply to that point.

Built on **Claude (Opus 4.8)** via the official Anthropic SDK, grounded with
live anime metadata (episode counts, synopsis) from **Jikan / MyAnimeList**.

## How the no-spoiler logic works

1. **You pick a show** (search box) and just start chatting — no need to type an
   episode number.
2. On every message, the app makes a quick **progress-inference** call: it reads
   the conversation and conservatively estimates the furthest point *you* have
   clearly reached (mapping any scene/arc you mention to its episode). It only
   moves forward, and stays put when unsure.
3. Then it answers with a **spoiler-gated system prompt**: Claude is told your
   current episode and is forbidden from revealing, hinting at, or foreshadowing
   anything beyond it — including teasy "wait till you see…" lines.

If it doesn’t yet know where you are, it assumes the very beginning and stays
maximally careful.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # from console.anthropic.com
#   Windows: setx ANTHROPIC_API_KEY "sk-ant-..."
```

Requires **Python 3.9+**.

## Run

```bash
python app.py
# open http://localhost:5000
```

Or chat in the terminal:

```bash
python anime_companion.py "Jujutsu Kaisen"
```

## Files

| File | Purpose |
| --- | --- |
| `anime_companion.py` | Core engine — Claude client, progress inference, spoiler-safe prompt, Jikan data. Also a small CLI. |
| `app.py` | Flask web app (chat page + streaming `/api/chat` + `/api/search`). |
| `templates/index.html` | The chat UI (anime picker, streaming bubbles, live "where you are"). |
| `data/progress.json` | Remembers how far you are in each show (created at runtime, git-ignored). |

## Notes

- Your **Anthropic API key** is read from the environment and never stored or
  sent anywhere except Anthropic.
- Anime data comes from the free, key-less **Jikan** API (MyAnimeList).
- Spoiler-safety is strong but not infallible — it relies on the model and on
  your show being in its knowledge. If it ever slips, tell it and it will adjust.
