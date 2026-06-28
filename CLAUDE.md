# CLAUDE.md — project orientation

> A spoiler-safe **anime chat companion**. Talk to it about any anime; it knows
> the whole show but never spoils past where you are — and it infers your
> progress from what you say rather than asking for an episode number.

## Stack

- **Claude Opus 4.8** (`claude-opus-4-8`) via the official `anthropic` Python SDK.
- **Flask** web app, browser chat with SSE streaming.
- **Jikan / MyAnimeList** (`api.jikan.moe`) for anime metadata (no key).
- Requires `ANTHROPIC_API_KEY` in the environment.

## Files

| File | Purpose |
| --- | --- |
| `anime_companion.py` | Core engine: `search_anime`, `AnimeCompanion.infer_progress`, `.system_prompt`, `.chat_stream`. Plus a CLI (`python anime_companion.py "<anime>"`). |
| `app.py` | Flask routes: `/` page, `/api/search`, `/api/progress`, `/api/chat` (SSE stream). Progress persisted in `data/progress.json`. |
| `templates/index.html` | Chat UI: anime picker, streaming bubbles, live "where you are" banner. |

## The two-step spoiler logic (the heart of it)

1. **Infer progress** — `infer_progress()` makes a structured-output Claude call
   that reads the conversation and returns the furthest episode the *user* has
   clearly reached (0 if unknown). Conservative: advances only on clear signals,
   picks the earlier episode when unsure, only regresses on an explicit correction.
2. **Answer gated** — `system_prompt()` tells Claude the user's current episode
   and forbids revealing/hinting/foreshadowing anything beyond it; `chat_stream()`
   streams the reply.

## Conventions

- Default to `claude-opus-4-8`; use the official SDK (no raw HTTP).
- Read the API key from the environment — never hardcode or log it.
- Keep `anime_companion.py` usable without Flask (the CLI proves this).
- Spoiler-safety is the top priority — when changing prompts, preserve the
  "knows everything, reveals nothing past the user's point" contract.

## Ideas / next steps

- Map progress to arcs (not just episode numbers) for shows with named arcs.
- Support manga (Jikan `/manga`) with chapter-based progress.
- Let the app detect the anime from free text instead of requiring the picker.
- Optional: a quick "recap up to where I am" command.
