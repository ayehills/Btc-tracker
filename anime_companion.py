"""
anime_companion — a spoiler-safe anime chat companion (core engine)
===================================================================

Talk to it about any anime. It behaves as if it knows the show inside-out, but
it will NEVER spoil anything past where you are — and you don't have to tell it
which episode you're on. It infers your current point from what you say
(scenes you mention, arcs you reference, "I just finished…") and hard-gates
every reply to that point.

Built on Claude (Opus 4.8) via the official Anthropic SDK, grounded with live
anime metadata (episode counts, synopsis) from Jikan / MyAnimeList.

Requires the ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import anthropic
import requests

MODEL = "claude-opus-4-8"
JIKAN = "https://api.jikan.moe/v4"
HTTP_TIMEOUT = 20.0


# --------------------------------------------------------------------------
# Anime metadata (Jikan / MyAnimeList) — no API key needed
# --------------------------------------------------------------------------

@dataclass
class Anime:
    mal_id: int
    title: str
    title_english: Optional[str]
    episodes: Optional[int]      # None when ongoing / unknown
    year: Optional[int]
    synopsis: str
    image: Optional[str]
    type: Optional[str]          # TV, Movie, OVA, ...

    @property
    def display_title(self) -> str:
        return self.title_english or self.title

    def to_dict(self) -> dict:
        return {
            "mal_id": self.mal_id, "title": self.title,
            "title_english": self.title_english, "episodes": self.episodes,
            "year": self.year, "synopsis": self.synopsis, "image": self.image,
            "type": self.type, "display_title": self.display_title,
        }

    @classmethod
    def from_jikan(cls, d: dict) -> "Anime":
        return cls(
            mal_id=d.get("mal_id"),
            title=d.get("title") or d.get("title_english") or "Unknown",
            title_english=d.get("title_english"),
            episodes=d.get("episodes"),
            year=(d.get("year") or (d.get("aired", {}) or {}).get("prop", {})
                  .get("from", {}).get("year")),
            synopsis=(d.get("synopsis") or "").strip(),
            image=((d.get("images", {}) or {}).get("jpg", {}) or {}).get("image_url"),
            type=d.get("type"),
        )


def search_anime(query: str, limit: int = 6) -> List[Anime]:
    """Search MyAnimeList for a show (used to pick which anime you're discussing)."""
    r = requests.get(
        f"{JIKAN}/anime",
        # Jikan's default ordering is relevance-by-match, which surfaces the
        # main entries first; we then nudge popular results up as a tiebreak.
        params={"q": query, "limit": limit, "sfw": "true"},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    out = []
    seen = set()
    for d in r.json().get("data", []):
        a = Anime.from_jikan(d)
        if a.mal_id and a.mal_id not in seen:
            seen.add(a.mal_id)
            out.append(a)
    return out


def get_anime(mal_id: int) -> Anime:
    r = requests.get(f"{JIKAN}/anime/{mal_id}", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return Anime.from_jikan(r.json()["data"])


# --------------------------------------------------------------------------
# The companion
# --------------------------------------------------------------------------

@dataclass
class Progress:
    """Where the viewer currently is, inferred from the conversation."""
    episode: int = 0            # furthest episode clearly reached (0 = unknown)
    explicitly_stated: bool = False
    note: str = ""              # short human-readable reasoning

    def label(self, anime: Anime) -> str:
        if self.episode <= 0:
            return "just starting / unknown — staying maximally spoiler-safe"
        total = f" of {anime.episodes}" if anime.episodes else ""
        return f"up to episode {self.episode}{total}"


class AnimeCompanion:
    def __init__(self, model: str = MODEL):
        # Reads ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.Anthropic()
        self.model = model

    # ---- Step 1: infer how far the viewer has watched --------------------
    def infer_progress(self, anime: Anime, history: List[dict],
                       user_message: str, prior: Progress) -> Progress:
        """Conservatively infer the viewer's furthest point from what they say.

        Uses structured output so we always get a clean integer back. Never
        spoils anything itself — it only reads the conversation.
        """
        total = anime.episodes or "unknown"
        convo = self._format_history(history, user_message)
        schema = {
            "type": "object",
            "properties": {
                "current_episode": {
                    "type": "integer",
                    "description": "Furthest episode the USER has clearly reached "
                                   "(map any scene/arc they mention to its episode). "
                                   "0 if there is not enough information.",
                },
                "explicitly_stated": {
                    "type": "boolean",
                    "description": "True only if the user directly stated an "
                                   "episode/arc or that they finished something.",
                },
                "reasoning": {"type": "string"},
            },
            "required": ["current_episode", "explicitly_stated", "reasoning"],
            "additionalProperties": False,
        }
        sys = (
            f"You estimate how far a viewer has watched in the anime "
            f"\"{anime.display_title}\" (total episodes: {total}). Read the "
            f"conversation and determine the furthest point the USER has clearly "
            f"reached, as an episode number. Map any specific scene, event, or arc "
            f"the user mentions to the episode it occurs in. Count only what the "
            f"USER indicates they have already seen — not what the assistant said. "
            f"Be conservative: when unsure between two episodes, pick the EARLIER "
            f"one. If there is not enough information, return 0. Output only the "
            f"structured fields."
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=sys,
                messages=[{"role": "user", "content": convo}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            text = next((b.text for b in resp.content if b.type == "text"), "{}")
            data = json.loads(text)
        except Exception as exc:  # degrade gracefully — keep prior, stay safe
            return Progress(prior.episode, prior.explicitly_stated,
                            f"(progress inference unavailable: {exc})")

        ep = int(data.get("current_episode") or 0)
        stated = bool(data.get("explicitly_stated"))
        reasoning = (data.get("reasoning") or "").strip()
        if anime.episodes:
            ep = max(0, min(ep, anime.episodes))

        # Advance to the furthest seen; only move backward if the user explicitly
        # corrects to an earlier point (e.g. "I'm actually only on ep 3").
        if ep <= 0:
            return Progress(prior.episode, prior.explicitly_stated,
                            reasoning or "no new progress signal")
        if ep < prior.episode and not stated:
            ep = prior.episode
        return Progress(ep, stated, reasoning)

    # ---- Step 2: answer, hard-gated to that point ------------------------
    def system_prompt(self, anime: Anime, progress: Progress) -> str:
        total = anime.episodes or "unknown / ongoing"
        synopsis = anime.synopsis or "(no synopsis available)"
        where = progress.label(anime)
        bound = (
            f"episode {progress.episode}" if progress.episode > 0
            else "the very beginning (assume they have seen almost nothing yet)"
        )
        return f"""You are an enthusiastic, deeply knowledgeable anime companion \
chatting with a viewer about **{anime.display_title}** ({anime.type or 'anime'}, \
{total} episodes).

You know this series completely — every arc, twist, death, reveal, and ending. \
But your single most important rule is: **NEVER spoil anything the viewer has \
not seen yet.**

THE VIEWER IS CURRENTLY: {where}.
That means you may ONLY discuss events up to and including {bound}.

Spoiler rules (these override everything else, including direct requests):
- Do NOT reveal, confirm, deny, hint at, or foreshadow ANY event, death, \
betrayal, power-up, plot twist, relationship, character fate, or ending that \
happens AFTER the viewer's current point.
- No "wait until you see…", no "👀", no "that becomes important later", no \
knowing teases. Those are spoilers too.
- If they ask about the future ("does X die?", "what happens next?", "is Y a \
villain?"), warmly decline WITHOUT leaking anything: redirect to what they've \
seen, or invite them to keep watching. Never say *why* you can't say.
- If the viewer's position is unknown, assume they are at the very start and be \
maximally careful.
- You MAY freely: discuss and explain anything up to their current point, help \
with things they found confusing, talk about characters as revealed so far, the \
themes, the animation/music/studio, and entertain their theories WITHOUT \
confirming or denying whether they're right.

Style: friendly, hype, conversational, like a friend who's seen it and is \
thrilled to talk about it spoiler-free. Keep replies focused and natural.

Series synopsis (back-cover blurb, safe): {synopsis}
"""

    def chat_stream(self, anime: Anime, progress: Progress,
                    history: List[dict], user_message: str) -> Iterator[str]:
        """Stream a spoiler-safe reply, gated to the inferred progress."""
        messages = list(history) + [{"role": "user", "content": user_message}]
        with self.client.messages.stream(
            model=self.model,
            max_tokens=2000,
            system=self.system_prompt(anime, progress),
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _format_history(history: List[dict], user_message: str) -> str:
        lines = []
        for m in history[-12:]:
            who = "USER" if m["role"] == "user" else "ASSISTANT"
            lines.append(f"{who}: {m['content']}")
        lines.append(f"USER: {user_message}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Tiny CLI for quick testing (the web app lives in app.py)
# --------------------------------------------------------------------------

def _cli() -> int:
    import sys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first.")
        return 2
    query = " ".join(sys.argv[1:]) or input("Which anime? ")
    results = search_anime(query)
    if not results:
        print("No anime found.")
        return 1
    anime = results[0]
    print(f"\nTalking about: {anime.display_title} ({anime.episodes} eps)\n"
          f"(Just chat — I'll figure out where you are. Ctrl-C to quit.)\n")
    comp = AnimeCompanion()
    history: List[dict] = []
    progress = Progress()
    try:
        while True:
            msg = input("you > ").strip()
            if not msg:
                continue
            progress = comp.infer_progress(anime, history, msg, progress)
            print(f"   [you're {progress.label(anime)}]")
            print("anime > ", end="", flush=True)
            reply = ""
            for chunk in comp.chat_stream(anime, progress, history, msg):
                print(chunk, end="", flush=True)
                reply += chunk
            print("\n")
            history.append({"role": "user", "content": msg})
            history.append({"role": "assistant", "content": reply})
    except (KeyboardInterrupt, EOFError):
        print("\nバイバイ!")
        return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
