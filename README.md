# AEGIS

AEGIS is a personal AI assistant that runs entirely on your own hardware — no cloud account, no subscription, none of your data leaving your network. Think Jarvis, but local-first: it keeps your tasks, remembers what matters to you, watches your homelab, tracks your spending, and plans your groceries, all powered by models running on your own machines.

It's built on a simple bet: the useful personal assistant of the future shouldn't require handing your life to someone else's servers. The hard problems it works through are the ones that bet demands — bounded agency (the model proposes, policy disposes), explicit authorization for anything consequential, recovery after restarts, and independence from any single model vendor.

> **Status: alpha.** Under active development — not a finished product. See [Status](#status).

[Demo video/GIF coming]

## What it does today

- **Tasks, chores, groceries, pantry** — natural-language management ("add milk to the grocery list") with a Today dashboard showing what needs attention.
- **Homelab monitoring** — discovers hosts on your network, tracks service health and inventory, surfaces problems before you go looking.
- **Finance tracking** — CSV import with categorized spending views and budget checks, all stored locally in PostgreSQL.
- **Local chat + web UI** — ask questions over your own documents, calendar reads, weather, and live research, answered by Ollama models on your hardware. Includes an interactive Constellation map of everything it can do.
- **Safety by design** — every action the model proposes passes explicit authorization and independent verification; nothing consequential happens without approval, and blocked work retries within bounded budgets instead of dying silently. See [SECURITY_MODEL.md](SECURITY_MODEL.md).

## Status

**Works**
- Task / chore / grocery / pantry management via CLI and browser UI
- Homelab discovery, inventory, and health monitoring
- Finance CSV import, spending views, budget checks
- Local web UI with interactive Constellation capability map
- Document search and summaries, calendar reads, weather, live research
- Recovery-first runtime: interrupted objectives resume safely instead of failing silently

**In progress**
- The road to daily-driver reliability — the original goal, not reached yet
- Pack-first capability routing experiments (retrieval-assisted routing shows promise in evals; not yet promotable)
- Conversation-first chat polish
- Re-evaluating the model floor: built and tested around 8B local models, and part of the current ceiling looks like model capability rather than architecture

**Future**
- Mobile and voice clients
- Live smart-home control (Home Assistant integration; live acceptance pending)
- Outbound notifications (mobile/chat delivery not yet wired)
- Optional cloud sync lane (local-first stays the default)

## Quickstart

You need **Python 3.11+**, **PostgreSQL**, and [Ollama](https://ollama.ai) running with a model installed. Everything else runs locally.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[live]'
./scripts/aegis --init        # creates config with private permissions; refuses to overwrite
```

Edit the generated `.env`: point `AEGIS_OLLAMA_URL` at your Ollama (usually `http://127.0.0.1:11434`) and fill in the database settings. Then:

```sh
./scripts/aegis --check                 # readiness check; explains how to fix what's missing
./scripts/aegis --web                  # start the local browser UI (loopback only)
./scripts/aegis --once "Show my tasks." # one-shot request, no UI needed
```

If a service is unavailable, `--web` still opens a diagnostics shell that tells you how to fix it. Nothing is exposed beyond loopback unless you choose to expose it.

## License

Source-available for review; all rights reserved — not open-source licensed.

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — system boundaries and design invariants
- [SECURITY_MODEL.md](SECURITY_MODEL.md) — authorization, verification, trust boundaries
- [VISION.md](VISION.md) — the long-term product vision
- [CORE_CONCEPTS.md](CORE_CONCEPTS.md) — cognition, truth, and security invariants
- [PACK_SPEC.md](PACK_SPEC.md) — modular Pack contracts
- [MODEL_STRATEGY.md](MODEL_STRATEGY.md) — model roles and evaluation
- [docs/OWNER_DOGFOOD.md](docs/OWNER_DOGFOOD.md) — installed-runtime dogfood protocol
