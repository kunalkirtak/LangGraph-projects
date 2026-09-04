# LangGraph Projects

A hands-on portfolio of three progressively advanced agentic workflows, each built with **[LangGraph](https://langchain-ai.github.io/langgraph/)** and **Google Gemini**. Every project treats orchestration as an explicit graph — typed state, pure node functions, and graph-owned control flow — rather than a chain of nested function calls, and each one builds directly on the pattern introduced by the last.

Every project runs **locally or in Google Colab** with no Docker and no paid infrastructure beyond a free Gemini API key. Each ships with its own runnable CLI (`app.py`), a fully mocked/offline `pytest` suite, and a Jupyter notebook walkthrough.

## Why this exists

A single Python function that calls an LLM a few times in a row looks simple, but it doesn't scale: error handling gets duplicated everywhere, there's no single source of truth for what data exists at each step, and adding branching, parallelism, retries, or human review later means a rewrite rather than an extension. LangGraph turns a workflow into an explicit graph over shared, typed state — this repo is a deliberate, incremental tour of that idea.

## Projects

| # | Project | Pattern | What it adds over the previous project |
|---|---------|---------|-------------------------------------------|
| 1 | [`01_research_intelligence`](./01_research_intelligence) | **Sequential workflow** | A topic goes in; three graph nodes (research → analysis → report) build on shared state to produce a structured Markdown report. Establishes state, nodes, edges, retries, and error handling. |
| 2 | [`02-sales-agent`](./02-sales-agent) | **Conditional workflow** | A sales lead is normalized, then an LLM-assisted qualification step produces a score, and `add_conditional_edges` routes the lead down genuinely different paths (research + outreach vs. nurture) based on that state. |
| 3 | [`03-human-approval-workflow`](./03-human-approval-workflow) | **Human-in-the-loop workflow** | An AI risk assessment decides whether a request needs a human decision. High-risk requests **interrupt** the graph, checkpoint state, and **resume** exactly where they paused once a human approves or rejects. |

```text
PROJECT 1                    PROJECT 2                       PROJECT 3
Sequential                   Conditional                     Human-in-the-loop
Research → Analysis → Report Lead → Qualify → Route           Analyze → Risk → Approve/Reject → Execute
                              ├── Qualified → Research → Outreach   (pauses + resumes on real state)
                              └── Unqualified → Nurture
```

Each project folder is self-contained and has its own detailed README covering architecture diagrams, state models, LangGraph concepts demonstrated, setup, environment variables, tests, limitations, and future improvements.

## Common architecture

Every graph in this repo shares the same core shape:

- **State** — a single `TypedDict` (or Pydantic model) that every node reads from and writes to. Nodes never call each other directly; they only read/write shared state.
- **Nodes** — pure functions of `(state) -> partial state update`, each a factory closing over an injectable LLM client so it never imports `langchain-google-genai` directly.
- **Edges / conditional edges** — transitions owned by the compiled graph (`add_edge`, `add_conditional_edges`), not by `if/else` logic wrapped around the graph.
- **Retries** — a bounded, exponential-backoff retry around each LLM call, plus a graph-level `RetryPolicy` as a safety net.
- **Error handling** — failures are captured into `state["errors"]` and surfaced as a status (e.g. `*_failed`), never raised past `graph.invoke()`; downstream nodes detect the status and skip rather than operate on missing data.
- **Testing** — every graph is exercised end-to-end against a fake/deterministic LLM client (`tests/fakes.py` / `tests/conftest.py`), so the full test suite runs offline with no API key and no network calls.

## Tech stack

- Python 3.10+
- [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph`, conditional edges, checkpointing/`interrupt()` in project 3)
- [LangChain Core](https://github.com/langchain-ai/langchain) + `langchain-google-genai`
- Google **Gemini** as the underlying LLM
- Pydantic for structured LLM output (projects 2 and 3)
- `python-dotenv` for environment configuration
- `pytest` with fake LLM clients for fully offline test suites
- Jupyter notebooks for an end-to-end runnable walkthrough of each project

## Getting started

Each project can be run independently:

```bash
cd 0X-<project-name>
pip install -r requirements.txt

cp .env.example .env
# fill in GOOGLE_API_KEY (and optionally GEMINI_MODEL)
```

Get a free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). In Colab, prefer the built-in Secrets manager over pasting a key into a cell:

```python
from google.colab import userdata
import os
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
```

Then, from inside a project folder:

```bash
python app.py      # run the CLI
pytest             # run the offline test suite (no API key required)
```

## Repository structure

```text
LangGraph-projects/
├── 01_research_intelligence/     # Sequential graph: research → analysis → report
├── 02-sales-agent/                # Conditional graph: lead qualification + routing
├── 03-human-approval-workflow/    # Interrupt/resume graph: risk-gated human approval
├── LICENSE
└── README.md
```

## What this portfolio demonstrates

- Modeling multi-step LLM workflows as explicit graphs — shared typed state, pure node functions, and graph-owned transitions — instead of nested function calls.
- Conditional routing driven by state the graph itself produced (`add_conditional_edges`), including compound routing logic with multiple decision points.
- Human-in-the-loop control flow: pausing a graph mid-execution with `interrupt()`, checkpointing state, and resuming from exactly where it left off once a human responds.
- Isolating LLM calls behind a minimal client interface so every graph can be tested end-to-end (real state merging, real retries, real routing) against a fake model — no network access needed in CI.
- Separating concerns cleanly across a project: state schema, prompts, node logic, graph wiring, and the CLI entry point each live in their own file.

These are portfolio-grade demonstrations of LangGraph orchestration fundamentals, not deployed production systems — none of them include a production database, an auth layer, or hosted infrastructure.

## License

Released under the [MIT License](./LICENSE).
