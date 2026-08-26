"""
app.py

Interactive CLI for the Intelligent Sales Lead Agent.

Usage:
    python app.py                 # interactive prompts
    python app.py --example qualified    # run examples/qualified_lead.json
    python app.py --example unqualified  # run examples/unqualified_lead.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph import build_graph
from utils import logger

BANNER = "=" * 60


def _prompt(label: str, required: bool = True) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value or not required:
            return value
        print(f"  '{label}' is required.")


def collect_lead_interactively() -> dict:
    print(BANNER)
    print("INTELLIGENT SALES LEAD AGENT")
    print(BANNER)
    print()
    lead = {
        "lead_name": _prompt("Lead name"),
        "company": _prompt("Company"),
        "role": _prompt("Role"),
        "industry": _prompt("Industry", required=False),
        "company_size": _prompt("Company size", required=False),
        "need": _prompt("Business need"),
        "budget": _prompt("Budget", required=False),
        "urgency": _prompt("Urgency", required=False),
    }
    return {k: v for k, v in lead.items() if v != ""}


def load_example(name: str) -> dict:
    path = Path(__file__).parent / "examples" / f"{name}_lead.json"
    return json.loads(path.read_text())


def render_result(final_state: dict) -> None:
    print()
    print(BANNER)
    print("LEAD ANALYSIS")
    print(BANNER)
    print()

    if final_state.get("errors") and not final_state.get("qualification_status"):
        print("The lead could not be processed:")
        for err in final_state["errors"]:
            print(f"  - {err}")
        print()
        return

    status = final_state.get("qualification_status", "unknown").upper()
    print(f"Qualification Score: {final_state.get('qualification_score', 'N/A')}")
    print(f"Status: {status}")
    print()
    print("Reason:")
    print(final_state.get("qualification_reason", "N/A"))
    print()
    print("Next Action:")
    print(final_state.get("next_action", "N/A"))
    print()

    if final_state.get("qualification_status") == "qualified":
        print("Research Brief:")
        print(final_state.get("research", "N/A"))
        print()
        print("Generated Outreach:")
        print(final_state.get("outreach_message", "N/A"))
    else:
        print("Nurture Message:")
        print(final_state.get("nurture_message", "N/A"))

    if final_state.get("errors"):
        print()
        print("Non-fatal errors recorded during execution:")
        for err in final_state["errors"]:
            print(f"  - {err}")
    print()


def run(lead: dict) -> dict:
    graph = build_graph()
    logger.info("Workflow started")
    final_state = graph.invoke(lead)
    logger.info("Workflow finished")
    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent Sales Lead Agent")
    parser.add_argument(
        "--example",
        choices=["qualified", "unqualified"],
        help="Run a bundled example lead instead of prompting interactively.",
    )
    args = parser.parse_args()

    if args.example:
        lead = load_example(args.example)
    else:
        lead = collect_lead_interactively()

    print()
    print("Workflow started...")
    final_state = run(lead)
    render_result(final_state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
