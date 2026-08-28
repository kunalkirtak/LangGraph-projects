"""
Command-line entry point for the Enterprise Human Approval Workflow.

Run with:

    python app.py

Requires GOOGLE_API_KEY to be set in the environment (or a local .env file)
because analyze_request calls Gemini for the risk assessment.
"""

from __future__ import annotations

from langgraph.types import Command

from graph import build_graph
from state import new_state


def _print_header() -> None:
    print("=" * 60)
    print("ENTERPRISE HUMAN APPROVAL WORKFLOW")
    print("=" * 60)


def _prompt_float(label: str) -> float:
    while True:
        raw = input(label)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a numeric amount.")


def main() -> None:
    _print_header()

    requester = input("Requester: ")
    department = input("Department: ")
    action = input("Requested action: ")
    amount = _prompt_float("Amount: ")
    reason = input("Reason: ")

    graph = build_graph()
    thread_id = f"cli-{requester}-{action}"[:64]
    config = {"configurable": {"thread_id": thread_id}}

    state = new_state(requester, department, action, amount, reason)

    print("\nAnalyzing request...\n")
    result = graph.invoke(state, config=config)

    if result.get("errors") and not result.get("risk_score"):
        print("Request could not be processed:")
        for err in result["errors"]:
            print(f"  - {err}")
        return

    print(f"Risk score: {result.get('risk_score')}")
    print(f"Risk level: {result.get('risk_level')}\n")

    # A paused run is detected by inspecting the checkpointed graph state
    # rather than guessing from the returned dict - this is the reliable,
    # version-stable way to tell "the graph is genuinely interrupted" apart
    # from "the graph finished".
    snapshot = graph.get_state(config)
    pending_interrupts = snapshot.interrupts

    if pending_interrupts:
        payload = pending_interrupts[0].value if hasattr(pending_interrupts[0], "value") else pending_interrupts[0]
        print("Human approval required.\n")
        print("Approval request:")
        for key, value in payload.items():
            print(f"  {key}: {value}")
        print()

        decision = ""
        while decision not in ("approve", "reject"):
            decision = input("Decision [approve/reject]: ").strip().lower()

        final_state = graph.invoke(Command(resume=decision), config=config)

        print(f"\nHuman decision: {decision.upper()}\n")
        print("Execution:")
        print(final_state.get("execution_result"))
    else:
        print("No human approval required.\n")
        print("Execution:")
        print(result.get("execution_result"))


if __name__ == "__main__":
    main()
