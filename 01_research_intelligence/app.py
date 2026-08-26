"""
CLI entrypoint for the AI Research Intelligence Pipeline.

Usage:
    python app.py
"""

from __future__ import annotations

from config import configure_logging
from graph import build_graph
from state import initial_state
from utils import new_execution_metadata, save_report

logger = configure_logging()


def run(topic: str) -> dict:
    """Execute the compiled graph for a given topic and return final state."""
    print("Workflow started")
    logger.info("Workflow started | topic=%s", topic)

    graph = build_graph()
    state = initial_state(topic)
    state["metadata"] = new_execution_metadata()

    final_state = graph.invoke(state)

    status = final_state.get("status", "unknown")
    if final_state.get("research"):
        print("Research completed")
    if final_state.get("analysis"):
        print("Analysis completed")
    if final_state.get("report"):
        print("Report generated")

    if final_state.get("errors"):
        print("\nCompleted with errors:")
        for err in final_state["errors"]:
            print(f"  - {err}")

    logger.info("Workflow finished | status=%s", status)
    return final_state


def main() -> None:
    topic = input("Enter research topic: ").strip()
    if not topic:
        print("No topic provided. Exiting.")
        return

    final_state = run(topic)

    report = final_state.get("report")
    if report:
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60 + "\n")

        save = input("Save report to research_report.md? [Y/n]: ").strip().lower()
        if save in ("", "y", "yes"):
            path = save_report(report)
            print(f"Report saved to {path.resolve()}")
    else:
        print("No report was generated. Check the errors above.")


if __name__ == "__main__":
    main()
