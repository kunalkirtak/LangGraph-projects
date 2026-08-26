from graph import build_graph
from state import initial_state
from tests.fakes import FakeLLM


def test_graph_compiles_with_expected_nodes():
    graph = build_graph(llm=FakeLLM())

    node_names = set(graph.get_graph().nodes.keys())

    # LangGraph always includes the virtual __start__/__end__ nodes
    # alongside the ones we registered.
    for expected in ("research", "analysis", "report"):
        assert expected in node_names


def test_graph_execution_path_is_sequential():
    graph = build_graph(llm=FakeLLM())
    edges = graph.get_graph().edges

    edge_pairs = {(edge.source, edge.target) for edge in edges}

    assert ("__start__", "research") in edge_pairs
    assert ("research", "analysis") in edge_pairs
    assert ("analysis", "report") in edge_pairs
    assert ("report", "__end__") in edge_pairs


def test_end_to_end_workflow_populates_full_state():
    llm = FakeLLM()
    graph = build_graph(llm=llm)
    state = initial_state("LLM inference optimization")

    final_state = graph.invoke(state)

    assert final_state["status"] == "report_complete"
    assert "Fake research synthesis" in final_state["research"]
    assert "Fake analysis body" in final_state["analysis"]
    assert final_state["report"].startswith("# Executive Summary")
    assert final_state["errors"] == []

    # Three distinct prompts were sent: research, analysis, report.
    assert len(llm.calls) == 3


def test_analysis_node_receives_research_output():
    """Demonstrates state propagation: analysis reads what research wrote."""
    llm = FakeLLM()
    graph = build_graph(llm=llm)
    state = initial_state("Vector databases")

    graph.invoke(state)

    research_prompt, analysis_prompt, report_prompt = llm.calls
    assert "Fake research synthesis body" in analysis_prompt
    assert "Fake analysis body" in report_prompt


def test_workflow_recovers_via_retry_on_transient_failure():
    """The research node fails once, then succeeds on retry."""
    llm = FakeLLM(fail_on={"Produce a structured research synthesis"}, fail_times=1)
    graph = build_graph(llm=llm)
    state = initial_state("Retry demonstration topic")

    final_state = graph.invoke(state)

    assert final_state["status"] == "report_complete"
    assert final_state["errors"] == []
    # First call raised and was retried, so more than one call was made
    # against the research prompt before it (and the pipeline) succeeded.
    research_calls = [c for c in llm.calls if "Produce a structured research synthesis" in c]
    assert len(research_calls) >= 2


def test_workflow_records_error_when_all_retries_exhausted():
    """If a node fails on every attempt, the error is captured in state,
    not raised past the graph boundary, and downstream nodes are skipped."""
    llm = FakeLLM(fail_on={"Produce a structured research synthesis"}, fail_times=999)
    graph = build_graph(llm=llm)
    state = initial_state("Always failing topic")

    final_state = graph.invoke(state)

    # The failure happened in research; analysis/report cascade to a
    # "skipped" status rather than silently producing empty content.
    assert final_state["status"] == "report_skipped"
    assert any("Research node failed" in e for e in final_state["errors"])
    assert final_state["report"] == ""
    assert final_state["research"] == ""
