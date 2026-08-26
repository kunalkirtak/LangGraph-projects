from utils import mark_node_complete, new_execution_metadata, save_report


def test_save_report_writes_expected_content(tmp_path):
    report = "# Title\n\nSome report body.\n"
    target = tmp_path / "out" / "research_report.md"

    result_path = save_report(report, filename=str(target))

    assert result_path == target
    assert target.exists()
    assert target.read_text(encoding="utf-8") == report


def test_save_report_default_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = "# Report\n"

    result_path = save_report(report)

    assert result_path.name == "research_report.md"
    assert result_path.exists()


def test_new_execution_metadata_has_expected_keys():
    metadata = new_execution_metadata()

    assert "execution_id" in metadata
    assert "start_time" in metadata
    assert metadata["completed_nodes"] == []


def test_new_execution_metadata_ids_are_unique():
    first = new_execution_metadata()
    second = new_execution_metadata()

    assert first["execution_id"] != second["execution_id"]


def test_mark_node_complete_appends_without_mutating_original():
    metadata = new_execution_metadata()

    updated = mark_node_complete(metadata, "research")

    assert updated["completed_nodes"] == ["research"]
    # Original metadata dict must remain untouched (pure function).
    assert metadata["completed_nodes"] == []

    updated_again = mark_node_complete(updated, "analysis")
    assert updated_again["completed_nodes"] == ["research", "analysis"]
