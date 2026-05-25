"""Smoke tests for UI non-blocking guardrails."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402


class FakeStreamlit:
    """Minimal Streamlit stub for render_pdf_viewer's unloaded path."""

    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.messages: list[str] = []

    def warning(self, message: str) -> None:
        self.messages.append(message)

    def caption(self, message: str) -> None:
        self.messages.append(message)

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def info(self, message: str) -> None:
        self.messages.append(message)


def main() -> None:
    test_workspace_reload_decision()
    test_pdf_viewer_is_lazy_by_default()
    test_queue_lane_states_show_running_and_waiting_content()
    test_index_state_prefers_fresh_database_status()
    test_upload_pipeline_enqueues_parse_and_waiting_index()
    print("ui nonblocking guard tests passed")


def test_workspace_reload_decision() -> None:
    processed = {
        "saved_file": {"paper_id": "paper-1"},
        "parse_status": "running",
        "chunks": [],
    }
    running_paper = {
        "paper_id": "paper-1",
        "updated_at": "2026-05-24 10:00:00",
        "parse_status": "running",
        "index_status": "queued",
        "translation_status": "not_started",
    }
    assert not app.should_reload_processed_pdf(processed, running_paper)

    succeeded_paper = dict(running_paper, parse_status="succeeded", updated_at="2026-05-24 10:01:00")
    assert app.should_reload_processed_pdf(processed, succeeded_paper)

    loaded = {
        "saved_file": {"paper_id": "paper-1"},
        "parse_status": "succeeded",
        "chunks": [{"chunk_id": "paper-1:0000", "text": "ready"}],
    }
    index_changed = dict(succeeded_paper, index_status="running", updated_at="2026-05-24 10:02:00")
    assert not app.should_reload_processed_pdf(loaded, index_changed)
    assert app.paper_status_signature(succeeded_paper) != app.paper_status_signature(index_changed)


def test_pdf_viewer_is_lazy_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = Path(tmp_dir) / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

        fake_st = FakeStreamlit()
        original_st = app.st
        original_read_bytes = Path.read_bytes

        def fail_if_read(path: Path) -> bytes:
            raise AssertionError(f"PDF bytes were read before explicit load: {path}")

        try:
            app.st = fake_st
            Path.read_bytes = fail_if_read  # type: ignore[method-assign]
            app.render_pdf_viewer(str(pdf_path))
        finally:
            Path.read_bytes = original_read_bytes  # type: ignore[method-assign]
            app.st = original_st

        assert any("点击后再读取" in message for message in fake_st.messages)


def test_queue_lane_states_show_running_and_waiting_content() -> None:
    running_job = {
        "job_id": 10,
        "job_type": "parse",
        "status": "running",
        "file_name": "running-paper.pdf",
    }
    queued_job = {
        "job_id": 11,
        "job_type": "parse",
        "status": "queued",
        "file_name": "queued-paper.pdf",
    }
    blocked_job = {
        "job_id": 12,
        "job_type": "index",
        "status": "queued",
        "file_name": "waiting-index.pdf",
        "paper_id": "paper-12",
        "paper_parse_status": "running",
    }

    running_html = app.render_queue_lane("解析队列", running_job)
    queued_html = app.render_queue_lane("解析队列", queued_job)
    blocked_html = app.render_queue_lane("索引队列", blocked_job)
    row_html = app.render_queue_rows([blocked_job], 1)

    assert "running-paper.pdf" in running_html
    assert "运行中" in running_html
    assert "queued-paper.pdf" in queued_html
    assert "排队中" in queued_html
    assert "waiting-index.pdf" in blocked_html
    assert "等待解析完成" in blocked_html
    assert "pm-queue-remove" in row_html
    assert "pm_cancel_queue_job=12" in row_html


def test_index_state_prefers_fresh_database_status() -> None:
    fake_st = FakeStreamlit()
    fake_st.session_state["index_state_paper-db"] = {"vector": "排队中", "bm25": "排队中"}
    original_st = app.st
    original_current_user_id = app.current_user_id
    original_get_accessible_paper = app.get_accessible_paper

    try:
        app.st = fake_st
        app.current_user_id = lambda: 7
        app.get_accessible_paper = lambda paper_id, user_id: {
            "paper_id": paper_id,
            "index_status": "succeeded",
        }
        state = app.local_index_state("paper-db")
    finally:
        app.st = original_st
        app.current_user_id = original_current_user_id
        app.get_accessible_paper = original_get_accessible_paper

    assert state == {"vector": "已构建", "bm25": "已构建"}


def test_upload_pipeline_enqueues_parse_and_waiting_index() -> None:
    paper = {
        "paper_id": "paper-auto",
        "team_id": 3,
        "project_id": 5,
        "save_path": "data/uploads/paper-auto.pdf",
        "parse_status": "not_started",
        "index_status": "unknown",
    }
    jobs: list[dict[str, Any]] = []
    status_updates: list[tuple[str, dict[str, Any]]] = []

    original_st = app.st
    original_current_user_id = app.current_user_id
    original_latest_job_for_paper = app.latest_job_for_paper
    original_enqueue_job = app.enqueue_job
    original_update_paper_status = app.update_paper_status

    def fake_latest_job_for_paper(team_id: int, paper_id: str, job_type: str | None = None) -> dict[str, Any] | None:
        return None

    def fake_enqueue_job(
        job_type: str,
        user_id: int,
        team_id: int,
        project_id: int | None = None,
        paper_id: str | None = None,
        payload: dict[str, Any] | None = None,
        max_attempts: int = 3,
    ) -> int:
        job_id = 100 + len(jobs) + 1
        jobs.append(
            {
                "job_id": job_id,
                "job_type": job_type,
                "user_id": user_id,
                "team_id": team_id,
                "project_id": project_id,
                "paper_id": paper_id,
                "payload": payload or {},
                "max_attempts": max_attempts,
            }
        )
        return job_id

    def fake_update_paper_status(paper_id: str, **fields: Any) -> None:
        status_updates.append((paper_id, fields))

    try:
        app.st = FakeStreamlit()
        app.current_user_id = lambda: 9
        app.latest_job_for_paper = fake_latest_job_for_paper
        app.enqueue_job = fake_enqueue_job
        app.update_paper_status = fake_update_paper_status

        pipeline = app.enqueue_upload_processing_pipeline(paper)
    finally:
        app.st = original_st
        app.current_user_id = original_current_user_id
        app.latest_job_for_paper = original_latest_job_for_paper
        app.enqueue_job = original_enqueue_job
        app.update_paper_status = original_update_paper_status

    assert pipeline["parse_job_id"] == 101
    assert pipeline["index_job_id"] == 102
    assert [job["job_type"] for job in jobs] == ["parse", "index"]
    assert jobs[0]["payload"]["include_images"] is False
    assert jobs[0]["payload"]["auto_index"] is False
    assert jobs[1]["payload"]["waiting_for_parse"] is True
    assert ("paper-auto", {"parse_status": "queued", "index_status": "queued"}) in status_updates
    assert ("paper-auto", {"index_status": "queued"}) in status_updates
    assert app.upload_processing_message(False, pipeline).startswith("已保存到论文库")


if __name__ == "__main__":
    main()
