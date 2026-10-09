"""slugify, save_outputs and the tools' no-network edge cases."""
import json

import pytest

import research
import tools
from agents import REPORT_PATH, SOURCES_PATH
from research import save_outputs, slugify

SOURCES = [{"n": 1, "id": "x", "url": "https://example.com", "title": "X", "date": "2025-01-01", "source": "web"}]


@pytest.mark.parametrize("topic, slug", [
    ("survey about world model", "survey-about-world-model"),
    ("../../x", "x"),
    ("", "topic"),
    ("   ", "topic"),
    ("!!!", "topic"),
    ("Khảo sát mô hình thế giới", "khao-sat-mo-hinh-the-gioi"),
])
def test_slugify(topic, slug):
    assert slugify(topic) == slug


def test_slugify_is_capped_at_60_chars():
    slug = slugify("word " * 40)
    assert len(slug) <= 60 and not slug.endswith("-")


def fake_download(files):
    return lambda backend, paths: {p: files.get(p) for p in paths}


@pytest.mark.parametrize("files", [
    {},                                                                        # nothing produced
    {REPORT_PATH: b"   ", SOURCES_PATH: json.dumps(SOURCES).encode()},         # empty report
    {REPORT_PATH: b"# Report [1]"},                                            # no sources.json
    {REPORT_PATH: b"# Report [1]", SOURCES_PATH: b"{not json"},                # invalid sources.json
    {REPORT_PATH: b"# Report [1]", SOURCES_PATH: b"[]"},                       # no sources
])
def test_failed_run_writes_nothing(tmp_path, monkeypatch, files):
    monkeypatch.setattr(research, "download", fake_download(files))
    with pytest.raises(RuntimeError):
        save_outputs(None, "t", [], 1.0, "m", reports_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_successful_run_writes_three_files(tmp_path, monkeypatch):
    files = {REPORT_PATH: b"# Report [1]", SOURCES_PATH: json.dumps(SOURCES).encode()}
    monkeypatch.setattr(research, "download", fake_download(files))
    path = save_outputs(None, "my topic", [], 12.34, "m", reports_dir=tmp_path)
    assert path == tmp_path / "my-topic.md"
    meta = json.loads((tmp_path / "my-topic.meta.json").read_text(encoding="utf-8"))
    assert meta["n_sources"] == 1 and meta["source_families"] == ["web"] and meta["elapsed_s"] == 12.3
    assert (tmp_path / "my-topic.sources.json").exists()


def test_empty_queries_return_no_results_without_network(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network called")
    monkeypatch.setattr(tools.httpx, "request", no_network)
    assert tools.arxiv_search.invoke({"query": "\"\" : AND OR"}) == "NO RESULTS"
    assert tools.hf_search_papers.invoke({"query": "  "}) == "NO RESULTS"
    assert tools.web_search.invoke({"query": ""}) == "NO RESULTS"


def test_tool_errors_become_strings_and_hide_the_exa_key(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret-key-123")
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)

    def broken(method, url, **kwargs):
        raise tools.httpx.ConnectError(f"cannot reach {url}")
    monkeypatch.setattr(tools.httpx, "request", broken)
    for result in (tools.web_search.invoke({"query": "x"}), tools.web_fetch.invoke({"url": "https://example.com"}),
                   tools.hf_search_papers.invoke({"query": "x"})):
        assert result.startswith("ERROR:")
        assert "secret-key-123" not in result
