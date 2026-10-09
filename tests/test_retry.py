"""with_retry: backoff, jitter, Retry-After, cap, attempts; no real sleeping."""
import pytest

import tools
from tools import RetryableError, with_retry


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.time, "sleep", calls.append)
    return calls


def flaky(failures, error=lambda: RetryableError("busy")):
    """A function that raises `failures` times, then returns "ok"."""
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise error()
        return "ok"

    fn.state = state
    return fn


def test_success_first_try_does_not_sleep(sleeps):
    assert with_retry(lambda: "ok") == "ok"
    assert sleeps == []


def test_retries_until_success(sleeps):
    fn = flaky(2)
    assert with_retry(fn, attempts=5) == "ok"
    assert fn.state["calls"] == 3
    assert len(sleeps) == 2


def test_gives_up_after_attempts_without_sleeping_after_last(sleeps):
    fn = flaky(99)
    with pytest.raises(RetryableError):
        with_retry(fn, attempts=4)
    assert fn.state["calls"] == 4
    assert len(sleeps) == 3


def test_non_retryable_error_is_not_retried(sleeps):
    fn = flaky(1, error=lambda: ValueError("bug"))
    with pytest.raises(ValueError):
        with_retry(fn, attempts=5)
    assert fn.state["calls"] == 1
    assert sleeps == []


def test_exponential_backoff_with_jitter_and_cap(sleeps, monkeypatch):
    monkeypatch.setattr(tools.random, "uniform", lambda a, b: b)   # maximum jitter
    with pytest.raises(RetryableError):
        with_retry(flaky(99), attempts=6, base=1.0, cap=10.0)
    # base*2**attempt plus up to the same amount of jitter, never above cap
    assert sleeps == [2.0, 4.0, 8.0, 10.0, 10.0]


def test_jitter_is_random(sleeps, monkeypatch):
    monkeypatch.setattr(tools.random, "uniform", lambda a, b: 0.0)  # no jitter
    with pytest.raises(RetryableError):
        with_retry(flaky(99), attempts=4, base=1.0, cap=100.0)
    assert sleeps == [1.0, 2.0, 4.0]


def test_retry_after_is_used_and_capped(sleeps):
    with pytest.raises(RetryableError):
        with_retry(flaky(99, error=lambda: RetryableError("429", retry_after=7)), attempts=2, cap=30.0)
    assert sleeps == [7.0]
    sleeps.clear()
    with pytest.raises(RetryableError):
        with_retry(flaky(99, error=lambda: RetryableError("429", retry_after=500)), attempts=2, cap=30.0)
    assert sleeps == [30.0]
