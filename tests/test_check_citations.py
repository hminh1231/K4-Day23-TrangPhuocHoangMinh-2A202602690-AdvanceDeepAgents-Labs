"""check_citations.check: the rules of GUIDE part 4, plus the url/family check."""
from check_citations import check

SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-01",
     "source": "arxiv"},
    {"n": 2, "id": "2502.00002", "url": "https://huggingface.co/papers/2502.00002", "title": "B", "date": "2025-02-01",
     "source": "hf-search"},
    {"n": 3, "id": "blog", "url": "https://example.com/post", "title": "C", "date": "2025-03-01", "source": "web"},
]
REFS = """## References

[1] A. arxiv. https://arxiv.org/abs/2501.00001
[2] B. hf-search. https://huggingface.co/papers/2502.00002
[3] C. web. https://example.com/post
"""


def report(body, refs=REFS):
    return f"# Title\n\n{body}\n\n{refs}"


def test_valid_report_is_ok():
    assert check(report("Claim one [1]. Claim two [2][3]."), SOURCES) == []


def test_empty_sources():
    assert check(report("Claim [1]."), []) != []


def test_missing_references_heading():
    problems = check("# Title\n\nClaim [1][2][3].", SOURCES)
    assert any("References" in p for p in problems)


def test_cited_number_not_in_sources():
    problems = check(report("Claim [1][2][3][4]."), SOURCES)
    assert any("[4]" in p for p in problems)


def test_uncited_source():
    problems = check(report("Claim [1][2]."), SOURCES)
    assert any("[3] never cited" in p for p in problems)


def test_reference_numbers_do_not_count_as_citations():
    problems = check(report("Claim [1]."), SOURCES)
    assert any("[2] never cited" in p for p in problems)


def test_grouped_citations_are_expanded():
    assert check(report("Claims [1, 2] and [2-3]."), SOURCES) == []


def test_code_and_markdown_links_are_not_citations():
    body = "Claim [1][2]. Code `x[3]` and a link [3](https://example.com/post)."
    problems = check(report(body), SOURCES)
    assert any("[3] never cited" in p for p in problems)


def test_merged_reference_line_is_rejected():
    refs = REFS.replace("[3] C. web. https://example.com/post",
                        "[3] C; D. https://example.com/post https://example.com/other")
    problems = check(report("Claim [1][2][3].", refs), SOURCES)
    assert any("exactly one URL" in p for p in problems)


def test_reference_url_must_match_sources():
    refs = REFS.replace("https://example.com/post", "https://example.com/wrong")
    problems = check(report("Claim [1][2][3].", refs), SOURCES)
    assert any("!= sources.json url" in p for p in problems)


def test_missing_and_duplicate_reference_lines():
    refs = REFS.replace("[3] C. web. https://example.com/post\n", "[2] B. https://huggingface.co/papers/2502.00002\n")
    problems = check(report("Claim [1][2][3].", refs), SOURCES)
    assert any("listed more than once" in p for p in problems)
    assert any("[3] has no line" in p for p in problems)


def test_reference_to_unknown_source():
    problems = check(report("Claim [1][2][3].", REFS + "[9] Ghost. https://example.com/ghost\n"), SOURCES)
    assert any("[9] is not a source" in p for p in problems)


def test_bad_source_entries():
    bad = [dict(SOURCES[0], n="1"), dict(SOURCES[1], url="ftp://x"), SOURCES[2]]
    problems = check(report("Claim [1][2][3]."), bad)
    assert any("not an integer" in p for p in problems)
    assert any("does not start with http" in p for p in problems)


def test_duplicate_source_url():
    dup = SOURCES[:2] + [dict(SOURCES[2], url=SOURCES[0]["url"])]
    problems = check(report("Claim [1][2][3]."), dup)
    assert any("duplicates source [1]" in p for p in problems)


def test_family_must_match_url():
    wrong = [dict(SOURCES[0], url="https://arxiv.org/html/2501.00001v1")] + SOURCES[1:]
    refs = REFS.replace("https://arxiv.org/abs/2501.00001", "https://arxiv.org/html/2501.00001v1")
    problems = check(report("Claim [1][2][3].", refs), wrong)
    assert any("family 'arxiv'" in p for p in problems)
