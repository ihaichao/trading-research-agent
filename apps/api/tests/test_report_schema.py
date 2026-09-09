"""报告契约的测试。schema 是前后端的接口，所以它的规则必须被测死。"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from examples.sample_report import OUT as SAMPLE
from examples.sample_report import build
from pydantic import ValidationError

from tra.report import (
    Claim,
    ClaimKind,
    Report,
    Section,
    SectionKind,
    Source,
    SourceKind,
    make_source_id,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _source(url: str = "https://example.com/a") -> Source:
    return Source(
        id=make_source_id(url),
        kind=SourceKind.WEB,
        title="a",
        url=url,
        retrieved_at=NOW,
    )


def _report(sections: list[Section], sources: list[Source]) -> Report:
    return Report(
        ticker="EXMP",
        company_name="Example Corp",
        question="q",
        generated_at=NOW,
        data_as_of=NOW,
        sections=sections,
        sources=sources,
    )


def test_make_source_id_is_deterministic_and_locator_sensitive() -> None:
    a = make_source_id("https://example.com/x", "Item 1A")
    assert a == make_source_id("https://example.com/x", "Item 1A")
    assert a != make_source_id("https://example.com/x", "Item 7")
    assert len(a) == 10


def test_rejects_dangling_citation() -> None:
    section = Section(
        kind=SectionKind.SUMMARY,
        title="s",
        claims=[Claim(text="made up", source_ids=["deadbeef00"])],
    )
    with pytest.raises(ValidationError, match="unknown source ids"):
        _report([section], [_source()])


def test_rejects_duplicate_source_ids() -> None:
    src = _source()
    with pytest.raises(ValidationError, match="duplicate source ids"):
        _report(
            [Section(kind=SectionKind.SUMMARY, title="s")],
            [src, src.model_copy()],
        )


def test_rejects_uncited_claim() -> None:
    with pytest.raises(ValidationError):
        Claim(text="no source", source_ids=[])


def test_accepts_a_well_formed_report() -> None:
    src = _source()
    section = Section(
        kind=SectionKind.SUMMARY,
        title="s",
        claims=[Claim(text="cited", source_ids=[src.id])],
    )
    report = _report([section], [src])
    assert report.claim_count == 1
    assert report.source_by_id(src.id) is src
    assert report.source_by_id("0000000000") is None
    assert report.disclaimer


def test_sample_report_is_valid_and_matches_the_checked_in_fixture() -> None:
    """样例文件是前端的开发数据源，必须和当前 schema 保持同步。

    这个测试挂了说明你改了 schema 但忘了 `make schema`。
    """
    assert SAMPLE.exists(), "run: make schema"
    on_disk = json.loads(SAMPLE.read_text(encoding="utf-8"))
    Report.model_validate(on_disk)
    assert on_disk == build().model_dump(mode="json")


def test_round_trips_through_json() -> None:
    report = build()
    assert Report.model_validate_json(report.model_dump_json()) == report


def test_a_limitation_may_cite_nothing_but_a_finding_may_not() -> None:
    """**"每句都有出处"这条承诺没有松动，只是被说清楚了。**

    它管的是"关于公司的论断"。"这份证据回答不了这个问题"没有出处可引——
    以前的 schema 不区分这两者，于是把唯一诚实的回答判成了非法输入。
    """
    Claim(
        text="A P/E ratio cannot be computed: share count is not in the evidence.",
        kind=ClaimKind.LIMITATION,
    )
    with pytest.raises(ValidationError):
        Claim(text="Revenue reached 46,743.")
    with pytest.raises(ValidationError):
        Claim(text="Guidance is absent.", kind=ClaimKind.LIMITATION, source_ids=["aaaaaaaaaa"])
