"""Uploaded documents: encodings, formats and chunking (#68)."""

import pytest

from app.utils.file_parser import FileParser, split_text_into_chunks

TEXT = (
    "東海市交通運輸委員會今天批准凌雲飛行開始試營運。計程車工會表示反對，"
    "並要求政府設立轉職補助。凌雲飛行執行長陳志遠說，首批六架航空器將在下月載客。"
) * 3


@pytest.mark.parametrize("encoding", ["utf-8", "big5", "gb18030"])
def test_text_in_common_chinese_encodings_is_read(tmp_path, encoding):
    path = tmp_path / "news.txt"
    source = TEXT if encoding != "gb18030" else TEXT.replace("東", "东").replace("會", "会")
    path.write_bytes(source.encode(encoding))
    assert FileParser.extract_text(str(path)) == source


def test_markdown_is_read_like_text(tmp_path):
    path = tmp_path / "news.md"
    path.write_text("# 標題\n\n" + TEXT, encoding="utf-8")
    assert FileParser.extract_text(str(path)).startswith("# 標題")


def test_unsupported_and_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileParser.extract_text(str(tmp_path / "nope.txt"))
    other = tmp_path / "sheet.xlsx"
    other.write_bytes(b"x")
    with pytest.raises(ValueError):
        FileParser.extract_text(str(other))
    assert FileParser.is_supported("a.PDF") and not FileParser.is_supported("a.docx")


def test_several_files_are_labelled_and_a_failure_does_not_stop_the_rest(tmp_path):
    good = tmp_path / "a.txt"
    good.write_text("第一份", encoding="utf-8")
    merged = FileParser.extract_from_multiple([str(good), str(tmp_path / "missing.txt")])
    assert "=== 文档 1: a.txt ===\n第一份" in merged
    assert "文档 2" in merged and "提取失败" in merged


def test_chunks_cover_the_text_with_overlap():
    assert split_text_into_chunks("短文", chunk_size=500) == ["短文"]
    assert split_text_into_chunks("   ", chunk_size=500) == []
    chunks = split_text_into_chunks(TEXT * 5, chunk_size=200, overlap=50)
    assert len(chunks) > 1 and all(len(c) <= 200 for c in chunks)
    joined = "".join(chunks)
    for sentence in ("凌雲飛行開始試營運", "首批六架航空器"):
        assert sentence in joined
