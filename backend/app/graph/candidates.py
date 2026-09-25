"""Entity-name candidates from Chinese text without a generative model.

Each candidate carries boundary ``variants`` (longest first). When there is
more than one, the extractor asks System One which variant is the complete
proper name, then asks for its type (with a ``none`` option that drops
fragments and generic words).

Sources, unioned:

* organisations: a span ending in an organisation suffix (局, 委員會, 公司,
  工會, 研究所 ...). It extends left character by character and stops at
  punctuation or a function word. Variants start at each jieba word
  boundary inside that span, so 本地食品大廠星河食品股份有限公司 offers
  星河食品股份有限公司 as well;
* person names: jieba POS ``nr*`` tokens (adjacent ones joined), plus names
  after a title word (教授, 執行長, 議員 ...) that start with a common
  surname, offered as 3- and 2-character variants;
* quoted names: text inside 「」, 『』 or “”.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ORG_SUFFIXES = (
    "委員會", "工作組", "聯誼會", "研究所", "研究院", "基金會", "電視台",
    "有限公司", "公司", "工會", "協會", "大學", "學院", "政府", "醫院", "銀行", "日報",
    "聯盟", "集團", "科技", "法院", "議會", "人大", "部門",
    "局", "委", "署", "廳", "黨",
)
# Single-character suffixes also occur inside ordinary words (委託, 局面);
# they only count when the next character does not continue such a word.
_SINGLE_SUFFIX_BLOCKERS = {
    "局": "面勢限部長",
    "委": "託屈婉任員",
    "署": "名理",
    "廳": "堂",
    "黨": "羽",
}
STOP_WORDS = (
    "與", "和", "及", "並", "同", "由", "在", "向", "對", "從", "將", "把", "被", "為",
    "的", "了", "是", "等", "據", "則", "而", "也", "亦", "即", "或", "給", "讓", "使",
    "於", "以", "至", "到", "經", "當", "若", "如",
    "根據", "代表", "表示", "指出", "發表", "召開", "宣佈", "宣布", "提交", "針對", "關於",
    "包括", "以及", "隨著", "呼籲", "要求", "質疑", "認為", "強調", "希望", "報導",
    "委託", "批評", "本土", "本地", "首批",
)
TITLE_WORDS = (
    "主任委員", "副局長", "局長", "副市長", "市長", "執行長", "董事長", "總經理", "總裁",
    "發言人", "教授", "研究員", "議員", "委員", "理事長", "主席", "會長", "院長", "所長",
    "記者", "律師", "醫師", "代表", "部長", "署長", "處長", "主任",
)
# Common Chinese surnames (traditional forms), used to anchor names after titles.
SURNAMES = set(
    "陳林黃張李王吳劉蔡楊許鄭謝洪郭邱曾廖賴徐周葉蘇莊呂江何蕭羅高潘簡朱鍾彭游詹胡施沈"
    "余趙盧梁顏柯孫魏翁戴范宋方鄧杜傅侯曹薛丁卓馬阮董唐溫藍蔣石古紀姚連馮歐程湯田康姜"
    "汪白鄒尤巫鐘黎涂龔嚴韓袁金童陸夏柳邵錢伍倪于譚駱熊任甘秦顧毛章史官萬俞雷饒闕凌崔"
    "尹孔辛龍雲易喬賀賈聶"
)
_PUNCT = set("，。、；：！？（）()「」『』“”\"'《》〈〉,.;:!?\n\r\t 　【】[]")
_QUOTED_RE = re.compile(r"[「『“]([^「」『』“”]{2,20})[」』”]")
_CJK_RE = re.compile(r"^[一-鿿]+$")
MAX_ORG_CHARS = 16
MAX_VARIANTS = 10


@dataclass(frozen=True)
class Candidate:
    text: str
    start: int
    end: int
    source: str  # "person" | "org" | "quoted"
    variants: tuple[str, ...] = field(default=())

    def options(self) -> list[str]:
        return list(self.variants) if self.variants else [self.text]


def _pos_tokens(text: str) -> list[tuple[str, str, int, int]]:
    import jieba
    import jieba.posseg as pseg

    jieba.setLogLevel(60)
    tokens = []
    position = 0
    for pair in pseg.cut(text):
        word, flag = pair.word, pair.flag
        start = text.find(word, position)
        if start < 0:
            continue
        end = start + len(word)
        tokens.append((word, flag, start, end))
        position = end
    return tokens


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def _persons(text: str, tokens: list[tuple[str, str, int, int]]) -> list[Candidate]:
    found: list[Candidate] = []
    i = 0
    while i < len(tokens):
        word, flag, start, end = tokens[i]
        if flag.startswith("nr"):
            j = i + 1
            while (
                j < len(tokens)
                and tokens[j][1].startswith("nr")
                and tokens[j][2] == end
                and (end - start) + len(tokens[j][0]) <= 4
            ):
                end = tokens[j][3]
                j += 1
            name = text[start:end]
            variants = [name]
            # jieba often drops the surname: 高|明哲 -> offer 高明哲 too.
            if len(name) == 2 and start > 0 and text[start - 1] in SURNAMES:
                variants.insert(0, text[start - 1 : end])
                start -= 1
            if 2 <= len(name) <= 4 and _is_cjk(name):
                found.append(Candidate(variants[0], start, end, "person", tuple(variants)))
            i = j
        else:
            i += 1
    for title in TITLE_WORDS:
        for match in re.finditer(re.escape(title), text):
            pos = match.end()
            if pos < len(text) and text[pos] in SURNAMES:
                variants = [
                    text[pos : pos + n]
                    for n in (3, 2)
                    if pos + n <= len(text) and _is_cjk(text[pos : pos + n])
                ]
                if variants:
                    found.append(
                        Candidate(variants[0], pos, pos + len(variants[0]), "person", tuple(variants))
                    )
    return found


def _suffix_matches(text: str) -> list[tuple[int, int, str]]:
    matches = []
    for suffix in ORG_SUFFIXES:
        for match in re.finditer(re.escape(suffix), text):
            start, end = match.span()
            if any(
                text.startswith(longer, start)
                for longer in ORG_SUFFIXES
                if len(longer) > len(suffix) and longer.startswith(suffix)
            ):
                continue  # 委 inside 委員會, 公司 inside 公司治理 is fine below
            if len(suffix) == 1 and end < len(text) and text[end] in _SINGLE_SUFFIX_BLOCKERS[suffix]:
                continue
            # 科技 in 創新科技局: the name continues into another suffix.
            if any(
                text.startswith(other, end)
                and not (len(other) == 1 and end + 1 < len(text)
                         and text[end + 1] in _SINGLE_SUFFIX_BLOCKERS[other])
                for other in ORG_SUFFIXES
            ):
                continue
            matches.append((start, end, suffix))
    return matches


def _left_edge(text: str, suffix_start: int, end: int, *, cross_orgs: bool) -> int:
    """Walk left to punctuation or a stop word; unless ``cross_orgs``, also
    stop where another organisation name ends (科技局|聯合交通...)."""

    begin = suffix_start
    while begin > 0 and end - begin < MAX_ORG_CHARS:
        prefix = text[:begin]
        if prefix[-1] in _PUNCT or not _is_cjk(prefix[-1]):
            break
        if any(prefix.endswith(word) for word in STOP_WORDS):
            break
        if not cross_orgs and begin != suffix_start and any(
            prefix.endswith(s) for s in ORG_SUFFIXES
        ):
            break
        begin -= 1
    return begin


def _organisations(text: str, tokens: list[tuple[str, str, int, int]]) -> list[Candidate]:
    token_starts = {t[2] for t in tokens}
    by_end: dict[int, Candidate] = {}
    for start, end, suffix in _suffix_matches(text):
        begin = _left_edge(text, start, end, cross_orgs=False)
        wide = _left_edge(text, start, end, cross_orgs=True)
        if end - begin <= len(suffix) and end - wide <= len(suffix):
            continue
        starts = [begin] + [s for s in range(begin + 1, start) if s in token_starts]
        # A name that spans another organisation (東海大學|法學院) is offered
        # after the in-boundary variants.
        if wide < begin:
            starts.append(wide)
        variants = []
        for s in starts:
            name = text[s:end]
            if len(name) > len(suffix) and _is_cjk(name) and name not in variants:
                variants.append(name)
        if not variants:
            continue
        variants = variants[:MAX_VARIANTS]
        current = by_end.get(end)
        if current is None or len(variants[0]) > len(current.text):
            by_end[end] = Candidate(variants[0], end - len(variants[0]), end, "org", tuple(variants))
    return list(by_end.values())


def _quoted(text: str) -> list[Candidate]:
    return [
        Candidate(m.group(1), m.start(1), m.end(1), "quoted", (m.group(1),))
        for m in _QUOTED_RE.finditer(text)
        if not any(ch in "，。；、" for ch in m.group(1))
    ]


def find_candidates(text: str) -> list[Candidate]:
    """Candidates in order of first appearance, one per distinct span text.

    A candidate whose span lies inside another candidate's span and is one
    of that candidate's variants is dropped (the variant choice covers it).
    """

    tokens = _pos_tokens(text)
    items = _organisations(text, tokens) + _persons(text, tokens) + _quoted(text)
    items.sort(key=lambda c: (c.start, -(c.end - c.start)))
    kept: list[Candidate] = []
    seen: set[str] = set()
    for item in items:
        if item.text in seen:
            continue
        if any(
            other.start <= item.start and item.end <= other.end and item.text in other.options()
            for other in kept
        ):
            continue
        seen.add(item.text)
        kept.append(item)
    return kept
