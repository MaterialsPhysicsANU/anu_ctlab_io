import re

from lark import Lark, Token, Transformer
from lark.tree import Branch

_history_parser = Lark(
    r"""
    ?history: _NEWLINE* section_contents
    ?section_contents: ((kv_pair | section) _NEWLINE+)*
    ?section: section_header _NEWLINE section_contents _section_footer
    ?section_header: "BeginSection" _WS VALUE
    _section_footer: "EndSection"
    # ?kv_pair: KEY _WS [VALUE]
    ?kv_pair: KEY _WS [(_multi_line_string | VALUE)]
    _multi_line_string: "__start_multi_string__" _NEWLINE* MULTI_LINE_STRING "__end_multi_string__"

    KEY : /(?!EndSection)\w+/
    VALUE : /(?!__start_multi_string__)[^\n]+/
    MULTI_LINE_STRING : /((?!__end_multi_string__)(\n|.))+/

    COL_KEY : /(\w+\s*)+:/
    COL_VALUE : "<" KEY ">"

    _WS : /[^\S\r\n]+/
    _NEWLINE : [_WS] /\n/ [_WS]
""",
    start="history",
    parser="earley",
)

type KVPairs = dict[Token, Token | None]
type Section = dict[Token, SectionContents]
type SectionContents = dict[Token, Token | SectionContents | None]
type History = dict[Token, KVPairs | Section | None]


class _HistoryTransformer(Transformer[Token, History]):
    def kv_pair(self, tree: list[Branch[Token]]) -> KVPairs:
        assert isinstance(tree[0], Token)
        assert isinstance(tree[1], Token | None)
        return {tree[0]: tree[1]}

    def section(self, tree: list[Branch[Token] | SectionContents]) -> Section:
        assert isinstance(tree[0], Token)
        assert isinstance(tree[1], dict)  # KVPairs
        return {tree[0]: tree[1]}

    def section_header(self, tree: list[Branch[Token]]) -> Token:
        assert isinstance(tree[0], Token)
        return tree[0]

    def section_contents(self, tree: list[KVPairs | Section]) -> SectionContents:
        d: SectionContents = {}
        for i in tree:
            d |= i
        return d


def parse_history(history: str) -> History | str:
    if re.match(r"\n*([^\n\r])+:\s+([^\n]+)", history):
        try:
            lines = history.strip().split("\n")
            ks, vs = zip(*map(lambda x: x.split(":", 1), lines), strict=True)
            return dict(
                zip(
                    map(lambda x: x.strip(), ks),
                    map(lambda x: x.strip(), vs),
                    strict=True,
                )
            )
        except ValueError:
            return history
    else:
        hist = _HistoryTransformer().transform(_history_parser.parse(history))
        return hist
