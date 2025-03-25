#!/usr/bin/env python
import re
from lark import Lark, Transformer

_history_parser = Lark(
    r"""
    ?history: _NEWLINE* section_contents
    ?section_contents: ((kv_pair | section) _NEWLINE+)*
    ?section: section_header _NEWLINE section_contents _section_footer
    ?section_header: "BeginSection" _WS VALUE
    _section_footer: "EndSection"
    ?kv_pair: KEY _WS [VALUE]

    KEY : /(?!EndSection)\w+/
    VALUE : /\S+/

    COL_KEY : /(\w+\s*)+:/
    COL_VALUE : "<" KEY ">"

    _WS : /[^\S\r\n]+/
    _NEWLINE : [_WS] /\n/ [_WS]
""",
    start="history",
    parser="lalr",
)


class _HistoryTransformer(Transformer):
    def kv_pair(self, tree):
        return {tree[0].value: None if not tree[1] else tree[1].value}

    def section(self, tree):
        return {tree[0].value: tree[1]}

    def section_header(self, tree):
        return tree[0].value

    def section_contents(self, tree):
        d = {}
        for i in tree:
            d |= i
        return d


def parse_history(history):
    if re.match(r"\n*([^\n\r])+:\s+([^\n]+)", history):
        lines = history.strip().split("\n")
        ks, vs = zip(*map(lambda x: x.split(":"), lines))
        return dict(zip(map(lambda x: x.strip(), ks), map(lambda x: x.strip(), vs)))
    else:
        hist = _HistoryTransformer().transform(_history_parser.parse(history))
        return hist
