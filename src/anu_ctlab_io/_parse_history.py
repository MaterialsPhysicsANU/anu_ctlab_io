import re
from typing import Any

from lark import Lark, Token, Transformer
from lark.tree import Branch

_structured_parser = Lark(
    r"""
    ?history: _NEWLINE* section_contents
    ?section_contents: ((kv_pair | section) _NEWLINE+)*
    ?section: section_header _NEWLINE section_contents _section_footer
    ?section_header: "BeginSection" _WS VALUE
    _section_footer: "EndSection"
    ?kv_pair: KEY _WS [(_multi_line_string | VALUE)]
    _multi_line_string: "__start_multi_string__" _NEWLINE* MULTI_LINE_STRING "__end_multi_string__"

    KEY : /(?!EndSection)\w+/
    VALUE : /(?!__start_multi_string__)[^\n]+/
    MULTI_LINE_STRING : /((?!__end_multi_string__)(\n|.))+/

    _WS : /[^\S\r\n]+/
    _NEWLINE : [_WS] /\n/ [_WS]
""",
    start="history",
    parser="earley",
)


type KVPairs = dict[Token, Token | None]
type Section = dict[Token, SectionContents]
type SectionContents = dict[Token, Token | SectionContents | None]
type LogValue = str | list[str]
type LogContents = dict[str, LogValue]
type History = dict[str, Any]


class _StructuredTransformer(Transformer[Token, History]):
    def kv_pair(self, tree: list[Branch[Token]]) -> KVPairs:
        assert isinstance(tree[0], Token)
        assert isinstance(tree[1], Token | None)
        return {tree[0]: tree[1]}

    def section(self, tree: list[Branch[Token] | SectionContents]) -> Section:
        assert isinstance(tree[0], Token)
        assert isinstance(tree[1], dict)
        return {tree[0]: tree[1]}

    def section_header(self, tree: list[Branch[Token]]) -> Token:
        assert isinstance(tree[0], Token)
        return tree[0]

    def section_contents(self, tree: list[KVPairs | Section]) -> SectionContents:
        d: SectionContents = {}
        for i in tree:
            d |= i
        return d


def _parse_log_format(history: str) -> dict[str, Any]:
    """Parse log-style history using simple Python regex."""
    result: dict[str, Any] = {}
    log_texts: list[str] = []

    for line in history.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try to match key: value pattern (key must start with letter, followed by colon and space)
        match = re.match(r"^([A-Za-z][\w\s]+?):\s+(.+)$", line)
        if match:
            key = match.group(1).strip()
            value_str = match.group(2).strip()

            # Strip angle brackets
            value: Any = _strip_angle_brackets(value_str)

            # Handle repeated keys
            if key in result:
                existing = result[key]
                if isinstance(existing, list):
                    existing.append(value)
                else:
                    result[key] = [existing, value]
            else:
                result[key] = value
        else:
            # Unstructured line
            log_texts.append(line)

    if log_texts:
        result["_log_text"] = log_texts

    return result


def _convert_tokens_to_values(obj: object) -> Any:
    """Recursively convert Token objects to Python values."""
    if isinstance(obj, Token):
        value = obj.value

        # Strip angle brackets if present
        if isinstance(value, str):
            # Check for multiple angle brackets (array format)
            if value.startswith("<") and ">" in value:
                # Try to parse as array: <v1><v2><v3>
                matches = re.findall(r"<([^>]+)>", value)
                if len(matches) > 1:
                    # Multiple values - return as list
                    result: list[Any] = []
                    for match in matches:
                        # Try to convert to number
                        try:
                            if "." in match:
                                result.append(float(match))
                            else:
                                result.append(int(match))
                        except ValueError:
                            result.append(match)
                    return result
                elif len(matches) == 1:
                    # Single angle bracket value - strip brackets
                    value = matches[0]

            # Convert booleans
            if value in ("True", "true"):
                return True
            elif value in ("False", "false"):
                return False

        return value
    elif isinstance(obj, dict):
        return {
            _convert_tokens_to_values(k): _convert_tokens_to_values(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_convert_tokens_to_values(item) for item in obj]
    elif obj is None:
        return ""

    return obj


def _strip_angle_brackets(value: str) -> Any:
    """Strip angle brackets from values."""
    # Check for multiple angle brackets (array format)
    if value.startswith("<") and ">" in value:
        matches = re.findall(r"<([^>]+)>", value)
        if len(matches) > 1:
            # Multiple values - return as list
            result: list[float | int | str] = []
            for match in matches:
                try:
                    if "." in match:
                        result.append(float(match))
                    else:
                        result.append(int(match))
                except ValueError:
                    result.append(match)
            return result
        elif len(matches) == 1:
            # Single angle bracket - strip brackets
            return matches[0]

    return value


def _convert_log_values(obj: dict[str, Any]) -> dict[str, Any]:
    """Convert log values, converting booleans."""
    result: dict[str, Any] = {}

    for key, value in obj.items():
        if key == "_log_text":
            result[key] = value
        elif isinstance(value, list):
            converted_list: list[Any] = []
            for v in value:
                if v == "True" or v == "true":
                    converted_list.append(True)
                elif v == "False" or v == "false":
                    converted_list.append(False)
                else:
                    converted_list.append(v)
            result[key] = converted_list
        elif isinstance(value, str):
            # Convert booleans
            if value in ("True", "true"):
                result[key] = True
            elif value in ("False", "false"):
                result[key] = False
            else:
                result[key] = value
        else:
            result[key] = value

    return result


def parse_history(history: str) -> dict[str, Any]:
    """Parse MANGO history attribute into a dictionary.

    Handles both structured (BeginSection/EndSection) and log-style formats.
    Always returns a dict, never a string.

    Returns:
        Dictionary containing parsed history data. Token objects are converted
        to native Python types (str, bool, int, float, list).

        For structured format:
            Nested dicts for BeginSection/EndSection blocks

        For log format:
            Flat dict with keys
            Repeated keys become lists
            Unstructured text stored in '_log_text' key
            Angle brackets are stripped: <value> → value
            Multiple angle brackets become lists: <v1><v2><v3> → [v1, v2, v3]
    """
    # Try structured format first (BeginSection/EndSection)
    if "BeginSection" in history:
        try:
            parsed = _structured_parser.parse(history)
            transformed = _StructuredTransformer().transform(parsed)
            result: dict[str, Any] = _convert_tokens_to_values(transformed)
            return result
        except Exception:
            pass

    # Try log format (key: value pairs) - use Python regex for robustness
    if ":" in history:
        try:
            log_dict = _parse_log_format(history)
            if log_dict:  # If we parsed anything
                return _convert_log_values(log_dict)
        except Exception:
            pass

    # Complete fallback - return as raw text
    return {"_raw_history": history.strip()}


def serialize_history(history_dict: dict[str, Any], *, indent: int = 4) -> str:
    """Serialize a parsed history dictionary back to string format.

    Handles both structured (BeginSection/EndSection) and log-style formats.
    Attempts to detect format from dictionary structure and reconstruct appropriately.

    Args:
        history_dict: Parsed history dictionary from parse_history()
        indent: Number of spaces to indent nested sections (default: 4)

    Returns:
        String representation of history that can be written back to NetCDF

    Examples:
        >>> parsed = parse_history(raw_string)
        >>> reconstructed = serialize_history(parsed)
        >>> assert parse_history(reconstructed) == parsed  # Round-trip
    """
    # Handle raw history passthrough
    if "_raw_history" in history_dict and len(history_dict) == 1:
        return str(history_dict["_raw_history"])

    # Detect format: if any value is a dict (not _log_text), it's structured format
    has_nested_dicts = any(
        isinstance(v, dict) for k, v in history_dict.items() if k != "_log_text"
    )

    if has_nested_dicts:
        return _serialize_structured(history_dict, indent=indent)
    else:
        return _serialize_log(history_dict)


def _serialize_structured(data: dict[str, Any], indent: int = 4, level: int = 0) -> str:
    """Serialize structured BeginSection/EndSection format."""
    lines: list[str] = []
    indent_str = " " * (indent * level)

    for key, value in data.items():
        if isinstance(value, dict):
            # Nested section - add blank line before section at top level
            if level == 0 and lines:
                lines.append("")
            lines.append(f"{indent_str}BeginSection {key}")
            nested_content = _serialize_structured(value, indent, level + 1)
            if nested_content:
                lines.append(nested_content)
            lines.append(f"{indent_str}EndSection")
        else:
            # Key-value pair
            serialized_value = _serialize_value(value)
            # Align values at column 24 (similar to original format)
            padding = " " * max(1, 24 - len(indent_str) - len(key))
            lines.append(f"{indent_str}{key}{padding}{serialized_value}")

    result = "\n".join(lines)
    # Add trailing newline for parser compatibility at top level
    if level == 0:
        result += "\n"
    return result


def _serialize_log(data: dict[str, Any]) -> str:
    """Serialize log-style key: value format."""
    lines: list[str] = []

    for key, value in data.items():
        if key == "_log_text":
            # Unstructured text - output as-is
            if isinstance(value, list):
                lines.extend(value)
            else:
                lines.append(str(value))
        elif isinstance(value, list):
            # Repeated key - output multiple lines
            for item in value:
                serialized = _serialize_value(item)
                lines.append(f"{key}: {serialized}")
        else:
            # Single key-value pair
            serialized = _serialize_value(value)
            lines.append(f"{key}: {serialized}")

    return "\n".join(lines)


def _serialize_value(value: Any) -> str:
    """Serialize a single value to string format."""
    if isinstance(value, bool):
        return "True" if value else "False"
    elif isinstance(value, list):
        # Reconstruct angle bracket array format: [20.1, 30.2, 40.3] -> <20.1><30.2><40.3>
        return "".join(f"<{item}>" for item in value)
    elif value == "":
        return ""
    else:
        return str(value)
