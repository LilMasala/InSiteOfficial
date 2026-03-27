#!/usr/bin/env python3
from pathlib import Path
import re
import sys

SYMBOL_MAP = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "η": r"\eta",
    "θ": r"\theta",
    "κ": r"\kappa",
    "π": r"\pi",
    "ρ": r"\rho",
    "φ": r"\phi",
    "ξ": r"\xi",
    "τ": r"\tau",
    "ω": r"\omega",
    "Σ": r"\Sigma",
    "∈": r"\in",
    "∪": r"\cup",
    "≈": r"\approx",
    "≤": r"\leq",
    "≥": r"\geq",
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
}

MATHY_CHARS = set(SYMBOL_MAP.keys()) | set("_^\\=")

def convert_unicode_to_latex(text: str) -> str:
    for k, v in SYMBOL_MAP.items():
        text = text.replace(k, v)
    return text

def looks_mathy(s: str) -> bool:
    if any(ch in s for ch in MATHY_CHARS):
        return True
    if re.search(r"[A-Za-z]+_[A-Za-z0-9{}]+", s):
        return True
    if re.search(r"\\[A-Za-z]+", s):
        return True
    return False

def fix_inline_code(line: str) -> str:
    def repl(m):
        content = m.group(1)
        if looks_mathy(content):
            fixed = convert_unicode_to_latex(content)
            return f"${fixed}$"
        return m.group(0)
    return re.sub(r"`([^`]+)`", repl, line)

def literal_sub(pattern: str, replacement: str, text: str) -> str:
    return re.sub(pattern, lambda m: replacement, text)

def wrap_unicode_tokens(line: str) -> str:
    for uni, latex in SYMBOL_MAP.items():
        math_repl = f"${latex}$"

        # e.g. δ-credible -> $\delta$-credible
        line = literal_sub(
            rf"(?<![$\\]){re.escape(uni)}(?=[A-Za-z0-9_-])",
            math_repl,
            line,
        )

        # e.g. trustδ -> trust$\delta$
        line = literal_sub(
            rf"(?<=[A-Za-z0-9_-]){re.escape(uni)}(?![$\\])",
            math_repl,
            line,
        )

        # standalone symbol
        line = literal_sub(
            rf"(?<![$\\A-Za-z]){re.escape(uni)}(?![A-Za-z])",
            math_repl,
            line,
        )

    return line

def process_text(text: str) -> str:
    out = []
    in_fence = False
    fence_char = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            cur = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_char = cur
            elif cur == fence_char:
                in_fence = False
                fence_char = None
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        line = fix_inline_code(line)
        line = wrap_unicode_tokens(line)
        out.append(line)

    return "\n".join(out) + "\n"

def main():
    if len(sys.argv) != 2:
        print("usage: python3 fix_math.py path/to/file.md")
        sys.exit(1)

    path = Path(sys.argv[1])
    original = path.read_text(encoding="utf-8")
    fixed = process_text(original)

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    path.write_text(fixed, encoding="utf-8")

    print(f"fixed:  {path}")
    print(f"backup: {backup}")

if __name__ == "__main__":
    main()