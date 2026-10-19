"""Splice ui-src/dist/open-point-ui.js into placement_bench.html between its REACT_UI_START/END markers. Build the
bundle first (npm run build in ui-src)."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATE = os.path.join(ROOT, "placement_bench.html")
BUNDLE = os.path.join(ROOT, "ui-src", "dist", "open-point-ui.js")

START_MARKER = "<!--REACT_UI_START-->"
END_MARKER = "<!--REACT_UI_END-->"


def splice(template_path: str, bundle_path: str) -> str:
    with open(bundle_path, "r", encoding="utf-8") as fh:
        bundle_js = fh.read()
    with open(template_path, "r", encoding="utf-8") as fh:
        html = fh.read()

    start = html.find(START_MARKER)
    end = html.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise RuntimeError(
            f"{template_path}: {START_MARKER}/{END_MARKER} markers not found or out of order"
        )
    # A literal </script> would end the inline script element early; refuse rather than assume it never appears.
    if "</script" in bundle_js:
        raise RuntimeError(
            f"{bundle_path}: contains a literal '</script' - would truncate the "
            f"inline <script> tag early. Check the build output before splicing."
        )

    block = f"{START_MARKER}\n<script>\n{bundle_js}\n</script>\n{END_MARKER}"
    return html[:start] + block + html[end + len(END_MARKER):]


def main() -> int:
    if not os.path.exists(BUNDLE):
        print(
            f"error: {BUNDLE} does not exist - build it first:\n"
            f'  docker run --rm -v "{os.path.join(ROOT, "ui-src")}:/ui-src" -w /ui-src '
            f'node:20-slim sh -c "npm install && npm run build"',
            file=sys.stderr,
        )
        return 1
    html = splice(TEMPLATE, BUNDLE)
    with open(TEMPLATE, "w", encoding="utf-8", newline="\n") as fh:     # LF on Windows too
        fh.write(html)
    print(f"spliced {BUNDLE} into {TEMPLATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
