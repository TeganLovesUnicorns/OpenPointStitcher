"""Writing R12 ASCII DXF, the oldest form every CAD program reads: no handles or object tables to get wrong."""

from __future__ import annotations

import numpy as np


def ascii_text(text: str) -> str:
    """Text a DXF can hold: anything outside ASCII becomes '?'."""
    return text.encode("ascii", "replace").decode("ascii")


class DxfWriter:
    """One DXF file's text: the header and layer table on creation, entities added in order, text() at the end."""

    def __init__(self, layers) -> None:
        self._out: list[str] = []
        add = self.add
        add(0, "SECTION"); add(2, "HEADER"); add(9, "$ACADVER"); add(1, "AC1009")
        add(9, "$INSBASE"); add(10, 0.0); add(20, 0.0); add(30, 0.0)
        add(0, "ENDSEC")
        add(0, "SECTION"); add(2, "TABLES")
        add(0, "TABLE"); add(2, "LTYPE"); add(70, 1)
        add(0, "LTYPE"); add(2, "CONTINUOUS"); add(70, 0); add(3, "Solid line"); add(72, 65); add(73, 0); add(40, 0.0)
        add(0, "ENDTAB")
        add(0, "TABLE"); add(2, "LAYER"); add(70, len(layers))
        for name, colour in layers:
            add(0, "LAYER"); add(2, name); add(70, 0); add(62, colour); add(6, "CONTINUOUS")
        add(0, "ENDTAB")
        add(0, "ENDSEC")
        add(0, "SECTION"); add(2, "ENTITIES")

    def add(self, code: int, value) -> None:
        self._out.append(f"{code:3d}")
        self._out.append(str(value))

    def polyline(self, points: np.ndarray, layer: str, closed: bool) -> None:
        """A flat (2D) polyline at height 0."""
        add = self.add
        add(0, "POLYLINE"); add(8, layer); add(66, 1); add(70, 1 if closed else 0)
        for x, y in points:
            add(0, "VERTEX"); add(8, layer); add(10, f"{x:.4f}"); add(20, f"{y:.4f}"); add(30, "0.0")
        add(0, "SEQEND"); add(8, layer)

    def polyline3d(self, points: np.ndarray, layer: str) -> None:
        """A closed 3D polyline."""
        add = self.add
        add(0, "POLYLINE"); add(8, layer); add(66, 1); add(10, 0.0); add(20, 0.0); add(30, 0.0)
        add(70, 9)                                               # closed (1) and 3D (8)
        for x, y, z in points:
            add(0, "VERTEX"); add(8, layer); add(10, f"{x:.4f}"); add(20, f"{y:.4f}"); add(30, f"{z:.4f}")
            add(70, 32)                                          # a 3D polyline's vertex
        add(0, "SEQEND"); add(8, layer)

    def label(self, x: float, y: float, z, height: float, value: str, layer: str) -> None:
        """A text entity; z is written as given ("0.0" on a plan)."""
        add = self.add
        add(0, "TEXT"); add(8, layer); add(10, f"{x:.4f}"); add(20, f"{y:.4f}")
        add(30, z if isinstance(z, str) else f"{z:.4f}")
        add(40, height); add(1, ascii_text(value))

    def text(self) -> str:
        """The finished file."""
        self.add(0, "ENDSEC")
        self.add(0, "EOF")
        return "\n".join(self._out) + "\n"
