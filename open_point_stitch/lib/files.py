"""Saving files so a crash or a stopped job never leaves a half-written one behind."""

from __future__ import annotations

import json
import os
import threading
from typing import Callable, Optional


class TrainedModel:
    """A trained model's JSON file, read once and refused if it was trained on other features than this code computes.
    Tests may set `cached` to a made-up model."""

    def __init__(self, path: str, features, what: str) -> None:
        self.path, self.features, self.what = path, list(features), what
        self.cached: Optional[dict] = None

    def load(self, path: Optional[str] = None) -> dict:
        """The default file's model, kept after the first read; any other path is read fresh."""
        if path is not None and path != self.path:
            return self._read(path)
        if self.cached is None:
            self.cached = self._read(self.path)
        return self.cached

    def _read(self, path: str) -> dict:
        with open(path) as fh:
            model = json.load(fh)
        if list(model["features"]) != self.features:
            raise ValueError(f"{self.what} was trained on {model['features']}, this code computes {self.features}")
        return model


def _write_atomic(path: str, mode: str, write: Callable, **open_kwargs) -> None:
    """Whole file or nothing: write(fh) fills a temporary file beside path, which is flushed to disk and swapped in.
    The temporary name ends in .tmp (a stopped job's clean-up removes those) and is unique per process and thread."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, mode, **open_kwargs) as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_json_atomic(path: str, obj, **dump_kwargs) -> None:
    """obj as JSON; dump_kwargs go to json.dump (indent, separators, default)."""
    _write_atomic(path, "w", lambda fh: json.dump(obj, fh, **dump_kwargs))


def write_text_atomic(path: str, text: str, encoding: str = "utf-8", newline: Optional[str] = None) -> None:
    _write_atomic(path, "w", lambda fh: fh.write(text), encoding=encoding, newline=newline)


def write_bytes_atomic(path: str, data: bytes) -> None:
    _write_atomic(path, "wb", lambda fh: fh.write(data))


def write_drawing(folder: str, stem: str, dxf: str, png: bytes) -> tuple[str, Optional[str]]:
    """<stem>.dxf (ASCII, CRLF line ends as CAD programs expect) and, when png is not empty, <stem>.png beside it.
    Returns the two paths, the second None when no picture was written."""
    dxf_path = os.path.join(folder, stem + ".dxf")
    write_text_atomic(dxf_path, dxf, encoding="ascii", newline="\r\n")
    png_path = os.path.join(folder, stem + ".png")
    if not png:
        return dxf_path, None
    write_bytes_atomic(png_path, png)
    return dxf_path, png_path
