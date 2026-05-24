"""Smoke test for the configured Qwen VLM image description client."""

from __future__ import annotations

import struct
import sys
import tempfile
import zlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.vlm_client import QwenVLMClient


def main() -> None:
    if not settings.vlm_api_key:
        raise SystemExit("Missing VLM API key. Set DASHSCOPE_API_KEY or VLM_API_KEY in .env.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "red_square.png"
        image_path.write_bytes(build_rgb_png(16, 16, (255, 0, 0)))
        description = QwenVLMClient(max_tokens=128, temperature=0.0).describe(
            {
                "kind": "image",
                "caption": "Smoke test red square image",
                "alt_text": "single red square",
                "path": str(image_path),
                "mime_type": "image/png",
                "page_num": 1,
                "bbox": [0, 0, 16, 16],
            }
        )
    if not description:
        raise AssertionError("Qwen VLM returned an empty description")
    print("qwen vlm smoke test passed")
    print(description[:240])


def build_rgb_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Build a tiny RGB PNG without third-party dependencies."""
    raw_scanlines = b"".join(
        b"\x00" + bytes(rgb) * width
        for _ in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw_scanlines))
        + png_chunk(b"IEND", b"")
    )


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    payload = chunk_type + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


if __name__ == "__main__":
    main()
