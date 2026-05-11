"""Smoke tests for MinerU visual normalization."""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mineru_client import MinerUClient


ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def make_fixture_pdf(path: Path) -> None:
    import fitz

    document = fitz.open()
    for page_index in range(4):
        page = document.new_page(width=800, height=600)
        page.insert_text((30, 30), f"PaperMate visual normalization fixture {page_index}")
        page.draw_rect(fitz.Rect(90, 80, 390, 360))
        page.draw_rect(fitz.Rect(490, 240, 590, 340))
    document.save(path)
    document.close()


def fixture_content_list() -> list[dict]:
    return [
        {
            "type": "text",
            "text": "Figure 1 Composite visual cluster",
            "bbox": [96, 80, 324, 92],
            "page_idx": 0,
        },
        {
            "type": "image",
            "img_path": "images/top-left.jpg",
            "image_caption": ["Figure 1 Composite visual cluster"],
            "bbox": [100, 100, 180, 180],
            "page_idx": 0,
        },
        {
            "type": "image",
            "img_path": "images/top-right.jpg",
            "bbox": [220, 102, 300, 182],
            "page_idx": 0,
        },
        {
            "type": "image",
            "img_path": "images/bottom-left.jpg",
            "bbox": [105, 250, 185, 330],
            "page_idx": 0,
        },
        {
            "type": "table",
            "img_path": "images/table.jpg",
            "table_caption": ["Table panel"],
            "bbox": [220, 252, 320, 332],
            "page_idx": 0,
        },
        {
            "type": "equation",
            "text": "$$\nE = mc^2\n$$",
            "bbox": [330, 254, 380, 332],
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": "Caption line near the visual cluster",
            "bbox": [100, 336, 380, 352],
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": "Far body paragraph that must not expand the figure crop",
            "bbox": [100, 500, 380, 530],
            "page_idx": 0,
        },
        {
            "type": "image",
            "img_path": "images/unrelated-top.jpg",
            "bbox": [100, 100, 180, 180],
            "page_idx": 1,
        },
        {
            "type": "image",
            "img_path": "images/unrelated-bottom.jpg",
            "bbox": [500, 250, 580, 330],
            "page_idx": 1,
        },
        {
            "type": "table",
            "img_path": "images/pure-table.jpg",
            "table_caption": ["Pure table"],
            "table_body": "<table><tr><td>Column</td><td>Value</td></tr><tr><td>A</td><td>1</td></tr></table>",
            "bbox": [100, 100, 300, 200],
            "page_idx": 2,
        },
        {
            "type": "equation",
            "text": "$$\na=b\n$$",
            "bbox": [100, 100, 200, 150],
            "page_idx": 3,
        },
    ]


def fixture_markdown() -> str:
    return "\n\n".join(
        [
            "![](images/top-left.jpg)",
            "![](images/top-right.jpg)",
            "![](images/table.jpg)",
            "$$\nE = mc^2\n$$",
            "![](images/unrelated-top.jpg)",
            "![](images/unrelated-bottom.jpg)",
            "| Column | Value |\n| --- | --- |\n| A | 1 |",
            "![](images/pure-table.jpg)",
            "$$\na=b\n$$",
        ]
    )


def image_payloads_from_blocks(tmp_path: Path, blocks: list[dict]) -> list[dict]:
    images = []
    for block in blocks:
        image_path = tmp_path / f"{block['visual_id']}.png"
        image_path.write_bytes(ONE_PIXEL_PNG)
        images.append(
            {
                "label": block["label"],
                "path": str(image_path),
                "mime_type": "image/png",
                "kind": block["kind"],
                "source_paths": block.get("source_paths", []),
                "visual_id": block["visual_id"],
                "caption": block.get("caption", ""),
                "table_body": block.get("table_body", ""),
                "source_kinds": block.get("source_kinds", [block["kind"]]),
                "contains_equation": bool(block.get("contains_equation")),
            }
        )
    return images


def assert_clustered_blocks(blocks: list[dict]) -> None:
    assert [block["label"] for block in blocks] == ["图1", "图2", "图3", "表1", "公式1"]

    first = blocks[0]
    assert first["kind"] == "image"
    assert first["source_kinds"] == ["image", "table", "equation"]
    assert first["contains_equation"] is True
    assert set(first["source_paths"]) == {
        "images/top-left.jpg",
        "images/top-right.jpg",
        "images/bottom-left.jpg",
        "images/table.jpg",
    }
    assert first["bbox"] == [96.0, 80.0, 380.0, 352.0]

    assert blocks[1]["bbox"] == [100.0, 100.0, 180.0, 180.0]
    assert blocks[2]["bbox"] == [500.0, 250.0, 580.0, 330.0]
    assert blocks[3]["kind"] == "table"
    assert blocks[4]["kind"] == "equation"


def assert_normalized_markdown(normalized_markdown: str) -> None:
    assert normalized_markdown.count("图1") == 1
    assert normalized_markdown.count("图2") == 1
    assert normalized_markdown.count("图3") == 1
    assert "表1" not in normalized_markdown
    assert normalized_markdown.count("公式1") == 1
    assert "图4" not in normalized_markdown
    assert "| Column | Value |" in normalized_markdown
    assert "pure-table.jpg" not in normalized_markdown
    assert "$$" not in normalized_markdown


def test_merge_and_markdown_rewrite_without_pdf(tmp_path: Path) -> None:
    client = MinerUClient()
    content_list = fixture_content_list()
    blocks = client.build_visual_blocks(content_list)
    merged = client.merge_visual_blocks(blocks, content_list)
    assert_clustered_blocks(merged)

    images = image_payloads_from_blocks(tmp_path, merged)
    normalized_markdown = client.replace_markdown_images_with_links(fixture_markdown(), images)
    normalized_markdown = client.replace_equations_with_visual_links(normalized_markdown, images)
    assert_normalized_markdown(normalized_markdown)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_merge_and_markdown_rewrite_without_pdf(tmp_path)

        try:
            import fitz  # noqa: F401
        except ImportError:
            print("PyMuPDF is not installed; skipped real PDF crop assertions.")
            print("MinerU visual normalization smoke tests passed.")
            return

        pdf_path = tmp_path / "fixture.pdf"
        output_dir = tmp_path / "mineru"
        output_dir.mkdir()
        make_fixture_pdf(pdf_path)

        normalized_markdown, images = MinerUClient().normalize_visual_outputs(
            fixture_markdown(),
            fixture_content_list(),
            [],
            output_dir,
            pdf_path,
        )

        labels = [image["label"] for image in images]
        assert labels == ["图1", "图2", "图3", "表1", "公式1"], labels
        assert images[0]["kind"] == "image"
        assert images[0]["source_kinds"] == ["image", "table", "equation"]
        assert set(images[0]["source_paths"]) == {
            "images/top-left.jpg",
            "images/top-right.jpg",
            "images/bottom-left.jpg",
            "images/table.jpg",
        }
        assert all(Path(image["path"]).exists() for image in images)
        assert_normalized_markdown(normalized_markdown)

    print("MinerU visual normalization smoke tests passed.")


if __name__ == "__main__":
    main()
