"""MinerU PDF-to-Markdown API client."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import re
import shutil
import subprocess
import time
from urllib.parse import unquote
import zipfile
from pathlib import Path
from typing import Any

import requests

from config import settings
from src.errors import ErrorCode, MinerUError
from src.logger import get_logger


logger = get_logger(__name__)


DONE_STATES = {"done", "success", "completed", "complete"}
FAILED_STATES = {"failed", "fail", "error"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
VISUAL_TYPES = {"image", "table", "equation"}
VISUAL_LABEL_PREFIX = {"image": "图", "table": "表", "equation": "公式"}
VISUAL_KIND_PRIORITY = {"image": 0, "table": 1, "equation": 2}
VISUAL_ROW_OVERLAP_THRESHOLD = 0.35
VISUAL_CLUSTER_VERTICAL_GAP = 120.0
VISUAL_CLUSTER_HORIZONTAL_OVERLAP_THRESHOLD = 0.25
VISUAL_CAPTION_VERTICAL_GAP = 120.0
VISUAL_CAPTION_HORIZONTAL_OVERLAP_THRESHOLD = 0.30
VISUAL_CROP_MARGIN = 8.0


class MinerUClient:
    """Client for MinerU's local batch file upload extract API."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.api_token = settings.mineru_api_token
        self.base_url = settings.mineru_base_url.rstrip("/")
        self.session = session or requests.Session()

    def pdf_to_markdown(
        self,
        file_path: str | Path,
        paper_id: str,
        file_name: str | None = None,
        include_images: bool = False,
    ) -> dict[str, Any]:
        """Upload a local PDF to MinerU and save the returned Markdown."""
        if not self.api_token:
            raise MinerUError(ErrorCode.MINERU_API_TOKEN_MISSING)

        path = Path(file_path)
        upload_name = file_name or path.name

        batch_id, upload_url = self.request_upload_url(upload_name, paper_id)
        self.upload_file(upload_url, path)
        file_result = self.wait_for_result(batch_id, upload_name, paper_id)
        outputs = self.download_outputs(file_result, paper_id, path, include_images=include_images)

        return {
            "paper_id": paper_id,
            "batch_id": batch_id,
            "markdown": outputs["markdown"],
            "markdown_path": outputs["markdown_path"],
            "content_list": outputs.get("content_list"),
            "content_list_path": outputs.get("content_list_path"),
            "images": outputs.get("images", []),
            "parser": "mineru",
        }

    def request_upload_url(self, file_name: str, paper_id: str) -> tuple[str, str]:
        """Request a temporary upload URL from MinerU."""
        payload = {
            "enable_formula": settings.mineru_enable_formula,
            "enable_table": settings.mineru_enable_table,
            "language": settings.mineru_language,
            "model_version": settings.mineru_model_version,
            "files": [
                {
                    "name": file_name,
                    "is_ocr": settings.mineru_is_ocr,
                    "data_id": paper_id,
                }
            ],
        }

        try:
            response = self.session.post(
                self.url("/api/v4/file-urls/batch"),
                headers=self.json_headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 0:
                logger.error("MinerU upload URL request failed: %s", data)
                raise MinerUError(ErrorCode.MINERU_UPLOAD_URL_FAILED)

            result = data["data"]
            batch_id = str(result["batch_id"])
            upload_urls = result["file_urls"]
            if not upload_urls:
                raise KeyError("file_urls is empty")
            return batch_id, str(upload_urls[0])
        except MinerUError:
            raise
        except (KeyError, TypeError, ValueError, requests.RequestException) as exc:
            logger.exception("MinerU upload URL request failed")
            raise MinerUError(ErrorCode.MINERU_UPLOAD_URL_FAILED) from exc

    def upload_file(self, upload_url: str, file_path: Path) -> None:
        """Upload the PDF bytes to MinerU's temporary upload URL."""
        try:
            with file_path.open("rb") as file_obj:
                response = self.session.put(upload_url, data=file_obj, timeout=180)
            if response.status_code not in {200, 201, 204}:
                logger.error(
                    "MinerU file upload failed, status=%s body=%s",
                    response.status_code,
                    response.text[:1000],
                )
                raise MinerUError(ErrorCode.MINERU_UPLOAD_FAILED)
        except MinerUError:
            raise
        except (OSError, requests.RequestException) as exc:
            logger.exception("MinerU file upload failed")
            raise MinerUError(ErrorCode.MINERU_UPLOAD_FAILED) from exc

    def wait_for_result(self, batch_id: str, file_name: str, paper_id: str) -> dict[str, Any]:
        """Poll MinerU until the batch file result is done or failed."""
        deadline = time.monotonic() + settings.mineru_poll_timeout
        last_state = ""

        while time.monotonic() < deadline:
            payload = self.get_batch_result(batch_id)
            file_result = self.pick_file_result(payload, file_name, paper_id)

            if file_result:
                state = str(file_result.get("state", "")).lower()
                last_state = state or last_state
                if state in DONE_STATES:
                    return file_result
                if state in FAILED_STATES:
                    logger.error("MinerU parse failed: %s", file_result)
                    raise MinerUError(ErrorCode.MINERU_PARSE_FAILED)

            time.sleep(max(1.0, settings.mineru_poll_interval))

        logger.error("MinerU parse timeout, batch_id=%s last_state=%s", batch_id, last_state)
        raise MinerUError(ErrorCode.MINERU_PARSE_TIMEOUT)

    def get_batch_result(self, batch_id: str) -> dict[str, Any]:
        """Fetch one batch result payload from MinerU."""
        try:
            response = self.session.get(
                self.url(f"/api/v4/extract-results/batch/{batch_id}"),
                headers=self.auth_headers(),
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 0:
                logger.error("MinerU batch result request failed: %s", data)
                raise MinerUError(ErrorCode.MINERU_PARSE_FAILED)
            return data
        except MinerUError:
            raise
        except (ValueError, requests.RequestException) as exc:
            logger.exception("MinerU batch result request failed")
            raise MinerUError(ErrorCode.MINERU_PARSE_FAILED) from exc

    def pick_file_result(
        self,
        payload: dict[str, Any],
        file_name: str,
        paper_id: str,
    ) -> dict[str, Any] | None:
        """Pick this PDF's result from a batch payload."""
        data = payload.get("data") or {}
        results = (
            data.get("extract_result")
            or data.get("extract_results")
            or data.get("results")
            or data.get("files")
            or []
        )
        if isinstance(results, dict):
            results = [results]

        for result in results:
            if not isinstance(result, dict):
                continue
            if result.get("data_id") == paper_id:
                return result
            if result.get("file_name") == file_name or result.get("name") == file_name:
                return result

        return results[0] if len(results) == 1 and isinstance(results[0], dict) else None

    def download_outputs(
        self,
        file_result: dict[str, Any],
        paper_id: str,
        pdf_path: Path | None = None,
        include_images: bool = False,
    ) -> dict[str, Any]:
        """Download Markdown and optional content_list from MinerU result URLs."""
        output_dir = settings.mineru_output_dir / paper_id
        output_dir.mkdir(parents=True, exist_ok=True)

        markdown = ""
        content_list: list[dict[str, Any]] | None = None
        images: list[dict[str, Any]] = []

        zip_url = self.first_result_value(file_result, ("full_zip_url", "zip_url", "fullZipUrl"))
        md_url = self.first_result_value(
            file_result,
            ("md_url", "markdown_url", "markdownUrl", "full_md_url"),
        )
        content_list_url = self.first_result_value(
            file_result,
            ("content_list_url", "contentListUrl", "content_list_json_url"),
        )

        if zip_url:
            try:
                # Prefer the full zip because it carries Markdown, content_list,
                # and images together; direct md_url is only a text fallback.
                markdown, content_list, images = self.download_zip_outputs(
                    str(zip_url),
                    output_dir,
                    include_images=include_images,
                )
            except MinerUError:
                if not md_url:
                    raise
                logger.warning("MinerU zip download failed, falling back to md_url: %s", md_url)
                images = []
        else:
            images = []

        if not markdown and md_url:
            markdown = self.download_text(str(md_url))

        if content_list is None and content_list_url:
            content_list = self.download_content_list(str(content_list_url))

        if not markdown and not md_url:
            logger.error("MinerU result has no downloadable Markdown URL: %s", file_result)
            raise MinerUError(ErrorCode.MINERU_NO_MARKDOWN)

        if not markdown.strip():
            raise MinerUError(ErrorCode.MINERU_NO_MARKDOWN)

        if include_images and content_list:
            try:
                markdown, images = self.normalize_visual_outputs(
                    markdown,
                    content_list,
                    images,
                    output_dir,
                    pdf_path,
                )
            except Exception:
                logger.exception("MinerU visual normalization failed; falling back to archive images.")
                if images:
                    markdown = self.replace_markdown_images_with_links(markdown, images)
        elif include_images and images:
            # Streamlit cannot safely preview arbitrary local image paths inside
            # Markdown, so the saved images are exposed as data URI links.
            markdown = self.replace_markdown_images_with_links(markdown, images)

        markdown_path = output_dir / "full.md"
        markdown_path.write_text(markdown, encoding="utf-8")

        content_list_path = None
        if content_list is not None:
            content_list_path = output_dir / "content_list.json"
            content_list_path.write_text(
                json.dumps(content_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {
            "markdown": markdown,
            "markdown_path": str(markdown_path.resolve()),
            "content_list": content_list,
            "content_list_path": str(content_list_path.resolve()) if content_list_path else None,
            "images": images,
        }

    def first_result_value(self, file_result: dict[str, Any], keys: tuple[str, ...]) -> Any:
        """Return the first non-empty value from a MinerU result payload."""
        for key in keys:
            value = file_result.get(key)
            if value:
                return value
        return None

    def download_zip_outputs(
        self,
        url: str,
        output_dir: Path,
        include_images: bool = False,
    ) -> tuple[str, list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Download MinerU full zip and extract Markdown plus content_list."""
        try:
            zip_bytes = self.download_bytes(url, timeout=180)
            zip_buffer = io.BytesIO(zip_bytes)
            if not zipfile.is_zipfile(zip_buffer):
                logger.error(
                    "MinerU result is not a valid zip, url=%s first_bytes=%r",
                    url,
                    zip_bytes[:200],
                )
                raise MinerUError(ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED)

            zip_buffer.seek(0)
            with zipfile.ZipFile(zip_buffer) as archive:
                names = archive.namelist()
                md_name = self.pick_archive_member(names, "full.md", ".md")
                if not md_name:
                    raise MinerUError(ErrorCode.MINERU_NO_MARKDOWN)

                markdown = archive.read(md_name).decode("utf-8", errors="replace")
                images = self.extract_archive_images(archive, output_dir) if include_images else []

                content_list = None
                content_name = self.pick_archive_member(names, "content_list.json", None)
                if content_name:
                    loaded_content = json.loads(
                        archive.read(content_name).decode("utf-8", errors="replace")
                    )
                    if isinstance(loaded_content, dict):
                        loaded_content = (
                            loaded_content.get("content_list")
                            or loaded_content.get("data")
                            or loaded_content.get("items")
                        )
                    if isinstance(loaded_content, list):
                        content_list = loaded_content

                return markdown, content_list, images
        except MinerUError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile, requests.RequestException) as exc:
            logger.exception("MinerU result zip download failed")
            raise MinerUError(ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED) from exc

    def extract_archive_images(
        self,
        archive: zipfile.ZipFile,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Extract image artifacts from a MinerU zip into a dedicated folder."""
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images: list[dict[str, Any]] = []
        used_names: set[str] = set()
        for index, member in enumerate(archive.namelist(), start=1):
            if member.endswith("/"):
                continue
            suffix = Path(member).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue

            original_name = Path(member).name or f"image_{index}{suffix}"
            safe_name = self.unique_image_name(original_name, used_names)
            image_path = images_dir / safe_name
            image_path.write_bytes(archive.read(member))
            mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            images.append(
                {
                    "label": f"图{len(images) + 1}",
                    "archive_name": member,
                    "file_name": safe_name,
                    "path": str(image_path.resolve()),
                    "mime_type": mime_type,
                }
            )

        return images

    def normalize_visual_outputs(
        self,
        markdown: str,
        content_list: list[dict[str, Any]],
        archive_images: list[dict[str, Any]],
        output_dir: Path,
        pdf_path: Path | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Rebuild visual outputs from content_list order and PDF bboxes."""
        visual_blocks = self.build_visual_blocks(content_list)
        if not visual_blocks:
            if archive_images:
                return self.replace_markdown_images_with_links(markdown, archive_images), archive_images
            return markdown, []

        archive_by_ref = self.build_archive_image_lookup(archive_images)
        merged_blocks = self.merge_visual_blocks(visual_blocks, content_list)
        normalized_images = self.materialize_visual_blocks(
            merged_blocks,
            archive_by_ref,
            output_dir,
            pdf_path,
            content_list,
        )
        if not normalized_images:
            if archive_images:
                return self.replace_markdown_images_with_links(markdown, archive_images), archive_images
            return markdown, []

        markdown = self.replace_markdown_images_with_links(markdown, normalized_images)
        markdown = self.replace_equations_with_visual_links(markdown, normalized_images)
        return markdown, normalized_images

    def build_visual_blocks(self, content_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build ordered image/table/equation blocks from MinerU content_list."""
        blocks: list[dict[str, Any]] = []
        seen_sources: set[str] = set()

        for order, item in enumerate(content_list):
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type") or "").strip().lower()
            if item_type not in VISUAL_TYPES:
                continue

            bbox = self.parse_bbox(item.get("bbox"))
            if bbox is None:
                continue

            source_paths = self.item_source_paths(item)
            unique_sources = []
            for source_path in source_paths:
                if source_path in seen_sources:
                    continue
                unique_sources.append(source_path)
                seen_sources.add(source_path)

            if source_paths and not unique_sources:
                continue

            page_idx = self.safe_int(item.get("page_idx"), default=0)
            blocks.append(
                {
                    "kind": item_type,
                    "order": order,
                    "page_idx": max(0, page_idx),
                    "bbox": bbox,
                    "source_paths": unique_sources,
                    "caption": self.item_caption(item),
                    "table_body": self.item_table_body(item),
                }
            )

        return sorted(blocks, key=self.visual_sort_key)

    def item_source_paths(self, item: dict[str, Any]) -> list[str]:
        """Return normalized source image paths referenced by one content item."""
        sources: list[str] = []
        for key in ("img_path", "image_path", "path"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                sources.append(self.normalize_content_path(value))
        return list(dict.fromkeys(source for source in sources if source))

    def item_caption(self, item: dict[str, Any]) -> str:
        """Return the first caption-like text for a visual item."""
        for key in ("image_caption", "table_caption", "caption"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
            if isinstance(value, list):
                joined = " ".join(str(part).strip() for part in value if str(part).strip())
                if joined:
                    return " ".join(joined.split())
        return ""

    def item_table_body(self, item: dict[str, Any]) -> str:
        """Return structured table content when MinerU provides it."""
        for key in ("table_body", "table_html", "html"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def merge_visual_blocks(
        self,
        blocks: list[dict[str, Any]],
        content_list: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Merge visual blocks into page-local figure clusters and assign labels."""
        rows = self.build_visual_rows(blocks)
        text_blocks = self.build_visual_text_blocks(content_list or [])
        self.attach_text_bboxes_to_visual_rows(rows, text_blocks)
        clusters = self.build_visual_clusters(rows)

        merged: list[dict[str, Any]] = []
        for cluster_rows in clusters:
            cluster_blocks = [
                block
                for row in cluster_rows
                for block in row.get("blocks", [])
            ]
            source_kinds = self.unique_visual_kinds(cluster_blocks)
            kind = self.primary_visual_kind(source_kinds)
            captions = [
                str(block.get("caption") or "")
                for block in cluster_blocks
                if block.get("caption")
            ]
            table_bodies = [
                str(block.get("table_body") or "")
                for block in cluster_blocks
                if block.get("kind") == "table" and block.get("table_body")
            ]
            merged.append(
                {
                    "kind": kind,
                    "order": min(int(row.get("order", 0)) for row in cluster_rows),
                    "page_idx": int(cluster_rows[0]["page_idx"]),
                    "bbox": self.union_bboxes(
                        [row.get("bbox_with_text") or row["bbox"] for row in cluster_rows]
                    ),
                    "source_paths": self.merged_source_paths(cluster_blocks),
                    "caption": captions[0] if captions else "",
                    "table_body": table_bodies[0] if kind == "table" and table_bodies else "",
                    "source_kinds": source_kinds,
                    "contains_equation": "equation" in source_kinds,
                }
            )

        counters = {kind: 0 for kind in VISUAL_TYPES}
        labeled: list[dict[str, Any]] = []
        for block in sorted(merged, key=self.visual_sort_key):
            kind = str(block["kind"])
            counters[kind] += 1
            label = f"{VISUAL_LABEL_PREFIX.get(kind, '图')}{counters[kind]}"
            block["label"] = label
            block["visual_id"] = f"{kind}_{counters[kind]:04d}"
            labeled.append(block)

        return labeled

    def build_visual_rows(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group same-page visual blocks into horizontal visual rows."""
        page_blocks: dict[int, list[dict[str, Any]]] = {}
        for block in sorted(blocks, key=self.visual_sort_key):
            page_blocks.setdefault(int(block.get("page_idx") or 0), []).append(block)

        rows: list[dict[str, Any]] = []
        for page_idx, current_blocks in page_blocks.items():
            parent = list(range(len(current_blocks)))

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(left: int, right: int) -> None:
                root_left = find(left)
                root_right = find(right)
                if root_left != root_right:
                    parent[root_right] = root_left

            for left_index, left in enumerate(current_blocks):
                for right_index in range(left_index + 1, len(current_blocks)):
                    right = current_blocks[right_index]
                    if self.visual_blocks_share_row(left["bbox"], right["bbox"]):
                        union(left_index, right_index)

            grouped: dict[int, list[dict[str, Any]]] = {}
            for index, block in enumerate(current_blocks):
                grouped.setdefault(find(index), []).append(block)

            for row_blocks in grouped.values():
                row_blocks = sorted(row_blocks, key=self.visual_sort_key)
                bbox = self.union_bboxes([block["bbox"] for block in row_blocks])
                rows.append(
                    {
                        "page_idx": page_idx,
                        "order": min(int(block.get("order", 0)) for block in row_blocks),
                        "bbox": bbox,
                        "bbox_with_text": bbox,
                        "blocks": row_blocks,
                    }
                )

        return sorted(rows, key=self.visual_sort_key)

    def build_visual_clusters(self, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group adjacent visual rows into larger figure clusters."""
        if not rows:
            return []

        parent = list(range(len(rows)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for left_index, left in enumerate(rows):
            for right_index in range(left_index + 1, len(rows)):
                right = rows[right_index]
                if int(left.get("page_idx") or 0) != int(right.get("page_idx") or 0):
                    continue
                if self.visual_rows_should_cluster(left["bbox"], right["bbox"]):
                    union(left_index, right_index)

        grouped: dict[int, list[dict[str, Any]]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(find(index), []).append(row)

        clusters = [
            sorted(cluster_rows, key=self.visual_sort_key)
            for cluster_rows in grouped.values()
        ]
        return sorted(
            clusters,
            key=lambda cluster_rows: self.visual_sort_key(cluster_rows[0]),
        )

    def build_visual_text_blocks(
        self,
        content_list: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, Any]]]:
        """Return text bboxes that can expand nearby visual rows."""
        text_blocks: dict[int, list[dict[str, Any]]] = {}
        for order, item in enumerate(content_list):
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() != "text":
                continue
            bbox = self.parse_bbox(item.get("bbox"))
            if bbox is None:
                continue
            text = self.item_text(item)
            if not text:
                continue
            page_idx = max(0, self.safe_int(item.get("page_idx"), default=0))
            text_blocks.setdefault(page_idx, []).append(
                {
                    "order": order,
                    "page_idx": page_idx,
                    "bbox": bbox,
                    "text": text,
                }
            )

        for page_idx, blocks in text_blocks.items():
            text_blocks[page_idx] = sorted(blocks, key=self.visual_sort_key)
        return text_blocks

    def attach_text_bboxes_to_visual_rows(
        self,
        rows: list[dict[str, Any]],
        text_blocks: dict[int, list[dict[str, Any]]],
    ) -> None:
        """Expand each visual row with nearby caption-like text bboxes."""
        for row in rows:
            page_idx = int(row.get("page_idx") or 0)
            captions = [
                str(block.get("caption") or "")
                for block in row.get("blocks", [])
                if block.get("caption")
            ]
            attached_bboxes: list[list[float]] = []
            for text_block in text_blocks.get(page_idx, []):
                text_bbox = text_block["bbox"]
                if self.text_matches_visual_caption(text_block["text"], captions):
                    attached_bboxes.append(text_bbox)
                    continue
                if self.text_block_near_visual_row(row["bbox"], text_bbox):
                    attached_bboxes.append(text_bbox)

            if attached_bboxes:
                row["text_bboxes"] = attached_bboxes
                row["bbox_with_text"] = self.union_bboxes([row["bbox"], *attached_bboxes])

    def visual_blocks_share_row(self, left: list[float], right: list[float]) -> bool:
        """Return whether two visual bboxes belong to the same horizontal row."""
        left_height = max(0.0, left[3] - left[1])
        right_height = max(0.0, right[3] - right[1])
        if not left_height or not right_height:
            return False
        vertical_overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        return (
            vertical_overlap / min(left_height, right_height)
            >= VISUAL_ROW_OVERLAP_THRESHOLD
        )

    def visual_rows_should_cluster(self, left: list[float], right: list[float]) -> bool:
        """Return whether adjacent visual rows should be one figure cluster."""
        if self.vertical_gap(left, right) > VISUAL_CLUSTER_VERTICAL_GAP:
            return False
        return self.bboxes_horizontally_related(
            left,
            right,
            VISUAL_CLUSTER_HORIZONTAL_OVERLAP_THRESHOLD,
        )

    def text_block_near_visual_row(self, row_bbox: list[float], text_bbox: list[float]) -> bool:
        """Return whether a text bbox is close enough to expand one visual row."""
        if not self.bbox_is_above_or_below(row_bbox, text_bbox, VISUAL_CAPTION_VERTICAL_GAP):
            return False
        return self.bboxes_horizontally_related(
            row_bbox,
            text_bbox,
            VISUAL_CAPTION_HORIZONTAL_OVERLAP_THRESHOLD,
        )

    def item_text(self, item: dict[str, Any]) -> str:
        """Return normalized text content for one content_list item."""
        for key in ("text", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
            if isinstance(value, list):
                joined = " ".join(str(part).strip() for part in value if str(part).strip())
                if joined:
                    return " ".join(joined.split())
        return ""

    def text_matches_visual_caption(self, text: str, captions: list[str]) -> bool:
        """Return whether text appears to be an explicit visual caption."""
        text_key = self.normalize_text_for_match(text)
        if len(text_key) < 12:
            return False
        for caption in captions:
            caption_key = self.normalize_text_for_match(caption)
            if len(caption_key) < 12:
                continue
            if caption_key in text_key or text_key in caption_key:
                return True
        return False

    def normalize_text_for_match(self, text: str) -> str:
        """Normalize text for fuzzy caption matching."""
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    def bbox_is_above_or_below(
        self,
        anchor: list[float],
        candidate: list[float],
        max_gap: float,
    ) -> bool:
        """Return whether candidate sits just above or below anchor."""
        if candidate[3] <= anchor[1]:
            return anchor[1] - candidate[3] <= max_gap
        if candidate[1] >= anchor[3]:
            return candidate[1] - anchor[3] <= max_gap
        return False

    def bboxes_horizontally_related(
        self,
        left: list[float],
        right: list[float],
        overlap_threshold: float,
    ) -> bool:
        """Return whether two bboxes overlap or align horizontally."""
        left_width = max(0.0, left[2] - left[0])
        right_width = max(0.0, right[2] - right[0])
        if not left_width or not right_width:
            return False

        horizontal_overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        if horizontal_overlap / min(left_width, right_width) >= overlap_threshold:
            return True

        left_center = (left[0] + left[2]) / 2.0
        right_center = (right[0] + right[2]) / 2.0
        return (
            right[0] <= left_center <= right[2]
            or left[0] <= right_center <= left[2]
        )

    def vertical_gap(self, left: list[float], right: list[float]) -> float:
        """Return the vertical gap between two bboxes."""
        if left[3] < right[1]:
            return right[1] - left[3]
        if right[3] < left[1]:
            return left[1] - right[3]
        return 0.0

    def merged_source_paths(self, blocks: list[dict[str, Any]]) -> list[str]:
        """Return source image paths from several visual blocks, preserving order."""
        source_paths: list[str] = []
        for block in blocks:
            for source_path in block.get("source_paths") or []:
                if source_path not in source_paths:
                    source_paths.append(source_path)
        return source_paths

    def unique_visual_kinds(self, blocks: list[dict[str, Any]]) -> list[str]:
        """Return visual kinds present in a cluster, sorted by display priority."""
        kinds: list[str] = []
        for block in blocks:
            kind = str(block.get("kind") or "")
            if kind in VISUAL_TYPES and kind not in kinds:
                kinds.append(kind)
        return sorted(kinds or ["image"], key=lambda kind: VISUAL_KIND_PRIORITY.get(kind, 99))

    def primary_visual_kind(self, kinds: list[str]) -> str:
        """Return the display kind for one visual cluster."""
        return min(kinds or ["image"], key=lambda kind: VISUAL_KIND_PRIORITY.get(kind, 99))

    def materialize_visual_blocks(
        self,
        blocks: list[dict[str, Any]],
        archive_by_ref: dict[str, dict[str, Any]],
        output_dir: Path,
        pdf_path: Path | None,
        content_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Render normalized visual blocks to files, falling back to MinerU images."""
        normalized_dir = output_dir / "images" / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        page_units = self.infer_page_units(content_list)
        document = self.open_pdf_document(pdf_path)
        images: list[dict[str, Any]] = []

        try:
            for block in blocks:
                output_path = normalized_dir / f"{block['visual_id']}.png"
                path = self.crop_visual_block(document, block, output_path, page_units)
                if path is None:
                    fallback = self.fallback_archive_image(block, archive_by_ref)
                    if fallback is None:
                        continue
                    path = Path(str(fallback["path"]))

                mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
                image = {
                    "label": block["label"],
                    "archive_name": block.get("source_paths", [""])[0] if block.get("source_paths") else "",
                    "file_name": path.name,
                    "path": str(path.resolve()),
                    "mime_type": mime_type,
                    "kind": block["kind"],
                    "page_idx": block["page_idx"],
                    "bbox": block["bbox"],
                    "source_paths": block.get("source_paths", []),
                    "visual_id": block["visual_id"],
                    "caption": block.get("caption", ""),
                    "table_body": block.get("table_body", ""),
                    "source_kinds": block.get("source_kinds", [block["kind"]]),
                    "contains_equation": bool(block.get("contains_equation")),
                }
                images.append(image)
        finally:
            if document is not None:
                document.close()

        return images

    def crop_visual_block(
        self,
        document: Any,
        block: dict[str, Any],
        output_path: Path,
        page_units: dict[int, tuple[float, float]],
    ) -> Path | None:
        """Crop one content_list bbox from the original PDF."""
        if document is None:
            return None

        page_idx = int(block.get("page_idx") or 0)
        if page_idx < 0 or page_idx >= len(document):
            return None

        page = document[page_idx]
        unit_width, unit_height = page_units.get(page_idx, (0.0, 0.0))
        if unit_width <= 0 or unit_height <= 0:
            return None

        page_rect = page.rect
        bbox = self.expand_bbox(block["bbox"], VISUAL_CROP_MARGIN)
        clip = self.bbox_to_pdf_rect(bbox, page_rect, unit_width, unit_height)
        clip = clip & page_rect
        if clip.is_empty or clip.width <= 1 or clip.height <= 1:
            return None

        pixmap = page.get_pixmap(matrix=self.fitz_matrix(2.0), clip=clip, alpha=False)
        pixmap.save(str(output_path))
        return output_path

    def open_pdf_document(self, pdf_path: Path | None) -> Any:
        """Open a PDF document for visual cropping when PyMuPDF is available."""
        if pdf_path is None or not pdf_path.exists():
            return None
        try:
            import fitz

            return fitz.open(pdf_path)
        except ModuleNotFoundError:
            logger.warning("PyMuPDF is not installed; visual cropping is skipped. Install with: pip install PyMuPDF")
        except Exception:
            logger.warning("Failed to open PDF for visual cropping; visual crop images are skipped.", exc_info=True)
            return None

    def fitz_matrix(self, zoom: float) -> Any:
        """Create a PyMuPDF matrix without importing fitz at module import time."""
        import fitz

        return fitz.Matrix(zoom, zoom)

    def bbox_to_pdf_rect(self, bbox: list[float], page_rect: Any, unit_width: float, unit_height: float) -> Any:
        """Convert a MinerU bbox into a PyMuPDF page rect."""
        import fitz

        return fitz.Rect(
            page_rect.x0 + max(0.0, bbox[0] / unit_width * page_rect.width),
            page_rect.y0 + max(0.0, bbox[1] / unit_height * page_rect.height),
            page_rect.x0 + min(page_rect.width, bbox[2] / unit_width * page_rect.width),
            page_rect.y0 + min(page_rect.height, bbox[3] / unit_height * page_rect.height),
        )

    def fallback_archive_image(
        self,
        block: dict[str, Any],
        archive_by_ref: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the first raw MinerU image matching a normalized visual block."""
        for source_path in block.get("source_paths") or []:
            image = archive_by_ref.get(source_path) or archive_by_ref.get(Path(source_path).name)
            if image:
                return image
        return None

    def build_archive_image_lookup(self, images: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Index raw MinerU images by archive path and filename."""
        lookup: dict[str, dict[str, Any]] = {}
        for image in images:
            for key in ("archive_name", "file_name"):
                value = image.get(key)
                if isinstance(value, str) and value.strip():
                    normalized = self.normalize_content_path(value)
                    lookup[normalized] = image
                    lookup[Path(normalized).name] = image
        return lookup

    def infer_page_units(self, content_list: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
        """Infer MinerU page coordinate extents from all bboxes on each page."""
        units: dict[int, tuple[float, float]] = {}
        for item in content_list:
            if not isinstance(item, dict):
                continue
            bbox = self.parse_bbox(item.get("bbox"))
            if bbox is None:
                continue
            page_idx = max(0, self.safe_int(item.get("page_idx"), default=0))
            width, height = units.get(page_idx, (0.0, 0.0))
            units[page_idx] = (max(width, bbox[2]), max(height, bbox[3]))
        return {page: (max(width, 1.0), max(height, 1.0)) for page, (width, height) in units.items()}

    def replace_equations_with_visual_links(
        self,
        markdown: str,
        images: list[dict[str, Any]],
    ) -> str:
        """Replace display math blocks with equation images, preserving inline math."""
        equations = [
            image
            for image in images
            if image.get("kind") == "equation" or image.get("contains_equation")
        ]
        if not equations:
            return markdown

        equation_index = 0

        def replacement(match: re.Match[str]) -> str:
            nonlocal equation_index
            if equation_index >= len(equations):
                return match.group(0)
            image = equations[equation_index]
            equation_index += 1
            if image.get("kind") != "equation" and image.get("source_paths"):
                return ""
            return self.visual_link_markdown(image)

        return re.sub(r"\$\$\s*[\s\S]*?\s*\$\$", replacement, markdown)

    def visual_link_markdown(self, image: dict[str, Any], caption_override: str = "") -> str:
        """Return a Markdown link to one normalized visual image."""
        label = str(image.get("label") or "图")
        caption_text = caption_override or str(image.get("caption") or "")
        caption = f"：{caption_text}" if caption_text else ""
        return f"[此处含有{label}{caption}]({self.image_data_uri(image)})"

    def visual_image_is_table_only(self, image: dict[str, Any]) -> bool:
        """Return whether an extracted visual is a pure table screenshot."""
        kind = str(image.get("kind") or "").lower()
        source_kinds = {
            str(kind_value).lower()
            for kind_value in image.get("source_kinds") or []
            if str(kind_value).strip()
        }
        return (
            kind == "table"
            and "image" not in source_kinds
            and bool(str(image.get("table_body") or "").strip())
        )

    def parse_bbox(self, value: Any) -> list[float] | None:
        """Parse a bbox as [x0, y0, x1, y1]."""
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            x0, y0, x1, y1 = [float(part) for part in value]
        except (TypeError, ValueError):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return [x0, y0, x1, y1]

    def expand_bbox(self, bbox: list[float], margin: float) -> list[float]:
        """Expand a bbox by a small margin in MinerU coordinates."""
        return [bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin]

    def union_bboxes(self, bboxes: list[list[float]]) -> list[float]:
        """Return the union of several bboxes."""
        return [
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        ]

    def bbox_area(self, bbox: list[float]) -> float:
        """Return bbox area."""
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    def intersection_area(self, left: list[float], right: list[float]) -> float:
        """Return the intersection area of two bboxes."""
        width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        return width * height

    def visual_sort_key(self, block: dict[str, Any]) -> tuple[int, float, float, int]:
        """Sort visuals by page and page position."""
        bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        return (
            int(block.get("page_idx") or 0),
            float(bbox[1]),
            float(bbox[0]),
            int(block.get("order") or 0),
        )

    def normalize_content_path(self, value: str) -> str:
        """Normalize a MinerU archive/content path for matching."""
        return unquote(str(value or "").split()[0].strip("\"'").replace("\\", "/"))

    def safe_int(self, value: Any, default: int = 0) -> int:
        """Coerce a value to int."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def unique_image_name(self, file_name: str, used_names: set[str]) -> str:
        """Return a safe unique image filename."""
        clean_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name).strip("._")
        if not clean_name:
            clean_name = f"image_{len(used_names) + 1}.png"

        stem = Path(clean_name).stem
        suffix = Path(clean_name).suffix or ".png"
        candidate = f"{stem}{suffix}"
        counter = 2
        while candidate.lower() in used_names:
            candidate = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate.lower())
        return candidate

    def replace_markdown_images_with_links(
        self,
        markdown: str,
        images: list[dict[str, Any]],
    ) -> str:
        """Replace Markdown image tags with links to saved original images."""
        image_by_ref: dict[str, dict[str, Any]] = {}
        for image in images:
            source_paths = list(image.get("source_paths") or [])
            archive_name = str(image.get("archive_name") or "")
            if archive_name:
                source_paths.append(archive_name)
            file_name = str(image.get("file_name") or "")
            if file_name:
                source_paths.append(file_name)

            for source_path in source_paths:
                normalized = self.normalize_content_path(source_path)
                if not normalized:
                    continue
                image_by_ref[normalized] = image
                image_by_ref[Path(normalized).name] = image

        emitted_visuals: set[str] = set()

        def replacement(match: re.Match[str]) -> str:
            alt_text = match.group("alt").strip()
            raw_target = match.group("target").strip().strip("\"'")
            normalized_target = self.normalize_content_path(raw_target)
            image = image_by_ref.get(normalized_target) or image_by_ref.get(
                Path(normalized_target).name
            )
            if not image:
                label = f"图{len(emitted_visuals) + 1}"
                caption = f"：{alt_text}" if alt_text else ""
                return f"**此处含有{label}{caption}（图片文件未找到）**"

            if self.visual_image_is_table_only(image):
                return ""

            visual_id = str(image.get("visual_id") or image.get("path") or normalized_target)
            if visual_id in emitted_visuals:
                return ""
            emitted_visuals.add(visual_id)
            return self.visual_link_markdown(image, alt_text)

        return re.sub(
            r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)]+)\)",
            replacement,
            markdown,
        )

    def image_data_uri(self, image: dict[str, Any]) -> str:
        """Build a data URI for an extracted image."""
        image_path = Path(image["path"])
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{image['mime_type']};base64,{encoded}"

    def download_text(self, url: str) -> str:
        """Download a text artifact from MinerU."""
        try:
            return self.download_bytes(url, timeout=120).decode("utf-8", errors="replace")
        except MinerUError:
            raise
        except Exception as exc:
            logger.exception("MinerU Markdown download failed")
            raise MinerUError(ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED) from exc

    def download_content_list(self, url: str) -> list[dict[str, Any]] | None:
        """Download MinerU content_list JSON when a direct URL is available."""
        try:
            loaded_content = json.loads(
                self.download_bytes(url, timeout=120).decode("utf-8", errors="replace")
            )
            if isinstance(loaded_content, dict):
                loaded_content = (
                    loaded_content.get("content_list")
                    or loaded_content.get("data")
                    or loaded_content.get("items")
                )
            return loaded_content if isinstance(loaded_content, list) else None
        except Exception:
            logger.exception("MinerU content_list download failed")
            return None

    def download_bytes(self, url: str, timeout: int) -> bytes:
        """Download bytes with requests first, then curl as a Windows/CDN TLS fallback."""
        last_error: Exception | None = None

        for attempt in range(1, 4):
            try:
                response = self.session.get(
                    url,
                    headers=self.download_headers(),
                    stream=True,
                    timeout=(15, timeout),
                )
                response.raise_for_status()
                chunks = [
                    chunk
                    for chunk in response.iter_content(chunk_size=1024 * 1024)
                    if chunk
                ]
                return b"".join(chunks)
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "MinerU artifact download with requests failed, attempt=%s url=%s error=%s",
                    attempt,
                    url,
                    exc,
                )
                time.sleep(min(2 * attempt, 6))

        try:
            return self.download_bytes_with_curl(url, timeout)
        except MinerUError:
            raise
        except Exception as exc:
            logger.exception("MinerU artifact download failed after curl fallback")
            if last_error:
                raise MinerUError(ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED) from last_error
            raise MinerUError(ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED) from exc

    def download_bytes_with_curl(self, url: str, timeout: int) -> bytes:
        """Download bytes using the system curl binary when Python TLS fails."""
        curl_bin = shutil.which("curl.exe") or shutil.which("curl")
        if not curl_bin:
            raise MinerUError(ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED)

        command = [
            curl_bin,
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "30",
            "--max-time",
            str(timeout),
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--user-agent",
            "PaperMate/1.0",
            url,
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=timeout + 30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.exception("MinerU artifact download with curl failed")
            raise MinerUError(ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED) from exc

        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")[:1000]
            logger.error("MinerU artifact curl download failed: %s", stderr)
            raise MinerUError(ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED)

        if not completed.stdout:
            logger.error("MinerU artifact curl download returned empty body")
            raise MinerUError(ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED)

        logger.info("MinerU artifact downloaded with curl fallback: %s", url)
        return completed.stdout

    def pick_archive_member(
        self,
        names: list[str],
        preferred_suffix: str,
        fallback_suffix: str | None,
    ) -> str | None:
        """Pick a safe archive member by suffix without extracting arbitrary paths."""
        normalized = [name for name in names if not name.endswith("/")]

        for name in normalized:
            if name.lower().endswith(preferred_suffix.lower()):
                return name

        if fallback_suffix:
            for name in normalized:
                if name.lower().endswith(fallback_suffix.lower()):
                    return name

        return None

    def url(self, path: str) -> str:
        """Build a MinerU API URL."""
        return f"{self.base_url}{path}"

    def auth_headers(self) -> dict[str, str]:
        """Return Authorization headers for MinerU API calls."""
        return {"Authorization": f"Bearer {self.api_token}"}

    def json_headers(self) -> dict[str, str]:
        """Return JSON headers for MinerU API calls."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }

    def download_headers(self) -> dict[str, str]:
        """Return headers for MinerU artifact downloads."""
        return {
            "Accept": "application/zip,text/markdown,text/plain,*/*",
            "Connection": "close",
            "User-Agent": "PaperMate/1.0",
        }
