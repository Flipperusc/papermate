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
    ) -> dict[str, Any]:
        """Upload a local PDF to MinerU and save the returned Markdown."""
        if not self.api_token:
            raise MinerUError(ErrorCode.MINERU_API_TOKEN_MISSING)

        path = Path(file_path)
        upload_name = file_name or path.name

        batch_id, upload_url = self.request_upload_url(upload_name, paper_id)
        self.upload_file(upload_url, path)
        file_result = self.wait_for_result(batch_id, upload_name, paper_id)
        outputs = self.download_outputs(file_result, paper_id)

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

    def download_outputs(self, file_result: dict[str, Any], paper_id: str) -> dict[str, Any]:
        """Download Markdown and optional content_list from MinerU result URLs."""
        output_dir = settings.mineru_output_dir / paper_id
        output_dir.mkdir(parents=True, exist_ok=True)

        markdown = ""
        content_list: list[dict[str, Any]] | None = None

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
                markdown, content_list, images = self.download_zip_outputs(str(zip_url), output_dir)
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

        if images:
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
    ) -> tuple[str, list[dict[str, Any]] | None, list[dict[str, str]]]:
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
                images = self.extract_archive_images(archive, output_dir)

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
    ) -> list[dict[str, str]]:
        """Extract image artifacts from a MinerU zip into a dedicated folder."""
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images: list[dict[str, str]] = []
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
        images: list[dict[str, str]],
    ) -> str:
        """Replace Markdown image tags with links to saved original images."""
        image_by_ref: dict[str, dict[str, str]] = {}
        for image in images:
            archive_name = image["archive_name"].replace("\\", "/")
            image_by_ref[archive_name] = image
            image_by_ref[Path(archive_name).name] = image

        figure_index = 0

        def replacement(match: re.Match[str]) -> str:
            nonlocal figure_index
            alt_text = match.group("alt").strip()
            raw_target = match.group("target").strip().strip("\"'")
            normalized_target = unquote(raw_target.split()[0].strip("\"'").replace("\\", "/"))
            image = image_by_ref.get(normalized_target) or image_by_ref.get(
                Path(normalized_target).name
            )
            figure_index += 1
            label = image["label"] if image else f"图{figure_index}"
            caption = f"：{alt_text}" if alt_text else ""
            if not image:
                return f"**此处含有{label}{caption}（图片文件未找到）**"
            return f"[此处含有{label}{caption}]({self.image_data_uri(image)})"

        return re.sub(
            r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)]+)\)",
            replacement,
            markdown,
        )

    def image_data_uri(self, image: dict[str, str]) -> str:
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
