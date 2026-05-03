"""Custom exceptions and error codes for PaperMate."""

from __future__ import annotations

from enum import Enum


class PaperMateError(Exception):
    """Base exception for PaperMate."""


class ConfigurationError(PaperMateError):
    """Raised when application configuration is invalid."""


class PipelineError(PaperMateError):
    """Raised when a processing pipeline fails."""


class ErrorCode(str, Enum):
    """User-facing error codes."""

    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    EMPTY_FILE = "EMPTY_FILE"
    SAVE_FAILED = "SAVE_FAILED"
    PDF_PARSE_FAILED = "PDF_PARSE_FAILED"
    PDF_NO_TEXT = "PDF_NO_TEXT"
    MINERU_API_TOKEN_MISSING = "MINERU_API_TOKEN_MISSING"
    MINERU_UPLOAD_URL_FAILED = "MINERU_UPLOAD_URL_FAILED"
    MINERU_UPLOAD_FAILED = "MINERU_UPLOAD_FAILED"
    MINERU_PARSE_FAILED = "MINERU_PARSE_FAILED"
    MINERU_PARSE_TIMEOUT = "MINERU_PARSE_TIMEOUT"
    MINERU_RESULT_DOWNLOAD_FAILED = "MINERU_RESULT_DOWNLOAD_FAILED"
    MINERU_NO_MARKDOWN = "MINERU_NO_MARKDOWN"
    EMBEDDING_API_KEY_MISSING = "EMBEDDING_API_KEY_MISSING"
    EMBEDDING_CALL_FAILED = "EMBEDDING_CALL_FAILED"
    VECTOR_STORE_WRITE_FAILED = "VECTOR_STORE_WRITE_FAILED"
    VECTOR_STORE_SEARCH_FAILED = "VECTOR_STORE_SEARCH_FAILED"
    DEEPSEEK_API_KEY_MISSING = "DEEPSEEK_API_KEY_MISSING"
    DEEPSEEK_CALL_FAILED = "DEEPSEEK_CALL_FAILED"


ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_FILE_TYPE: "只支持上传 .pdf 文件。",
    ErrorCode.EMPTY_FILE: "上传的文件为空，请选择有效的 PDF 文件。",
    ErrorCode.SAVE_FAILED: "文件保存失败，请稍后重试。",
    ErrorCode.PDF_PARSE_FAILED: "PDF 解析失败，请确认文件未损坏后重试。",
    ErrorCode.PDF_NO_TEXT: "没有读取到有效文字，可能是扫描版 PDF，当前版本暂不支持 OCR。",
    ErrorCode.MINERU_API_TOKEN_MISSING: "MinerU API Token 未配置，请在 .env 文件中填写 MINERU_API_TOKEN。",
    ErrorCode.MINERU_UPLOAD_URL_FAILED: "MinerU 上传链接申请失败，请检查 API Token 或服务状态。",
    ErrorCode.MINERU_UPLOAD_FAILED: "PDF 上传到 MinerU 失败，请检查网络连接后重试。",
    ErrorCode.MINERU_PARSE_FAILED: "MinerU PDF 转 Markdown 失败，请检查文件内容、API Token 或账户额度。",
    ErrorCode.MINERU_PARSE_TIMEOUT: "MinerU PDF 转 Markdown 超时，请稍后重试或调大 MINERU_POLL_TIMEOUT。",
    ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED: "MinerU 解析结果下载失败，请稍后重试。",
    ErrorCode.MINERU_NO_MARKDOWN: "MinerU 解析完成，但没有找到 Markdown 结果。",
    ErrorCode.EMBEDDING_API_KEY_MISSING: "Embedding API Key 缺失，请在 .env 中配置 EMBEDDING_API_KEY。",
    ErrorCode.EMBEDDING_CALL_FAILED: "Embedding 调用失败，请检查模型、网络或服务配置。",
    ErrorCode.VECTOR_STORE_WRITE_FAILED: "Chroma 写入失败，请检查向量数据库目录权限。",
    ErrorCode.VECTOR_STORE_SEARCH_FAILED: "Chroma 检索失败，请稍后重试。",
    ErrorCode.DEEPSEEK_API_KEY_MISSING: "DeepSeek API Key 未配置，请检查 .env 文件中的 DEEPSEEK_API_KEY。",
    ErrorCode.DEEPSEEK_CALL_FAILED: "DeepSeek 模型调用失败，请检查 API Key、模型名称、网络连接或账户额度。",
}


class CodedPaperMateError(PaperMateError):
    """Base exception with a user-facing error code and message."""

    def __init__(self, code: ErrorCode, detail: str | None = None) -> None:
        self.code = code
        self.message = ERROR_MESSAGES[code]
        self.detail = detail

        error_text = f"{code.value}: {self.message}"
        if detail:
            error_text = f"{error_text} {detail}"
        super().__init__(error_text)


class UploadError(CodedPaperMateError):
    """Raised when an uploaded file cannot be accepted or saved."""


class PdfParseError(CodedPaperMateError):
    """Raised when PDF text cannot be extracted."""


class MinerUError(CodedPaperMateError):
    """Raised when MinerU PDF-to-Markdown conversion fails."""


class EmbeddingError(CodedPaperMateError):
    """Raised when embedding generation fails."""


class VectorStoreError(CodedPaperMateError):
    """Raised when vector storage or search fails."""


class LLMError(CodedPaperMateError):
    """Raised when language model generation fails."""
