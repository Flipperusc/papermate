"""Custom exceptions and error codes for PaperMate."""

from __future__ import annotations

from enum import Enum
from typing import Any


class PaperMateError(Exception):
    """Base exception for PaperMate."""


class ConfigurationError(PaperMateError):
    """Raised when application configuration is invalid."""


class PipelineError(PaperMateError):
    """Raised when a processing pipeline fails."""


class ErrorCode(str, Enum):
    """User-facing error codes.

    New codes are defined first. Legacy names remain as aliases so existing
    modules can keep importing them while the RAG stack is refactored.
    """

    FILE_NOT_PDF = "FILE_NOT_PDF"
    FILE_SAVE_FAILED = "FILE_SAVE_FAILED"
    PDF_TO_MARKDOWN_FAILED = "PDF_TO_MARKDOWN_FAILED"
    MARKDOWN_EMPTY = "MARKDOWN_EMPTY"
    CHUNK_EMPTY = "CHUNK_EMPTY"
    DEEPSEEK_API_KEY_MISSING = "DEEPSEEK_API_KEY_MISSING"
    DEEPSEEK_LLM_FAILED = "DEEPSEEK_LLM_FAILED"
    EMBEDDING_API_KEY_MISSING = "EMBEDDING_API_KEY_MISSING"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    VECTOR_INDEX_FAILED = "VECTOR_INDEX_FAILED"
    VECTOR_SEARCH_FAILED = "VECTOR_SEARCH_FAILED"
    BM25_INDEX_MISSING = "BM25_INDEX_MISSING"
    BM25_INDEX_FAILED = "BM25_INDEX_FAILED"
    BM25_SEARCH_FAILED = "BM25_SEARCH_FAILED"
    RRF_FUSION_FAILED = "RRF_FUSION_FAILED"
    HYBRID_RETRIEVAL_EMPTY = "HYBRID_RETRIEVAL_EMPTY"
    DB_WRITE_FAILED = "DB_WRITE_FAILED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

    # Legacy aliases used by current modules.
    INVALID_FILE_TYPE = "FILE_NOT_PDF"
    EMPTY_FILE = "EMPTY_FILE"
    SAVE_FAILED = "FILE_SAVE_FAILED"
    PDF_PARSE_FAILED = "PDF_TO_MARKDOWN_FAILED"
    PDF_NO_TEXT = "MARKDOWN_EMPTY"
    MINERU_API_TOKEN_MISSING = "MINERU_API_TOKEN_MISSING"
    MINERU_UPLOAD_URL_FAILED = "MINERU_UPLOAD_URL_FAILED"
    MINERU_UPLOAD_FAILED = "MINERU_UPLOAD_FAILED"
    MINERU_PARSE_FAILED = "PDF_TO_MARKDOWN_FAILED"
    MINERU_PARSE_TIMEOUT = "MINERU_PARSE_TIMEOUT"
    MINERU_RESULT_DOWNLOAD_FAILED = "MINERU_RESULT_DOWNLOAD_FAILED"
    MINERU_NO_MARKDOWN = "MARKDOWN_EMPTY"
    EMBEDDING_CALL_FAILED = "EMBEDDING_FAILED"
    VECTOR_STORE_WRITE_FAILED = "VECTOR_INDEX_FAILED"
    VECTOR_STORE_SEARCH_FAILED = "VECTOR_SEARCH_FAILED"
    DEEPSEEK_CALL_FAILED = "DEEPSEEK_LLM_FAILED"


ERROR_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.FILE_NOT_PDF: "只支持上传 PDF 文件。",
    ErrorCode.EMPTY_FILE: "上传的文件为空，请重新选择 PDF 文件。",
    ErrorCode.FILE_SAVE_FAILED: "文件保存失败，请稍后重试。",
    ErrorCode.PDF_TO_MARKDOWN_FAILED: "PDF 转 Markdown 失败，请检查文件或解析服务后重试。",
    ErrorCode.MARKDOWN_EMPTY: "Markdown 内容为空，请确认 PDF 可解析后重试。",
    ErrorCode.CHUNK_EMPTY: "论文切分结果为空，请换一份可解析的 PDF 后重试。",
    ErrorCode.MINERU_API_TOKEN_MISSING: "MinerU API Token 未配置，请在 .env 中填写 MINERU_API_TOKEN。",
    ErrorCode.MINERU_UPLOAD_URL_FAILED: "MinerU 上传链接申请失败，请检查 API Token 或服务状态。",
    ErrorCode.MINERU_UPLOAD_FAILED: "PDF 上传到 MinerU 失败，请检查网络连接后重试。",
    ErrorCode.MINERU_PARSE_TIMEOUT: "MinerU PDF 转 Markdown 超时，请稍后重试。",
    ErrorCode.MINERU_RESULT_DOWNLOAD_FAILED: "MinerU 解析结果下载失败，请稍后重试。",
    ErrorCode.DEEPSEEK_API_KEY_MISSING: "DeepSeek API Key 未配置，请在 .env 中填写 DEEPSEEK_API_KEY。",
    ErrorCode.DEEPSEEK_LLM_FAILED: "DeepSeek 生成失败，请检查 API Key、模型名称、网络或账户额度。",
    ErrorCode.EMBEDDING_API_KEY_MISSING: "Embedding API Key 未配置，请在 .env 中填写 EMBEDDING_API_KEY。",
    ErrorCode.EMBEDDING_FAILED: "Embedding 调用失败，请检查模型、网络或服务配置。",
    ErrorCode.VECTOR_INDEX_FAILED: "向量索引写入失败，请检查 Chroma 目录权限。",
    ErrorCode.VECTOR_SEARCH_FAILED: "向量检索失败，请稍后重试。",
    ErrorCode.BM25_INDEX_MISSING: "BM25 索引不存在，请先构建论文索引。",
    ErrorCode.BM25_INDEX_FAILED: "BM25 索引构建失败，请稍后重试。",
    ErrorCode.BM25_SEARCH_FAILED: "BM25 检索失败，请稍后重试。",
    ErrorCode.RRF_FUSION_FAILED: "RRF 融合排序失败，请稍后重试。",
    ErrorCode.HYBRID_RETRIEVAL_EMPTY: "没有检索到可用于回答的论文片段。",
    ErrorCode.DB_WRITE_FAILED: "数据库写入失败，请检查 SQLite 文件权限。",
    ErrorCode.UNKNOWN_ERROR: "系统发生未知错误，请稍后重试。",
}


class AppError(PaperMateError):
    """Application error with a safe user-facing message."""

    def __init__(
        self,
        code: ErrorCode | str,
        user_message: str | None = None,
        detail: str | None = None,
        recoverable: bool = True,
    ) -> None:
        self.code = coerce_error_code(code)
        self.user_message = user_message or ERROR_MESSAGES.get(
            self.code,
            ERROR_MESSAGES[ErrorCode.UNKNOWN_ERROR],
        )
        self.message = self.user_message
        self.detail = detail
        self.recoverable = recoverable

        error_text = f"{self.code.value}: {self.user_message}"
        if detail:
            error_text = f"{error_text} Detail: {detail}"
        super().__init__(error_text)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation without stack information."""
        return {
            "code": self.code.value,
            "user_message": self.user_message,
            "detail": self.detail,
            "recoverable": self.recoverable,
        }


def coerce_error_code(code: ErrorCode | str) -> ErrorCode:
    """Coerce enum values, enum names, and unknown strings to an ErrorCode."""
    if isinstance(code, ErrorCode):
        return code
    code_text = str(code)
    if code_text in ErrorCode.__members__:
        return ErrorCode[code_text]
    try:
        return ErrorCode(code_text)
    except ValueError:
        return ErrorCode.UNKNOWN_ERROR


class CodedPaperMateError(AppError):
    """Backward-compatible base exception with a user-facing code."""

    def __init__(
        self,
        code: ErrorCode,
        detail: str | None = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(code=code, detail=detail, recoverable=recoverable)


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
