# PaperMate

PaperMate 是一个面向论文阅读的本地 RAG 助手。当前版本支持 PDF 上传、MinerU PDF 转 Markdown、图片抽取、SQLite 数据保存、Chroma 向量索引、BM25 关键词索引、Hybrid Retrieval + RRF 问答、引用来源展示、用户反馈、Bad Case 记录、文献卡片生成与管理，以及 Docker 部署。

## 核心能力

- PDF 上传与解析：默认使用 MinerU 转 Markdown，并保留图片和页码信息。
- PyMuPDF fallback：可临时切换到本地 PDF 文本解析。
- 文本分块：将论文内容切成 chunks，并保存到 SQLite。
- Hybrid RAG：同时使用 Chroma 向量检索和 BM25 关键词检索，再用 RRF 融合排序。
- Query Processor：根据问题类型调整向量检索和关键词检索权重，并扩展常见论文关键词。
- Context Builder：由系统根据 chunk metadata 构造参考片段和 citations，避免模型编造引用。
- DeepSeek 生成：DeepSeek 只负责最终回答，不参与 embedding。
- 反馈闭环：回答下方可提交反馈，负面反馈会进入 Bad Case。
- 文献卡片：支持生成、保存、编辑、删除和查看已保存卡片。

## 目录结构

```text
papermate/
  app.py
  config.py
  requirements.txt
  .env.example
  README.md
  DEPLOYMENT.md
  data/
    uploads/
    chroma_db/
    bm25/
    mineru_outputs/
  logs/
    app.log
  scripts/
    init_db.py
    test_deepseek.py
    test_query_processor.py
    test_rrf.py
    test_context_builder.py
    test_hybrid_retriever.py
  src/
    db.py
    chunker.py
    embedding_client.py
    vector_store.py
    llm_client.py
    rag_pipeline.py
    feedback_service.py
    literature_card_service.py
    retrieval/
      tokenizer.py
      bm25_store.py
      query_processor.py
      rrf.py
      vector_retriever.py
      hybrid_retriever.py
      context_builder.py
```

## 快速开始

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts\init_db.py
streamlit run app.py
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/init_db.py
streamlit run app.py
```

启动后在浏览器打开 Streamlit 提示的本地地址，上传 PDF 后即可解析、构建索引并提问。

## 环境配置

复制 `.env.example` 为 `.env` 后，至少需要配置：

```env
PAPERMATE_APP_PASSWORD=your_password

MINERU_API_TOKEN=your_mineru_api_token

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
```

注意：DeepSeek 只用于最终回答，Embedding 仍走 `src/embedding_client.py` 中的 OpenAI-compatible Embedding API。不要把 DeepSeek 配成 Embedding 服务。

## MinerU 配置

默认 PDF 解析方式是 MinerU：

```env
PAPERMATE_PDF_PARSE_PROVIDER=mineru
MINERU_API_TOKEN=your_mineru_api_token
MINERU_BASE_URL=https://mineru.net
MINERU_MODEL_VERSION=vlm
MINERU_IS_OCR=true
MINERU_ENABLE_FORMULA=true
MINERU_ENABLE_TABLE=true
MINERU_LANGUAGE=en
MINERU_POLL_INTERVAL=3
MINERU_POLL_TIMEOUT=600
MINERU_OUTPUT_DIR=data/mineru_outputs
```

如果需要临时切换到本地 PyMuPDF 解析：

```env
PAPERMATE_PDF_PARSE_PROVIDER=pymupdf
```

## Hybrid RAG 配置

检索相关配置：

```env
CHROMA_DIR=data/chroma_db
BM25_DIR=data/bm25
VECTOR_TOP_K=20
BM25_TOP_K=20
FINAL_TOP_K=6
RRF_K=60
CONTEXT_MAX_CHARS=6000
```

问答链路：

```text
用户问题
  -> Query Processor 判断问题类型并扩展关键词
  -> Chroma Vector Search
  -> SQLite chunks 构建的 BM25 Search
  -> RRF 融合排序
  -> Context Builder 构造参考片段和 citations
  -> DeepSeek 生成回答
  -> 页面展示答案、引用来源和 Hybrid 检索细节
```

构建索引时，点击“构建论文索引（向量 + 关键词）”。成功后会同时写入：

- Chroma：`data/chroma_db/`
- BM25：`data/bm25/{paper_id}_bm25.pkl`
- BM25 payload：`data/bm25/{paper_id}_payloads.json`

如果 BM25 构建失败但 Chroma 成功，仍可使用语义检索；如果 Chroma 构建失败但 BM25 成功，仍可使用关键词检索。

## 使用流程

1. 上传 PDF。
2. PaperMate 调用 MinerU 将 PDF 转为 Markdown，并展示完整 Markdown。
3. Markdown 和图片保存到 `data/mineru_outputs/<paper_id>/`。
4. 系统将论文切分为 chunks，并写入 SQLite。
5. 点击“构建论文索引（向量 + 关键词）”。
6. 在“论文问答”中提问。
7. 回答区域会展示答案、引用来源、原文片段和 Hybrid 检索细节。
8. 在回答下方提交反馈；负面反馈会自动记录为 Bad Case。
9. 可生成并保存文献卡片，在“文献卡片库”页面管理。
10. 可在“反馈记录”页面查看用户反馈和 Bad Case。

## 验证脚本

基础检查：

```bash
python -m compileall .
```

模块测试：

```bash
python scripts/test_query_processor.py
python scripts/test_rrf.py
python scripts/test_context_builder.py
python scripts/test_hybrid_retriever.py
```

DeepSeek 连通性测试：

```bash
python scripts/test_deepseek.py
```

## Docker 部署

项目包含 Docker 部署文件：

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.streamlit/config.toml`
- `DEPLOYMENT.md`

推荐使用 Docker Compose：

```bash
cp .env.example .env
# 编辑 .env，填写 PAPERMATE_APP_PASSWORD、MINERU_API_TOKEN、DEEPSEEK_API_KEY、EMBEDDING_API_KEY
docker compose up -d --build
```

详细步骤见 `DEPLOYMENT.md`。

## 反馈记录

启动应用后，在侧边栏进入“反馈记录”页面，可以查看：

- 用户反馈列表
- 负面反馈数量
- Bad Case 列表
- 关联论文、问题、回答、反馈类型和补充说明

服务器上也可以直接查询：

```bash
docker compose exec papermate python -c "from src.feedback_service import list_feedback_records, list_bad_cases; print(list_feedback_records(20)); print(list_bad_cases(20))"
```

## 日志

底层异常会写入：

```text
logs/app.log
```

页面上只展示中文、简短、可操作的错误提示。
