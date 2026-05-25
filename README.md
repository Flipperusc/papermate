# PaperMate

PaperMate 是一个本地运行的论文阅读 RAG 助手。当前版本 `0.6.1` 基于 Streamlit，支持 PDF 上传、MinerU 转 Markdown、表格/公式归一化、可选图片解析、中文 Markdown 翻译、双语对照阅读、Chroma 向量检索、BM25 关键词检索、Hybrid Retrieval + RRF 问答、可信引用展示、用户反馈、Bad Case 归档，以及按用户管理的文献卡片库。

## 核心能力

- 本地账号：启动后先注册/登录本地用户，用户数据保存在 SQLite。
- PDF 解析：默认调用 MinerU 将 PDF 转为 Markdown，并保存 `full.md` 和 `content_list.json`；图片资源需在页面中选择“添加图片并重新解析”后才会纳入。
- 本地解析兜底：可切换到 PyMuPDF，只做文本抽取，不包含 MinerU 的 Markdown 和图片能力。
- 阅读器：支持原文、中文译文、双语对照三种阅读模式；中文译文可下载，不会覆盖原文。
- 文本分块：按语义相似度做 sentence-level chunking，并保留页码、章节和表格 metadata，写入 SQLite。
- 图片理解：默认不启用；用户选择添加图片并重新解析后，图片绑定到相邻文本 chunk 时会调用阿里云百炼 OpenAI 兼容接口，默认模型 `qwen3.6-plus`，描述会进入检索文本和 `images_json`。
- Hybrid RAG：用 Chroma 做向量检索，用 BM25 做关键词检索，再通过 RRF、DeepSeek rerank 和邻近证据扩展生成上下文。
- 查询增强：根据问题类型、实体、章节意图、图表公式编号调整检索查询和向量/BM25 权重。
- 可信引用：引用来源由系统根据 chunk metadata 生成，页面展示片段、页码、章节和检索细节。
- DeepSeek 生成：DeepSeek 负责问答生成、文献卡片生成和中文翻译；Embedding 单独配置，默认使用智谱 `embedding-3`。
- 文献卡片库：支持生成、保存、搜索、筛选、编辑、批量删除、创建和重命名卡片库。
- 反馈闭环：回答下方可提交反馈；负面反馈会自动写入 Bad Case，并在管理员页查看。
- Docker 部署：内置 `Dockerfile`、`docker-compose.yml` 和 Streamlit 配置。

## 项目结构

```text
papermate/
  app.py                         # Streamlit 入口和页面交互
  config.py                      # 环境变量与路径配置
  requirements.txt
  .env.example
  DEPLOYMENT.md
  Dockerfile
  docker-compose.yml
  .streamlit/config.toml
  scripts/
    init_db.py                   # 初始化 SQLite
    cleanup_runtime_data.py      # 清理未被 SQLite 引用的运行时文件
    test_deepseek.py
    test_mineru.py
    test_mineru_visual_normalization.py
    test_query_processor.py
    test_rrf.py
    test_context_builder.py
    test_hybrid_retriever.py
  src/
    auth_service.py              # 本地用户注册/登录
    db.py                        # SQLite 表结构和迁移
    pdf_parser.py                # MinerU / PyMuPDF 解析入口
    mineru_client.py             # MinerU API、结果下载和视觉元素归一化
    markdown_translator.py       # Markdown 中文翻译
    bilingual_aligner.py         # 双语阅读对齐
    chunker.py                   # 论文分块
    embedding_client.py          # 智谱 / OpenAI-compatible embedding
    vector_store.py              # Chroma 持久化向量库
    rag_pipeline.py              # Hybrid RAG 问答链路
    card_pipeline.py             # 文献卡片生成
    literature_card_service.py   # 文献卡片库持久化
    feedback_service.py          # 反馈和 Bad Case
    errors.py
    logger.py
    retrieval/
      bm25_store.py
      context_builder.py
      evidence_expander.py
      hybrid_retriever.py
      query_planner.py
      query_processor.py
      reranker.py
      rrf.py
      tokenizer.py
      vector_retriever.py
```

运行后常见数据目录：

```text
data/
  papermate.db                   # SQLite 数据库
  uploads/                       # 上传的 PDF
  markdown/<paper_id>/           # MinerU 输出，默认路径
  chroma_db/                     # Chroma 向量库
  bm25/                          # BM25 pickle 和 payload JSON
logs/
  app.log
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

启动后打开 Streamlit 提示的本地地址。首次使用需要在页面上注册一个本地账号，然后进入“论文工作台”上传 PDF。

## 必填配置

复制 `.env.example` 为 `.env` 后，至少填写下面几项：

```env
MINERU_API_TOKEN=your_mineru_api_token

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro

EMBEDDING_PROVIDER=zhipu
EMBEDDING_API_KEY=your_zhipu_api_key
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSIONS=2048
```

注意：

- 页面登录使用本地注册账号，不是 `.env` 里的 `PAPERMATE_APP_PASSWORD`。
- `PAPERMATE_APP_PASSWORD` 当前主要作为“反馈记录”管理员密码的备用值；也可以单独设置 `PAPERMATE_ADMIN_PASSWORD`。
- DeepSeek 不参与 Embedding。不要把 DeepSeek 配成 Embedding 服务。
- 切换 Embedding provider、模型或维度后，需要重新构建论文索引。
- 语义分块会在解析阶段调用 Embedding；如果 Embedding API Key 或网络不可用，解析任务会失败。
- 图片解析需要用户在页面中显式选择“添加图片并重新解析”，这会在分块阶段调用 Qwen VLM；请在 `.env` 填写 `DASHSCOPE_API_KEY`，也可以用 `VLM_API_KEY` 覆盖。

## 主要配置

应用与路径：

```env
PAPERMATE_APP_NAME=PaperMate
APP_ENV=local
PAPERMATE_LOG_LEVEL=INFO
PAPERMATE_APP_PASSWORD=
PAPERMATE_HOST_PORT=8501

DATA_DIR=data
UPLOAD_DIR=data/uploads
MARKDOWN_DIR=data/markdown
CHROMA_DIR=data/chroma_db
BM25_DIR=data/bm25
LOG_DIR=logs
DB_PATH=data/papermate.db
```

PDF 解析：

```env
PDF_PARSE_PROVIDER=mineru
MINERU_API_TOKEN=your_mineru_api_token
MINERU_BASE_URL=https://mineru.net
MINERU_MODEL_VERSION=vlm
MINERU_IS_OCR=true
MINERU_ENABLE_FORMULA=true
MINERU_ENABLE_TABLE=true
MINERU_LANGUAGE=en
MINERU_POLL_INTERVAL=3
MINERU_POLL_TIMEOUT=600
```

MinerU 输出目录默认使用 `MARKDOWN_DIR`，也可以显式设置：

```env
MINERU_OUTPUT_DIR=data/markdown
```

临时切换到本地 PyMuPDF：

```env
PDF_PARSE_PROVIDER=pymupdf
```

中文 Markdown 翻译：

```env
TRANSLATION_ENABLED=true
TRANSLATION_PROVIDER=deepseek
TRANSLATION_MODEL=deepseek-chat
TRANSLATION_CHUNK_SIZE=3500
TRANSLATION_TIMEOUT=60
```

Hybrid RAG：

```env
RAG_CHUNK_STRATEGY=semantic_multimodal
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=100
TABLE_LARGE_ROW_CHUNK_SIZE=20
TABLE_WIDE_COLUMN_GROUP_SIZE=9
DASHSCOPE_API_KEY=your_bailian_dashscope_api_key
VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VLM_MODEL=qwen3.6-plus
VLM_TIMEOUT=90
VLM_TEMPERATURE=0.1
VLM_MAX_TOKENS=512
VECTOR_TOP_K=40
BM25_TOP_K=40
FINAL_TOP_K=8
RRF_K=60
CONTEXT_MAX_CHARS=9000
CONTEXT_EXPAND_WINDOW=1
RERANK_ENABLED=true
RERANK_TOP_K=30
RERANK_BATCH_SIZE=8
```

## 使用流程

1. 注册或登录本地用户。
2. 首次登录会自动创建默认团队和默认项目；owner/admin 可在“团队管理”添加成员和创建项目。
3. 进入“论文工作台”，上传一篇 PDF；系统只保存到论文库，不会自动创建解析或索引任务。
4. 在另一个终端运行 `python scripts/worker.py`，worker 会并发维护解析、索引和其他任务队列。
5. 进入“论文库”，按项目、上传人、解析状态、索引状态和关键词筛选论文，选择当前要先解析或构建索引的论文。
6. 打开已解析论文后，可下载原文 Markdown，或创建中文 Markdown 翻译任务；如需图片参与检索，在工作台选择“添加图片并重新解析”。
7. 解析完成后点击“构建论文索引”，系统会入队构建 Chroma 向量索引和 BM25 关键词索引。
8. 在 Ask PaperMate 区域提问，回答会展示可信引用、原文片段和 Hybrid 检索细节。
9. 在回答下方提交反馈；负面反馈自动进入 Bad Case。
10. 创建文献卡片任务，worker 完成后会保存到所选卡片库；“文献卡片库”按团队范围展示卡片。
11. 在“反馈记录”页输入管理员密码，查看当前团队的反馈总览、Bad Case 和原始记录。

后台 worker，默认启动 parse / index / other 三条 lane，其中索引 lane 会跳过仍在解析中的论文，优先处理下一篇已经解析完成的论文：

```bash
python scripts/worker.py
```

只运行某一类任务 lane：

```bash
python scripts/worker.py --types parse
python scripts/worker.py --types index
python scripts/worker.py --types translate,card,eval
```

使用旧的单队列串行模式：

```bash
python scripts/worker.py --serial
```

只处理一个任务后退出，便于本地调试：

```bash
python scripts/worker.py --once
```

## RAG 链路

```text
PDF
  -> MinerU / PyMuPDF 解析
  -> Markdown + pages + images
  -> semantic multimodal chunking
  -> SQLite papers/chunks
  -> Chroma vector index + BM25 keyword index
  -> Query Processor 分类和扩展问题
  -> Vector Search + BM25 Search
  -> RRF 融合排序
  -> DeepSeek rerank（失败回退本地 rerank）
  -> 邻近 chunk 扩展
  -> Context Builder 生成片段和 citations
  -> DeepSeek 基于参考片段回答
  -> 页面展示答案、可信引用、检索细节和反馈入口
```

如果 Chroma 构建成功但 BM25 失败，仍可使用语义检索；如果 BM25 成功但 Chroma 失败，仍可使用关键词检索。两者都失败时需要检查 API Key、网络和本地目录权限。

## 验证脚本

基础语法检查：

```bash
python -m compileall app.py config.py src scripts
```

本地逻辑测试：

```bash
python scripts/test_query_processor.py
python scripts/test_query_planner.py
python scripts/test_rrf.py
python scripts/test_context_builder.py
python scripts/test_reranker.py
python scripts/test_bm25_store.py
python scripts/test_evidence_expander.py
python scripts/test_hybrid_retriever.py
python scripts/test_mineru_visual_normalization.py
```

检索评估：

```bash
python scripts/eval_retrieval.py data/retrieval_seed.jsonl --disable-llm-rerank
```

需要外部服务的连通性测试：

```bash
python scripts/test_deepseek.py
python scripts/test_mineru.py path/to/paper.pdf
```

## 清理运行时文件

先 dry-run 查看候选项：

```bash
python scripts/cleanup_runtime_data.py
```

确认后删除未被 SQLite 引用的上传文件、Markdown/MinerU 输出和 BM25 文件：

```bash
python scripts/cleanup_runtime_data.py --apply
```

也清理 Python 缓存：

```bash
python scripts/cleanup_runtime_data.py --include-cache --apply
```

脚本不会按单篇论文清理 Chroma 文件；如需彻底重建向量库，请手动备份后重建 `data/chroma_db`。

## Docker 部署

```bash
cp .env.example .env
# 编辑 .env，填写 MINERU_API_TOKEN、DEEPSEEK_API_KEY、EMBEDDING_API_KEY、DASHSCOPE_API_KEY 等配置
docker compose up -d --build
```

默认映射端口：

```text
http://localhost:8501
```

可通过 `.env` 修改宿主机端口：

```env
PAPERMATE_HOST_PORT=8501
```

Docker Compose 会持久化：

- `./data:/app/data`
- `./logs:/app/logs`

云服务器部署细节见 `DEPLOYMENT.md`。

## 日志与排查

应用日志写入：

```text
logs/app.log
```

常见问题：

- MinerU 报错：检查 `MINERU_API_TOKEN`、网络和 `MINERU_POLL_TIMEOUT`。
- DeepSeek 生成失败：检查 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`、账户额度和网络。
- Embedding 失败：检查 `EMBEDDING_API_KEY`、`EMBEDDING_PROVIDER`、模型和维度。
- 问答提示无索引：先在论文工作台点击“构建论文索引”。
- 反馈记录页打不开：配置 `PAPERMATE_ADMIN_PASSWORD` 或 `PAPERMATE_APP_PASSWORD`。
- 中文译文不可用：确认 `TRANSLATION_ENABLED=true`，并检查 DeepSeek 翻译模型配置。
