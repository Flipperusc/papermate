# PaperMate 项目说明

这个文件用于让 Codex 在 `G:\Flipper\xiangmu\papermate` 下开启新对话时自动获得项目上下文。修改项目时优先阅读本文件，再按需查看 `README.md`、`app.py` 和 `src/` 内的服务模块。

## 1. 项目整体架构

PaperMate 是一个本地运行的论文阅读与 RAG 助手，核心形态是 Streamlit 单体 Web 应用 + SQLite 本地数据库 + 后台 worker 任务队列 + Chroma/BM25 检索索引。应用面向论文 PDF 管理、解析、阅读、问答、翻译、文献卡片和反馈闭环。

主要技术栈：

- UI：`Streamlit`，主入口在 `app.py`。
- 配置：`config.py` 读取 `.env`，集中管理路径、模型、检索、解析和翻译参数。
- 数据库：`SQLite`，数据库文件默认是 `data/papermate.db`，表结构和迁移在 `src/db.py`。
- PDF 解析：默认使用 MinerU API；可通过 `PDF_PARSE_PROVIDER=pymupdf` 切换到本地 PyMuPDF 文本抽取。
- 向量检索：`Chroma` 持久化在 `data/chroma_db`。
- 关键词检索：BM25 索引持久化在 `data/bm25`。
- LLM：DeepSeek 负责问答生成、文献卡片生成、翻译和可选 rerank。
- Embedding：默认智谱 `embedding-3`，通过 OpenAI-compatible 接口封装在 `src/embedding_client.py`。
- VLM：默认 Qwen/OpenAI-compatible 接口，用于图片、表格和公式等视觉元素描述。
- 后台任务：`scripts/worker.py` 从 SQLite `jobs` 表领取 parse/index/translate/card/eval 任务。

主要目录和文件：

```text
papermate/
  app.py                         Streamlit 页面、交互、上传、阅读器、问答和管理入口
  config.py                      环境变量、路径、模型、检索和解析配置
  requirements.txt               Python 依赖
  Dockerfile                     容器镜像
  docker-compose.yml             本地/服务器 Compose 启动
  DEPLOYMENT.md                  部署说明
  .streamlit/config.toml         Streamlit 配置
  scripts/
    init_db.py                   初始化 SQLite
    worker.py                    后台任务 worker
    cleanup_runtime_data.py      清理未被数据库引用的运行文件
    test_*.py                    各子系统验证脚本
  src/
    auth_service.py              本地账号注册、登录、密码哈希
    team_service.py              团队、项目、角色和权限
    paper_service.py             论文库查询、状态更新、删除和文件清理
    db.py                        SQLite schema、迁移、papers/chunks/qa/feedback/jobs 持久化
    pdf_parser.py                MinerU/PyMuPDF 解析入口和元素归一化
    mineru_client.py             MinerU API、结果下载、Markdown/content_list/images 处理
    chunker.py                   语义、多模态、表格感知分块
    vector_store.py              Chroma 向量索引写入和查询
    retrieval/                   BM25、query planner、RRF、rerank、证据扩展和 context builder
    rag_pipeline.py              检索、构造 prompt、LLM 回答、引用和 QA 日志
    markdown_translator.py       Markdown 中文翻译
    bilingual_aligner.py         原文/译文对照阅读对齐
    card_pipeline.py             文献卡片生成 prompt 和上下文选择
    literature_card_service.py   卡片库和卡片 CRUD
    feedback_service.py          反馈和 Bad Case 归档
    job_service.py               jobs 表入队、领取、完成、失败、取消、重试
    llm_client.py                DeepSeek/OpenAI-compatible LLM 调用
    vlm_client.py                Qwen/OpenAI-compatible VLM 调用
```

运行数据默认写入：

```text
data/
  papermate.db                   SQLite 数据库
  uploads/                       上传的 PDF
  markdown/<paper_id>/           MinerU 输出 Markdown、content_list、图片资源
  chroma_db/                     Chroma 向量库
  bm25/                          BM25 索引
logs/
  app.log                        应用日志
```

## 2. 项目运行逻辑

### 应用启动

1. 用户运行 `streamlit run app.py`。
2. `app.py` 调用 `config.py` 中的 `settings` 读取 `.env`。
3. 数据库相关代码通过 `src/db.py` 的 `init_db()` 确保表结构、索引和迁移存在。
4. 未登录用户进入登录/注册页面；账号保存在 SQLite `users` 表，密码使用 PBKDF2-HMAC-SHA256 哈希。
5. 登录后，`team_service.ensure_user_workspace()` 确保用户至少拥有一个团队和一个默认项目。
6. 左侧栏根据当前团队、项目和角色展示页面：论文工作台、论文库、文献卡片库、团队管理、反馈记录等。

### 论文上传与任务流

1. 用户在论文工作台上传 PDF。
2. 上传文件保存到 `data/uploads`，`paper_service.file_sha256()` 计算哈希，论文元数据写入 SQLite `papers` 表。
3. 上传本身只保存文件和记录；解析、索引、翻译、卡片生成等耗时动作通过 `job_service.enqueue_job()` 写入 `jobs` 表。
4. 另一个进程运行 `python scripts/worker.py`，默认启动三条 lane：
   - `parse`
   - `index`
   - `translate,card,eval`
5. worker 使用 `claim_next_job()` 领取队列任务，并在任务完成或失败时更新 `jobs` 和 `papers` 状态。

### 解析与分块链路

1. parse 任务调用 `src/pdf_parser.py::parse_pdf()`。
2. 默认走 MinerU：`MinerUClient.pdf_to_markdown()` 把 PDF 转成 Markdown、`content_list.json` 和可选图片资源。
3. 如果使用 PyMuPDF，只抽取本地文本，不提供 MinerU 的 Markdown、图片和复杂版面能力。
4. `pdf_parser.py` 将 MinerU `content_list` 归一化为 text/table/image/equation 等有序元素。
5. `chunker.chunk_pages()` 对文本按章节、段落、句子和 embedding 相似度做语义分块；对表格做行/列分块；对图片和公式生成可检索描述并绑定到相邻 chunk。
6. 解析结果通过 `save_paper_and_chunks()` 写入 `papers` 和 `chunks` 表。

### 索引与问答链路

1. index 任务从 SQLite 读取 `chunks`。
2. `VectorStore.add_chunks()` 写入 Chroma 向量索引。
3. `BM25Store.build_index()` 写入 BM25 关键词索引。
4. 用户在 Ask PaperMate 提问时，`rag_pipeline.RAGPipeline.answer_question()` 进入问答链路。
5. `HybridRetriever.retrieve()` 先用 `query_planner.plan_query()` 判断问题类型、扩展关键词、识别章节意图和实体。
6. 系统并行使用向量检索和 BM25 检索，再用 RRF 融合排序。
7. 可选使用 DeepSeek rerank；随后 `EvidenceExpander` 扩展相邻 chunk，补齐上下文。
8. `context_builder.build_context()` 生成 LLM prompt context 和系统可信引用列表。
9. `LLMClient.generate()` 调用 DeepSeek 生成回答；回答、引用、检索细节和 QA 日志返回 UI。
10. 用户可对回答提交反馈，负面反馈会进入 Bad Case。

### 翻译、阅读和卡片链路

1. 翻译任务调用 `markdown_translator.translate_markdown_to_chinese()`，输出非破坏性的 `.zh.md` 文件，不覆盖原 Markdown。
2. 阅读器支持原文、中文译文、双语对照；双语对照通过 `bilingual_aligner.py` 将 Markdown 章节和块按顺序对齐。
3. 文献卡片任务调用 `card_pipeline.generate_literature_card()`，优先选择摘要、引言、方法、实验、结果、结论等高信号 chunk 作为上下文。
4. 生成的 Markdown 卡片由 `literature_card_service.py` 提取结构字段并保存到 SQLite `literature_cards` 与 `card_libraries` 相关数据中。

## 3. 已实现的具体功能

- 本地账号注册、登录、会话保持和退出。
- 团队、项目、成员、角色权限管理。
- 论文 PDF 上传、重复文件识别、论文库筛选、状态展示和删除。
- 后台任务队列：解析、索引、翻译、文献卡片、评估占位任务。
- MinerU PDF 转 Markdown、content_list 读取、图片/表格/公式资源归一化。
- PyMuPDF 本地文本解析兜底。
- 语义分块、多模态分块、表格分块和 chunk metadata 持久化。
- Chroma 向量索引构建和查询。
- BM25 关键词索引构建和查询。
- Hybrid Retrieval：query planning、向量检索、BM25 检索、RRF 融合、rerank、邻近证据扩展。
- 基于论文原文片段的 RAG 问答。
- 可信引用展示：页码、章节、chunk、检索来源、分数和原文片段。
- Markdown 中文翻译和中文 Markdown 下载。
- 原文、译文、双语对照阅读器。
- 文献卡片生成、保存、搜索/筛选、编辑、删除、批量删除、卡片库创建和重命名。
- 用户反馈记录、Bad Case 自动归档、管理员反馈记录查看。
- 队列进度 UI、任务取消、重试、清空队列和中断任务恢复。
- Docker/Compose 部署支持。

## 4. 具体功能的实现逻辑

### 本地账号与权限

- `auth_service.py` 负责用户名规范化、注册校验、密码哈希、登录验证和最后登录时间更新。
- `team_service.py` 定义 `viewer/editor/admin/owner` 四级角色，并用 `require_team_role()` 作为服务层权限门禁。
- 首次登录时自动创建个人团队和默认项目；团队成员和项目管理操作要求 admin 或 owner。

### 论文库

- 论文上传后先保存 PDF 文件，再把 `paper_id`、文件名、大小、路径、哈希、owner、team、project 和状态写入 `papers`。
- `paper_service.list_accessible_papers()` 按团队、项目、上传人、解析状态、索引状态和关键词筛选。
- 删除论文时会先校验 editor 权限，再删除数据库记录、关联任务，并清理 `data/` 下安全范围内的文件。

### 后台任务队列

- UI 不直接执行耗时任务，而是调用 `enqueue_job()` 写入 `jobs`。
- worker 通过 `claim_next_job()` 领取队列任务，状态从 `queued` 变为 `running`。
- 成功时 `complete_job()` 写入结果 JSON；失败时 `fail_job()` 写入错误并按任务类型回写论文状态。
- parse 失败会同步让等待中的 index 任务失败，避免索引未完成解析的论文。
- 默认 worker lane 将 parse、index、translate/card/eval 分开，减少长任务互相阻塞。

### PDF 解析

- `parse_pdf()` 根据 `settings.pdf_parse_provider` 分派到 MinerU 或 PyMuPDF。
- MinerU 路径会上传 PDF、轮询任务、下载压缩结果，并保存 Markdown、content_list 和图片。
- `elements_from_content_list()` 把 MinerU 内容转成统一元素，保留页码、bbox、caption、table_body、path、visual_id 等信息。
- 如果 MinerU content_list 缺失，会从 Markdown 兜底生成页面文本；如果没有任何文本会抛出 PDF_NO_TEXT。

### 分块

- `chunker.py` 先识别论文常见章节，例如 Abstract、Introduction、Method、Experiments、Results、Conclusion。
- 文本分块以段落和句子为基本单位，长文本会按大小拆分，chunk 保留 `page_num`、`section_title`、`chunk_type`。
- 语义分块使用 embedding 相似度辅助合并或切分句子，减少跨主题 chunk。
- 表格分块会解析 HTML/Markdown 表格，针对大表按行拆分、宽表按列组拆分，并生成表格摘要和 `tables_json`。
- 图片、公式和表格图片在 `include_images=True` 时会调用 Qwen VLM 生成描述，描述写入可检索文本和 `images_json`。

### 索引

- 向量索引使用 `vector_store.py`，将 enriched chunk text embedding 后写入 Chroma collection。
- BM25 索引使用 `retrieval/bm25_store.py`，将 chunk 搜索文本 tokenized 后持久化到 `data/bm25`。
- index 任务允许部分成功：Chroma 或 BM25 其中一个失败时，论文索引状态为 `partial`，问答时仍可走另一条检索通道。

### RAG 问答

- `query_planner.py` 用规则判断 exact/semantic/default 问题类型。
- exact 类问题提高 BM25 权重，semantic 类问题提高向量权重，default 两者均衡。
- `HybridRetriever` 分别调用 `VectorRetriever` 和 `BM25Store.search()`，单边失败时进入 vector 或 BM25 fallback。
- RRF 将不同检索列表按 rank 融合，保留来源、rank、BM25 分数和向量距离。
- `LLMReranker` 可用 DeepSeek 对候选 chunk 重排；不可用时有本地规则分数兜底。
- `EvidenceExpander` 取核心 chunk 的前后邻居，补充上下文连续性。
- `context_builder.py` 去重、控制字符预算，并生成系统持有的 citation 数据；UI 信任这份 citation，而不是让模型自由编造来源。
- `rag_pipeline.py` 的 prompt 明确要求只基于参考片段回答，证据不足时返回拒答文案。

### Markdown 翻译与双语阅读

- `markdown_translator.py` 会保护代码块、表格、公式等 Markdown 结构，再按块调用 DeepSeek 翻译。
- 翻译结果输出为源文件旁边的 `.zh.md`，不会覆盖原始 Markdown。
- `bilingual_aligner.py` 把原文和译文拆成章节与块，再按顺序对齐，供双语阅读器交错展示。

### 文献卡片

- `card_pipeline.py` 从持久化 chunk 中挑选前若干 chunk 和关键章节 chunk，组成最多约 18000 字符的上下文。
- LLM prompt 要求只使用原文证据，缺失字段写“原文未明确说明”。
- `literature_card_service.py` 将 Markdown 卡片解析成 title、authors、year、research_field、research_question、method_summary、datasets 等字段，便于列表、筛选和编辑。
- 卡片库按用户和团队维护，默认会为当前用户创建默认卡片库。

### 反馈与 Bad Case

- `feedback_service.py` 保存用户对回答的反馈，包含问题、回答、引用、检索细节和备注。
- 负面反馈会自动写入 `bad_cases`，用于后续分析检索或生成失败案例。
- 反馈记录页需要管理员密码，密码来源优先是 `PAPERMATE_ADMIN_PASSWORD`，再回退到 `PAPERMATE_APP_PASSWORD`。

## 常用命令

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\init_db.py
streamlit run app.py
```

后台 worker：

```bash
python scripts\worker.py
python scripts\worker.py --once
python scripts\worker.py --types parse
python scripts\worker.py --types index
python scripts\worker.py --types translate,card,eval
```

基础检查：

```bash
python -m compileall app.py config.py src scripts
```

常用验证脚本：

```bash
python scripts/test_query_processor.py
python scripts/test_query_planner.py
python scripts/test_rrf.py
python scripts/test_context_builder.py
python scripts/test_reranker.py
python scripts/test_bm25_store.py
python scripts/test_hybrid_retriever.py
python scripts/test_mineru_visual_normalization.py
```

## 关键配置

`.env` 至少通常需要：

```env
MINERU_API_TOKEN=...
DEEPSEEK_API_KEY=...
EMBEDDING_API_KEY=...
```

常见模型和检索配置：

```env
PDF_PARSE_PROVIDER=mineru
EMBEDDING_PROVIDER=zhipu
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSIONS=2048
DEEPSEEK_MODEL=deepseek-v4-pro
TRANSLATION_MODEL=deepseek-chat
RAG_CHUNK_STRATEGY=semantic_multimodal
VECTOR_TOP_K=40
BM25_TOP_K=40
FINAL_TOP_K=8
RRF_K=60
CONTEXT_MAX_CHARS=9000
RERANK_ENABLED=true
```

图片理解相关：

```env
DASHSCOPE_API_KEY=...
VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VLM_MODEL=qwen3.6-plus
VLM_ENABLED=true
```

## 给 Codex 的工作约定

- 当前仓库可能有用户未提交修改；不要回滚或覆盖无关文件。
- 修改功能前先确认入口在 `app.py`，业务逻辑通常在 `src/*_service.py`、`src/*_pipeline.py` 或 `src/retrieval/`。
- 不要把 API key、token 或 `.env` 内容写入文档、日志或测试输出。
- 涉及数据库字段时同步检查 `src/db.py` 的 schema、迁移和相关服务查询。
- 涉及异步任务时同时检查 UI 入队逻辑、`job_service.py`、`scripts/worker.py` 和论文状态字段。
- 涉及 RAG 质量时优先检查 query planner、BM25/Chroma 索引文本、RRF、rerank、evidence expansion 和 context builder，而不是只改 prompt。
- 涉及图片、表格或公式时检查 MinerU content_list 归一化、`chunker.py` 的 metadata 和 VLM 配置。
- 新增验证优先放到 `scripts/test_*.py`，保持可用的单文件脚本风格。
