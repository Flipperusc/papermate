# PaperMate 云服务器部署

推荐使用 Docker Compose 部署。下面以 Ubuntu 22.04/24.04 云服务器为例。

## 1. 准备服务器

在云厂商控制台放行 TCP 端口：

- 直接访问：放行 `8501`
- 使用 Nginx 反向代理：放行 `80` 和 `443`

安装 Docker：

```bash
sudo apt update
sudo apt install -y ca-certificates curl git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

重新登录服务器后确认：

```bash
docker --version
docker compose version
```

## 2. 上传项目

可以用 Git，也可以直接用 `scp`/SFTP 上传整个 `papermate` 目录。

```bash
git clone <your-repo-url> papermate
cd papermate
```

如果不是 Git 项目，把本地目录上传到服务器后进入目录即可。

## 3. 配置环境变量

```bash
cp .env.example .env
nano .env
```

至少填写这些值：

```env
PAPERMATE_APP_PASSWORD=your_strong_login_password

MINERU_API_TOKEN=your_mineru_api_token

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro

EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://api.openai.com/v1
```

`PAPERMATE_APP_PASSWORD` 是云端访问密码。公开部署时不要留空。

## 4. 启动服务

```bash
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f papermate
```

访问：

```text
http://服务器公网IP:8501
```

## 5. 数据持久化和备份

Docker Compose 已把以下目录挂载到宿主机：

- `./data:/app/data`
- `./logs:/app/logs`

这些内容会保留在服务器项目目录中：

- `data/papermate.db`：SQLite 数据库
- `data/uploads/`：上传的 PDF
- `data/chroma_db/`：Chroma 向量库
- `data/mineru_outputs/`：MinerU Markdown 和图片
- `logs/app.log`：应用日志

备份示例：

```bash
tar -czf papermate-backup-$(date +%F).tar.gz data logs .env
```

## 6. 更新版本

```bash
git pull
docker compose up -d --build
```

如果不是 Git 项目，重新上传文件后执行：

```bash
docker compose up -d --build
```

## 7. Nginx 反向代理可选

如果你有域名，例如 `papermate.example.com`，可以用 Nginx 转发到本机 `8501`。

```nginx
server {
    listen 80;
    server_name papermate.example.com;

    client_max_body_size 200m;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

生产环境建议再用 Certbot 配 HTTPS。

## 8. 查看用户反馈

推荐方式：打开 PaperMate 页面，在侧边栏进入“反馈记录”页面。

该页面包含：

- 用户反馈列表
- 负面反馈数量
- Bad Case 列表
- 对应问题、回答、反馈类型、补充说明和关联论文

也可以直接在服务器上用 Python 查询 SQLite：

```bash
docker compose exec papermate python -c "from src.feedback_service import list_feedback_records, list_bad_cases; print(list_feedback_records(20)); print(list_bad_cases(20))"
```

## 9. 常用排查

容器状态：

```bash
docker compose ps
```

应用日志：

```bash
docker compose logs -f papermate
tail -f logs/app.log
```

重新启动：

```bash
docker compose restart papermate
```

完全重建：

```bash
docker compose down
docker compose up -d --build
```
