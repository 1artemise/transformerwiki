# Transformer 学习 Wiki

这是一个纯前端静态站点，可以直接部署到 GitHub Pages。

## 本地预览

这个站点通过 `fetch` 加载 Markdown，建议用本地服务预览。

如果想用本地服务：

```bash
python -m http.server 8000
```

然后访问：

```text
http://localhost:8000
```

## GitHub Pages 部署

1. 把仓库推到 GitHub。
2. 打开仓库 Settings。
3. 进入 Pages。
4. Source 选择 `Deploy from a branch`。
5. Branch 选择 `main`，目录选择 `/docs`。
6. 保存后等待 GitHub 生成访问地址。

有自己的域名时，在 Pages 里设置 Custom domain 即可。

如果之后要绑定域名，可以在 `docs/` 下新增 `CNAME` 文件，内容只写你的域名，例如：

```text
example.com
```

## 同步 Transformer 笔记

网站内容来自 `docs/content/transformer/`。

如果后续修改了 `AI/transformer/` 下的 Obsidian 笔记，需要把 Markdown 和图片同步到 `docs/content/transformer/`。当前目录结构是：

```text
docs/content/transformer/
├── Transformer.md
├── 代码实现.md
├── 前置知识/
├── 关键模块/
└── 图/
```

同步后如果新增了笔记，还需要在 `docs/manifest.js` 里加入导航项。
