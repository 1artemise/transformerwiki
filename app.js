const contentRoot = "./content/transformer/";
const groups = window.TRANSFORMER_DOCS;
const allDocs = groups.flatMap(group => group.items);
const byPath = new Map(allDocs.filter(doc => doc.path).map(doc => [normalizePath(doc.path), doc]));
const byTitle = new Map(allDocs.map(doc => [doc.title.toLowerCase(), doc]));
const article = document.querySelector("#article");
const docNav = document.querySelector("#docNav");
const toc = document.querySelector("#toc");
const search = document.querySelector("#search");

renderNav();
window.addEventListener("hashchange", loadCurrentRoute);
loadCurrentRoute();

function renderNav() {
  docNav.innerHTML = groups.map(group => `
    <section class="nav-section">
      <p class="nav-section-title">${escapeHtml(group.title)}</p>
      ${group.items.map(doc => `<a href="#/${doc.id}" data-doc-id="${doc.id}">${escapeHtml(doc.title)}</a>`).join("")}
    </section>
  `).join("");

  search.addEventListener("input", () => {
    const keyword = search.value.trim().toLowerCase();
    docNav.querySelectorAll("a").forEach(link => {
      link.classList.toggle("hidden", keyword && !link.textContent.toLowerCase().includes(keyword));
    });
  });
}

async function loadCurrentRoute() {
  const id = decodeURIComponent(location.hash.replace(/^#\//, "")) || "home";
  const doc = allDocs.find(item => item.id === id) || allDocs[0];
  setActiveNav(doc.id);

  if (doc.home) {
    article.innerHTML = renderHome();
    renderToc();
    return;
  }

  try {
    const response = await fetch(contentRoot + encodeURI(doc.path));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const markdown = await response.text();
    article.innerHTML = renderMarkdown(markdown, doc);
    renderToc();
  } catch (error) {
    article.innerHTML = `<h1>加载失败</h1><p>无法加载 ${escapeHtml(doc.path)}。错误：${escapeHtml(error.message)}</p>`;
    toc.innerHTML = "";
  }
}

function setActiveNav(id) {
  docNav.querySelectorAll("a").forEach(link => {
    link.classList.toggle("active", link.dataset.docId === id);
  });
}

function renderHome() {
  return `
    <p class="home-kicker">Transformer learning wiki</p>
    <h1>先建立直觉，再补前置，最后读架构和论文</h1>
    <p>大多数 Transformer 教程你是看不懂的，因为它默认你已经知道很多前置知识。softmax、词嵌入、LayerNorm、残差连接、前馈网络、mask、QKV，这些东西如果单独都不熟，再被一张架构图串起来，人很容易被绕晕。作者艰难曲折学习后得出了这个学习经验（一家之言）</p>
    <p>这份站点是作者自己的学习笔记整理，推荐学习路线：先用b站上的 “<a href="https://www.bilibili.com/video/BV13z421U7cs?spm_id_from=333.788.recommend_more_video.2&trackid=web_related_0.router-related-2479604-s7xkd.1778464404440.361&vd_source=6f3b9bab813c14ca00fbb56363a43d7e" target="_blank" rel="noopener noreferrer">3Blue1Brown</a>” 和 “<a href="https://www.bilibili.com/video/BV1NCgVzoEG9/?spm_id_from=333.337.search-card.all.click&vd_source=6f3b9bab813c14ca00fbb56363a43d7e" target="_blank" rel="noopener noreferrer">飞天闪客</a>” 的视频建立直觉，不追求一步到位推完公式；然后补 softmax 的实现、LayerNorm 的思想、神经网络和前馈网络这些地基；最后再回到 <em>Attention Is All You Need</em>，把编码器、解码器、注意力和训练目标连起来。</p>
    <blockquote><strong>共同维护声明：</strong>这个 wiki 想建成持续迭代的共建知识库。欢迎通过 GitHub 的 Issue / Pull Request 提交勘误和补充；作者后续也会更新</blockquote>
    <p>协作入口：<a href="https://github.com/1artemise/transformerwiki" target="_blank" rel="noopener noreferrer">github.com/1artemise/transformerwiki</a></p>
    <div class="home-links">
      <a class="home-card" href="#/neural-network"><strong>1. 前置知识</strong><span>神经网络、FNN、损失函数、梯度问题。</span></a>
      <a class="home-card" href="#/softmax"><strong>2. Transformer 小积木</strong><span>softmax、词嵌入、LayerNorm、Add。</span></a>
      <a class="home-card" href="#/attention"><strong>3. 注意力机制</strong><span>QKV、缩放点积注意力、多头注意力、mask。</span></a>
      <a class="home-card" href="#/Transformer"><strong>4. 回到整体架构</strong><span>编码器、解码器、位置编码、最终输出。</span></a>
    </div>
  `;
}

function renderMarkdown(markdown, doc) {
  const lines = preprocessMarkdown(markdown, doc).split(/\r?\n/);
  let html = "";
  let inCode = false;
  let codeLang = "";
  let code = [];
  let inTable = false;
  let tableRows = [];
  let inList = false;
  let listType = "";

  const closeList = () => {
    if (!inList) return;
    html += `</${listType}>`;
    inList = false;
    listType = "";
  };

  const closeTable = () => {
    if (!inTable) return;
    html += renderTable(tableRows, doc);
    inTable = false;
    tableRows = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/, "");

    if (line.startsWith("```")) {
      if (!inCode) {
        closeList();
        closeTable();
        inCode = true;
        codeLang = line.slice(3).trim();
        code = [];
      } else {
        html += `<pre><code class="language-${escapeHtml(codeLang)}">${escapeHtml(code.join("\n"))}</code></pre>`;
        inCode = false;
      }
      continue;
    }

    if (inCode) {
      code.push(rawLine);
      continue;
    }

    if (isTableLine(line)) {
      closeList();
      inTable = true;
      tableRows.push(line);
      continue;
    }

    closeTable();

    if (!line.trim()) {
      closeList();
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      const text = stripInlineMarkdown(heading[2]);
      const id = slugify(text);
      html += `<h${level} id="${id}">${renderInline(heading[2], doc)}</h${level}>`;
      continue;
    }

    const unordered = line.match(/^\s*[-*]\s+(.+)$/);
    if (unordered) {
      if (!inList || listType !== "ul") {
        closeList();
        html += "<ul>";
        inList = true;
        listType = "ul";
      }
      html += `<li>${renderInline(unordered[1], doc)}</li>`;
      continue;
    }

    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (ordered) {
      if (!inList || listType !== "ol") {
        closeList();
        html += "<ol>";
        inList = true;
        listType = "ol";
      }
      html += `<li>${renderInline(ordered[1], doc)}</li>`;
      continue;
    }

    const quote = line.match(/^>\s?(.+)$/);
    if (quote) {
      closeList();
      html += `<blockquote>${renderInline(quote[1], doc)}</blockquote>`;
      continue;
    }

    closeList();
      html += `<p>${renderInline(line, doc)}</p>`;
  }

  closeList();
  closeTable();
  return html;
}

function preprocessMarkdown(markdown, doc) {
  let text = markdown.replace(/^\uFEFF/, "");

  text = replaceOutsideCode(text, chunk => {
    let output = chunk.replace(/!\[\[([^|\]]+)(?:\|[^\]]+)?\]\]/g, (_, rawPath) => {
      return `![${rawPath.trim()}](图/${rawPath.trim()})`;
    });

    output = output.replace(/!\[([^\]]*?)\|[^\]]+\]\(([^)]+)\)/g, "![$1]($2)");
    output = output.replace(/\[\[([^|\]#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]/g, (_, target, label) => {
      const docTarget = resolveWikiDoc(target.trim());
      const textLabel = label || target.trim().split("/").pop();
      return docTarget ? `[${textLabel}](${docTarget})` : textLabel;
    });

    return output;
  });

  return text;
}

function replaceOutsideCode(markdown, transform) {
  return markdown
    .split(/(```[\s\S]*?```)/g)
    .map(part => part.startsWith("```") ? part : transform(part))
    .join("");
}

function renderInline(value, doc = null) {
  let html = escapeHtml(value);

  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => {
    return `<img src="${escapeAttribute(resolveAssetPath(src, doc))}" alt="${escapeAttribute(alt)}">`;
  });

  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
    const resolved = resolveLink(href, doc);
    return `<a href="${escapeAttribute(resolved)}">${label}</a>`;
  });

  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

  return html;
}

function renderTable(rows, doc = null) {
  const filtered = rows.filter(row => !/^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(row));
  if (!filtered.length) return "";

  const cells = filtered.map(row => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim()));
  const [head, ...body] = cells;
  return `
    <table>
      <thead><tr>${head.map(cell => `<th>${renderInline(cell, doc)}</th>`).join("")}</tr></thead>
      <tbody>${body.map(row => `<tr>${row.map(cell => `<td>${renderInline(cell, doc)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  `;
}

function renderToc() {
  const headings = [...article.querySelectorAll("h2, h3, h4")];
  toc.innerHTML = headings.map(heading => {
    return `<a class="toc-${heading.tagName.toLowerCase()}" href="#${heading.id}">${heading.textContent}</a>`;
  }).join("");
}

function isTableLine(line) {
  return /^\s*\|.+\|\s*$/.test(line);
}

function resolveLink(href, doc = null) {
  if (/^https?:\/\//i.test(href)) return normalizeExternalUrl(href);
  const clean = decodeURIComponent(href).split("#")[0];
  const candidate = doc ? resolveRelativePath(doc.path, clean) : clean;
  const targetDoc = byPath.get(normalizePath(candidate)) || byPath.get(normalizePath(clean));
  return targetDoc ? `#/${targetDoc.id}` : resolveAssetPath(candidate);
}

function resolveWikiDoc(target) {
  if (target.startsWith("AI/transformer/")) {
    const path = target.replace(/^AI\/transformer\//, "") + ".md";
    const doc = byPath.get(normalizePath(path));
    return doc ? `#/${doc.id}` : "";
  }

  const title = target.split("/").pop().toLowerCase();
  const direct = byTitle.get(title);
  if (direct) return `#/${direct.id}`;

  const match = allDocs.find(doc => doc.path && doc.path.toLowerCase().endsWith(`${title}.md`));
  return match ? `#/${match.id}` : "";
}

function resolveAssetPath(src, doc = null) {
  const clean = decodeURIComponent(src).split("#")[0];
  if (/^https?:\/\//i.test(clean)) return normalizeExternalUrl(clean);
  if (clean.startsWith("data:")) return clean;

  if (normalizePath(clean).startsWith("图/")) {
    return contentRoot + encodeURI(normalizePath(clean));
  }

  const path = normalizePath(doc ? resolveRelativePath(doc.path, clean) : clean);
  if (path.startsWith("图/")) return contentRoot + encodeURI(path);
  if (path.includes("/图/")) return contentRoot + encodeURI(path.slice(path.indexOf("图/")));
  return contentRoot + encodeURI(path);
}

function normalizeExternalUrl(url) {
  return url.replace(/^http:\/\//i, "https://");
}

function resolveRelativePath(fromPath, relative) {
  const base = fromPath.split("/").slice(0, -1);
  for (const part of relative.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") base.pop();
    else base.push(part);
  }
  return base.join("/");
}

function normalizePath(path) {
  return decodeURIComponent(path).replace(/\\/g, "/").replace(/^\.\//, "");
}

function slugify(text) {
  return text.trim().toLowerCase().replace(/[^\p{Letter}\p{Number}]+/gu, "-").replace(/^-|-$/g, "") || "section";
}

function stripInlineMarkdown(text) {
  return text.replace(/!\[[^\]]*\]\([^)]+\)/g, "").replace(/\[([^\]]+)\]\([^)]+\)/g, "$1").replace(/\*\*|`/g, "");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/"/g, "&quot;");
}
