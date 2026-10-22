# STAT6207 Applied Deep Learning — 个人学习库

> 面向零基础学生的课程笔记、练习题与会话记录。
> 导师角色与工作流定义在 [`skills/stat6207-tutor/SKILL.md`](skills/stat6207-tutor/SKILL.md)。
> 仓库根目录的 `AGENTS.md` 是会话入口，确保每次新对话先读取进度和规则。

---

## 目录结构

```
stat6207-learning/
├── README.md                     # 本文件
├── skills/
│   └── stat6207-tutor/
│       └── SKILL.md              # 导师角色 / 工作流 / 质量标准
├── lectures/
│   ├── L1/
│   │   ├── L1_notes.md           # 逐页中英对照 + 零基础讲解
│   │   ├── L1_outline.md         # 1–2 页要领提纲（复习用）
│   │   ├── L1_quiz.md            # 10 题 + 详细解析
│   │   ├── L1_qa.md              # 答疑笔记本（提问 + 完整解答）
│   │   ├── L1_extracted.txt      # PDF 原始文本（备查）
│   │   └── images/               # 提取的插图 + 整页渲染图
│   ├── L2/
│   │   ├── L2_notes.md           # 逐页中英对照 + 零基础讲解（85 页）
│   │   ├── L2_outline.md         # 1–2 页要领提纲
│   │   ├── L2_quiz.md            # 10 题 + 详细解析
│   │   ├── L2_qa.md              # 答疑笔记本
│   │   ├── L2_extracted.txt      # PDF 原始文本
│   │   └── images/               # 提取插图
│   ├── L3/ … L6/                 # 结构同上，待填充
├── sessions/
│   ├── 20261022_062233_session.md  # L1 生成
│   ├── 20261022_062936_session.md  # 保存进度 + 固化启动规则
│   ├── 20261022_064921_session.md  # 数学难度回调 + 要领提纲
│   ├── 20261022_191214_session.md  # 答疑 + 术语规范 + 排版重排
│   └── 20261022_202044_session.md  # L2 生成（笔记本 / 提纲 / 练习 / 答疑）
└── tools/
    ├── extract_pdf.py            # 提取全文 + 全部图片
    └── extract_figures.py        # 只提取有意义的插图
```

---

## 课程进度

| 课 | 主题 | 笔记 | 提纲 | 练习 | 答疑 | 状态 |
|---|---|---|---|---|---|---|
| L1 | ML Foundations：什么是 ML、监督学习、KNN | [L1_notes](lectures/L1/L1_notes.md) | [L1_outline](lectures/L1/L1_outline.md) | [L1_quiz](lectures/L1/L1_quiz.md) | [L1_qa](lectures/L1/L1_qa.md) | ✅ 完成 |
| L2 | Linear Classifier、Transfer Learning、Local LLM | [L2_notes](lectures/L2/L2_notes.md) | [L2_outline](lectures/L2/L2_outline.md) | [L2_quiz](lectures/L2/L2_quiz.md) | [L2_qa](lectures/L2/L2_qa.md) | ✅ 完成 |
| L3 | LLM, RAG, CoT, Agentic AI | — | — | — | — | ⬜ 待开始 |
| L4 | Transformers, Quantization | — | — | — | — | ⬜ 待开始 |
| L5 | CNN, Vision Transformer | — | — | — | — | ⬜ 待开始 |
| L6 | Multimodal AI, VLM, Face Recognition | — | — | — | — | ⬜ 待开始 |

> 实际章节以老师课件为准；上表为 P11 的课程大纲。

---

## 下次从这里开始（2026-10-22 20:20）

- **已完成**：L2 全套（笔记 85 页 / 提纲 / 10 题 / 答疑笔记本）。
- **你正在做**：读 `L2_outline.md` 建立框架 → 读 `L2_notes.md` → 做 `L2_quiz.md`。
- **下一节**：L3 — LLM, RAG, CoT, Agentic AI（`2026_6207_Lecture_L3_v1.pdf`）。
- **继续方式**：对导师说 **"继续 L3"**；想先讲 L2 错题，说 **"讲 L2 错题"**。

> 每次会话开始，导师会先读 `SKILL.md` → 本文件 → `sessions/` 最新记录 → 本课 `Lx_qa.md` 索引，
> 再确认进度，不会从头重讲。

---

## 怎么用这个库

### 学习一节课

1. 先读 `Lx_outline.md` 的要领提纲，建立框架（5 分钟）。
2. 再读 `Lx_notes.md`，遇到公式先看"符号/直觉"，再回看公式。
3. 读完后合上笔记，做 `Lx_quiz.md`。
4. 对照答案解析，把错题对应的小节重读一遍。
5. 最后用提纲做一次"合书回忆"，讲不出来的回 notes 查。

### 复习

- 看每节末尾的"公式总表""术语表""10 个最容易错的地方"。
- 用"自评表"定位薄弱点。
- 翻 `Lx_qa.md`：里面是**你自己问过的问题**，通常正好是薄弱点。

### 提问

直接问，不用挑"值得问的"。
答案会给成四块：**一句话答案 → 展开 → 例子/手算 → 易错点/考点**，
并追加到对应的 `Lx_qa.md`，下次复习直接翻问答索引。

### 继续下一节

对导师说"继续 L2"即可。每节课结束后会更新：
- `lectures/Lx/` 下的笔记与练习
- `sessions/` 下的会话记录
- 本文件的进度表

---

## 快速开始（环境）

```powershell
# 1. 进入项目
cd stat6207-learning

# 2. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 安装常用库
pip install numpy pandas matplotlib scikit-learn torch

# 4. 提取新课件（示例：L2）
$env:PYTHONIOENCODING='utf-8'
python tools/extract_figures.py ..\2026_6207_lecture_L2_final.pdf lectures\L2\images
python tools/extract_pdf.py ..\2026_6207_lecture_L2_final.pdf lectures\L2\images --text lectures\L2\L2_extracted.txt
```

> Windows PowerShell 若禁止运行脚本：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## L1 速览

**三句话**：

1. 机器学习 = 用数据调参数，而不是人写规则。
2. 训练 = 用损失函数衡量误差 + 用梯度下降调参数。
3. 评估 = 训练集学、验证集选、测试集只考一次。

**核心公式**：

$$J(\theta)=\frac1n\sum_{i=1}^n\big(y_i-\hat y_i\big)^2,\qquad
\theta_{t+1}=\theta_t-\eta\nabla_\theta J(\theta_t)$$

**核心算法 KNN**：找最近的 $K$ 个邻居，分类投票、回归平均。

---

## L2 速览

**三句话**：

1. 线性分类器 = 内积打分 $w^Tx+b$，边界是超平面。
2. 迁移学习 = 借源域知识：冻结 / 微调 / 加权样本。
3. 本地 LLM = 先算显存 $\frac{P\times b}{8}+2$，再选量化与工具。

**核心公式**：

$$\sigma(z)=\frac{1}{1+e^{-z}},\qquad \log\frac{p}{1-p}=w^Tx,\qquad \text{Memory}=\frac{P\times b}{8}+2$$

**核心选择**：数据 <1000 冻结；1000–50000 微调；>50000 从头训。

---

## 备注

- 所有数学公式使用 LaTeX（`$...$` / `$$...$$`），建议用支持数学渲染的 Markdown 阅读器（VS Code + Markdown Preview Enhanced、Typora、Obsidian）。
- 图片来自课件截图/提取，仅用于个人学习。
- 课件原文版权归授课教师所有。
