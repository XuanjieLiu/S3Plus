#!/usr/bin/env python3
import html
import json
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
EXPS_DIR = BASE_DIR / "exps"

REPORT_NAME = "compare_2026-05-17_opvq_four_exps"
TITLE = "opVQLearn 四实验对比分析"

EXPERIMENTS = [
    {
        "short": "symm005_sps_balance",
        "exp_name": "2026.5.16_opvq_2code_fromMulBasedSps_lr3e4_balance",
        "label": "balance + symm0.05 + SPS-in-symm",
        "color": "#0f766e",
    },
    {
        "short": "symm005_sps_no_balance",
        "exp_name": "2026.5.16_opvq_2code_fromMulBasedSps_lr3e4_balanceNo",
        "label": "no balance + symm0.05 + SPS-in-symm",
        "color": "#b45309",
    },
    {
        "short": "symm001_no_sps_balance",
        "exp_name": "2026.5.17_opvq_2code_fromMulBasedSps_lr3e4_balance_symm001_noSpsVqSymm",
        "label": "balance + symm0.01, no SPS-in-symm",
        "color": "#2563eb",
    },
    {
        "short": "no_symm_balance",
        "exp_name": "2026.5.17_opvq_2code_fromMulBasedSps_lr3e4_balance_symm001_noSymm",
        "label": "balance + no symm",
        "color": "#7c3aed",
    },
]


def parse_record(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            epoch_text, payload = line.split("-", 1)
            row = {"epoch": int(epoch_text)}
            for item in payload.split(","):
                key, value = item.split(":", 1)
                row[key] = float(value)
            row["harmonic"] = harmonic(row.get("add_acc", 0.0), row.get("mm21_acc", 0.0))
            rows.append(row)
    return rows


def harmonic(add_acc, mm21_acc):
    denom = add_acc + mm21_acc
    return 0.0 if denom == 0 else 2.0 * add_acc * mm21_acc / denom


def row_at(rows, epoch):
    for row in rows:
        if row["epoch"] == epoch:
            return row
    raise KeyError(epoch)


def best_epoch(train_rows, eval_rows):
    eval_epochs = {row["epoch"] for row in eval_rows}
    candidates = [row for row in train_rows if row["epoch"] in eval_epochs]
    if not candidates:
        raise ValueError("No overlapping train/eval epochs")
    return min(candidates, key=lambda row: row["total_loss"])["epoch"]


def stable_rows(rows):
    return [row for row in rows if row.get("total_loss", 0.0) < 1e6]


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if abs(value) >= 1e6 or (value != 0 and abs(value) < 1e-4):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def find_op_image(exp_dir, stage, epoch, op):
    result_dir = exp_dir / "1" / ("TrainingResults" if stage == "train" else "EvalResults")
    stage_name = "train" if stage == "train" else "eval"
    pattern = f"op_assignment_{stage_name}_epoch_{epoch:06d}_{op}.png"
    path = result_dir / pattern
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def copy_images(exp_dir, out_dir, short, epoch):
    asset_dir = out_dir / "assets" / short
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    images = {"train": {}, "eval": {}}
    for stage in ("train", "eval"):
        for op in ("add", "mm21"):
            src = find_op_image(exp_dir, stage, epoch, op)
            dest = asset_dir / f"{stage}_{op}.png"
            shutil.copy2(src, dest)
            images[stage][op] = str(Path("assets") / short / dest.name)
    return images


def read_config_summary(config_path):
    text = config_path.read_text(encoding="utf-8")
    def contains(snippet):
        return snippet in text
    balance = "on" if contains("'use_balance_loss': True") else "off"
    symm = "on" if contains("'use_symm_loss': True") else "off"
    include_sps = "on" if contains("'include_sps_vq_loss': True") else "off"
    symm_scalar = "0.05" if contains("'loss_scalar': 0.05") and contains("'symm':") else "0.01"
    return {
        "balance": balance,
        "symm": symm,
        "symm_scalar": symm_scalar,
        "sps_in_symm": include_sps,
    }


def build_experiment(spec, out_dir):
    exp_dir = EXPS_DIR / spec["exp_name"]
    train_rows = parse_record(exp_dir / "1" / "Train_record.txt")
    eval_rows = parse_record(exp_dir / "1" / "Eval_record.txt")
    epoch = best_epoch(train_rows, eval_rows)
    train_best = row_at(train_rows, epoch)
    eval_best = row_at(eval_rows, epoch)
    stable_eval = stable_rows(eval_rows)
    stable_train = stable_rows(train_rows)
    best_stable_eval_h = max(stable_eval, key=lambda row: row["harmonic"]) if stable_eval else None
    first_explosion = next((row["epoch"] for row in train_rows if row["total_loss"] >= 1e6), None)
    images = copy_images(exp_dir, out_dir, spec["short"], epoch)
    return {
        **spec,
        "best_epoch": epoch,
        "records": {"train": train_rows, "eval": eval_rows},
        "best": {"train": train_best, "eval": eval_best},
        "best_stable_eval_h": best_stable_eval_h,
        "first_explosion": first_explosion,
        "images": images,
        "setup": read_config_summary(exp_dir / "config.py"),
        "stable_train_min_total": min(stable_train, key=lambda row: row["total_loss"]) if stable_train else None,
    }


def markdown_analysis(exps):
    best_eval = max(exps, key=lambda exp: exp["best"]["eval"]["harmonic"])
    best_stable = max(exps, key=lambda exp: exp["best_stable_eval_h"]["harmonic"])
    no_balance = next(exp for exp in exps if exp["short"] == "symm005_sps_no_balance")
    no_symm = next(exp for exp in exps if exp["short"] == "no_symm_balance")
    no_sps = next(exp for exp in exps if exp["short"] == "symm001_no_sps_balance")
    original = next(exp for exp in exps if exp["short"] == "symm005_sps_balance")
    return f"""# opVQLearn 四实验对比分析

分析日期：2026-05-17

## 核心结论

这四次实验说明了三个点。

1. `balance loss` 是必要的。`{no_balance['label']}` 全程 hard assignment collapse 到单个 code，best checkpoint 的 eval harmonic 只有 `{fmt(no_balance['best']['eval']['harmonic'])}`。
2. 关闭 symm 后训练拟合最好，但泛化没有同步改善。`{no_symm['label']}` 的 train harmonic 到 `{fmt(no_symm['best']['train']['harmonic'])}`，但 best checkpoint 的 eval harmonic 只有 `{fmt(no_symm['best']['eval']['harmonic'])}`。
3. 降低 symm 且移除 SPS-in-symm 确实避免了原始实验的天文级 `symm_loss/sps_vq_loss` 爆炸，但 `op_vq_loss` 后期变大，eval harmonic 仍只有 `{fmt(no_sps['best']['eval']['harmonic'])}`。

按 README 的 best epoch 规则，四者中 best-checkpoint eval harmonic 最高的是 `{best_eval['label']}`，为 `{fmt(best_eval['best']['eval']['harmonic'])}`。如果只看未爆炸稳定区间的 eval harmonic 峰值，最高是 `{best_stable['label']}`，epoch `{best_stable['best_stable_eval_h']['epoch']}` 达到 `{fmt(best_stable['best_stable_eval_h']['harmonic'])}`。

## 参数差异

- 原始 `balance + symm0.05 + SPS-in-symm`：有 balance，有强 symm，并把 repeated SPS VQ loss 加进 symm 路径。
- `balanceNo`：去掉 balance，其他和原始相同。
- `symm0.01 no SPS-in-symm`：保留 balance，把 symm scalar 从 `0.05` 降到 `0.01`，且 symm 路径不再累计 SPS VQ loss。
- `no symm`：保留 balance，完全关闭 symm loss。

## 解释

`balanceNo` 的失败非常清楚：没有 balance 时，VQ codebook 没有足够压力使用两个 code，模型退化成单 code decoder。`noSymm` 的训练集结果最好，说明 symm regularization 在当前实现下不是拟合训练集所必需的；但 eval 仍然很低，说明训练拟合和 add/mm21 语义泛化之间仍然有大断层。

`{original['label']}` 在 epoch `{original['first_explosion']}` 开始数值爆炸，因此 60000 附近的 accuracy 不能作为可信泛化。`{no_sps['label']}` 没有这种天文级爆炸，支持“repeated SPS-in-symm 会放大不稳定性”的判断；但它的 `op_vq_loss` 后期升高，说明移除 SPS-in-symm 后训练仍有另一个 codebook/encoder 对齐问题。

## 下一步

- 保留 balance；没有 balance 的路线可以暂时搁置。
- 重点比较 `noSymm` 和更弱的 symm：当前 symm 没带来泛化收益，下一步应单独验证 `symm_loss_scalar=0.001` 或只在更晚 epoch 打开 symm。
- 给 OpVQ 加 hard-assignment balance 或 moving-average usage balance，因为当前 soft balance 仍允许 hard usage 偏置。
- 新增 pair-level assignment 诊断：整体 accuracy 不能说明 q1/q2 是否真的按 add/mm21 分化。
"""


def md_to_html(md):
    lines = md.strip().splitlines()
    out = []
    in_ul = False
    in_table = False
    table_rows = []

    def flush_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def flush_table():
        nonlocal in_table, table_rows
        if not in_table:
            return
        out.append("<table>")
        for i, row in enumerate(table_rows):
            cells = [html.escape(cell.strip()).replace("`", "") for cell in row.strip("|").split("|")]
            if i == 1 and all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells):
                continue
            tag = "th" if i == 0 else "td"
            out.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
        out.append("</table>")
        in_table = False
        table_rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_ul()
            flush_table()
            continue
        if stripped.startswith("|"):
            flush_ul()
            in_table = True
            table_rows.append(stripped)
            continue
        flush_table()
        if stripped.startswith("# "):
            flush_ul()
            out.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            flush_ul()
            out.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{html.escape(stripped[2:])}</li>")
        elif len(stripped) > 3 and stripped[0].isdigit() and ". " in stripped[:4]:
            flush_ul()
            out.append(f"<p>{html.escape(stripped)}</p>")
        else:
            flush_ul()
            out.append(f"<p>{html.escape(stripped)}</p>")
    flush_ul()
    flush_table()
    return "\n".join(out)


def best_table_html(exps):
    rows = []
    for exp in exps:
        train = exp["best"]["train"]
        eval_row = exp["best"]["eval"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(exp['label'])}</td>"
            f"<td>{exp['best_epoch']}</td>"
            f"<td>{fmt(train['add_acc'])}</td><td>{fmt(train['mm21_acc'])}</td><td>{fmt(train['harmonic'])}</td>"
            f"<td>{fmt(eval_row['add_acc'])}</td><td>{fmt(eval_row['mm21_acc'])}</td><td>{fmt(eval_row['harmonic'])}</td>"
            f"<td>{fmt(eval_row['code1_rate'])} / {fmt(eval_row['code2_rate'])}</td>"
            f"<td>{exp['first_explosion'] if exp['first_explosion'] is not None else 'none'}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Experiment</th><th>best epoch</th>"
        "<th>train add</th><th>train mm21</th><th>train H</th>"
        "<th>eval add</th><th>eval mm21</th><th>eval H</th>"
        "<th>eval code1/code2</th><th>first train explosion</th></tr></thead><tbody>"
        + "\n".join(rows) + "</tbody></table>"
    )


def code_usage_html(exps):
    blocks = []
    for stage in ("train", "eval"):
        for exp in exps:
            row = exp["best"][stage]
            blocks.append(f"""
            <div class="usage-card stage-block" data-stage="{stage}">
              <h3><span style="background:{exp['color']}"></span>{html.escape(exp['label'])}</h3>
              {stacked_bar('overall', row['code1_rate'], row['code2_rate'])}
              {stacked_bar('add set', row['add_code1_rate'], row['add_code2_rate'])}
              {stacked_bar('mm21 set', row['mm21_code1_rate'], row['mm21_code2_rate'])}
            </div>
            """)
    return "\n".join(blocks)


def stacked_bar(label, code1, code2):
    return f"""
    <div class="stack-row">
      <div class="stack-label">{html.escape(label)}</div>
      <div class="stack"><div class="code1" style="width:{code1 * 100:.2f}%"></div><div class="code2" style="width:{code2 * 100:.2f}%"></div></div>
      <div class="stack-num">c1 {fmt(code1)} / c2 {fmt(code2)}</div>
    </div>
    """


def images_html(exps):
    chunks = []
    for stage in ("train", "eval"):
        for exp in exps:
            add = exp["images"][stage]["add"]
            mm = exp["images"][stage]["mm21"]
            chunks.append(f"""
            <div class="viz-row stage-block" data-stage="{stage}">
              <h3>{html.escape(exp['label'])} <small>best epoch {exp['best_epoch']}</small></h3>
              <div class="viz-grid">
                <figure><img src="{add}" alt="{html.escape(exp['label'])} {stage} add" onclick="openModal(this.src)"><figcaption>add set</figcaption></figure>
                <figure><img src="{mm}" alt="{html.escape(exp['label'])} {stage} mm21" onclick="openModal(this.src)"><figcaption>mm21 set</figcaption></figure>
              </div>
            </div>
            """)
    return "\n".join(chunks)


def html_page(exps, analysis_html):
    data = {
        "experiments": [
            {
                "short": exp["short"],
                "label": exp["label"],
                "color": exp["color"],
                "bestEpoch": exp["best_epoch"],
                "best": exp["best"],
                "records": exp["records"],
                "bestStableEvalH": exp["best_stable_eval_h"],
            }
            for exp in exps
        ]
    }
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(TITLE)}</title>
<style>
body {{ margin:0; font-family: Inter, Arial, sans-serif; color:#0f172a; background:#f8fafc; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 28px 80px; }}
h1 {{ font-size:32px; margin:0 0 12px; }}
h2 {{ margin-top:34px; font-size:22px; }}
p, li {{ color:#334155; line-height:1.65; }}
table {{ border-collapse:collapse; width:100%; background:white; border:1px solid #dbe4ee; border-radius:8px; overflow:hidden; }}
th,td {{ padding:10px 12px; border-bottom:1px solid #e5edf6; text-align:left; font-size:14px; }}
th {{ background:#eef4fb; color:#0f172a; }}
.panel {{ background:white; border:1px solid #dbe4ee; border-radius:10px; padding:18px; margin:18px 0; box-shadow:0 10px 28px rgba(15,23,42,.06); }}
.stage-toggle {{ position:fixed; top:18px; right:22px; z-index:10; display:flex; gap:6px; background:white; border:1px solid #cbd5e1; border-radius:999px; padding:5px; box-shadow:0 8px 24px rgba(15,23,42,.16); }}
.stage-toggle button {{ border:0; border-radius:999px; padding:8px 14px; background:transparent; font-weight:700; color:#475569; cursor:pointer; }}
.stage-toggle button.active {{ background:#0f172a; color:white; }}
canvas {{ width:100%; height:360px; display:block; }}
.usage-grid {{ display:grid; grid-template-columns:1fr; gap:14px; }}
.usage-card {{ border:1px solid #dbe4ee; background:#fff; border-radius:8px; padding:14px; }}
.usage-card h3 {{ margin:0 0 12px; font-size:16px; display:flex; align-items:center; gap:8px; }}
.usage-card h3 span {{ display:inline-block; width:12px; height:12px; border-radius:50%; }}
.stack-row {{ display:grid; grid-template-columns:90px 1fr 150px; gap:10px; align-items:center; margin:8px 0; }}
.stack-label,.stack-num {{ font-size:13px; color:#475569; }}
.stack {{ height:20px; display:flex; background:#f1f5f9; border:1px solid #dbe4ee; border-radius:999px; overflow:hidden; }}
.code1 {{ background:#38bdf8; }}
.code2 {{ background:#f97316; }}
.legend-inline {{ font-size:13px; color:#475569; margin:8px 0 0; }}
.viz-row {{ background:white; border:1px solid #dbe4ee; border-radius:10px; padding:16px; margin:16px 0; }}
.viz-row h3 {{ margin:0 0 12px; font-size:17px; }}
.viz-row small {{ color:#64748b; font-weight:500; }}
.viz-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
figure {{ margin:0; }}
figcaption {{ color:#475569; font-size:13px; margin-top:8px; }}
img {{ width:100%; border:1px solid #dbe4ee; border-radius:8px; cursor:zoom-in; background:white; }}
.hidden {{ display:none !important; }}
.modal {{ position:fixed; inset:0; background:rgba(15,23,42,.84); display:none; align-items:center; justify-content:center; z-index:30; padding:28px; }}
.modal img {{ max-width:96vw; max-height:92vh; width:auto; cursor:zoom-out; }}
@media (max-width: 780px) {{ .viz-grid {{ grid-template-columns:1fr; }} .stack-row {{ grid-template-columns:80px 1fr; }} .stack-num {{ grid-column:2; }} }}
</style>
</head>
<body>
<div class="stage-toggle"><button id="btn-train" onclick="setStage('train')">train</button><button id="btn-eval" onclick="setStage('eval')">eval</button></div>
<main>
<section class="panel analysis">{analysis_html}</section>

<section class="panel">
<h2>Best Checkpoint Summary</h2>
{best_table_html(exps)}
</section>

<section class="panel">
<h2>Best-Epoch Accuracy</h2>
<canvas id="barChart" width="1120" height="360"></canvas>
</section>

<section class="panel">
<h2>Harmonic Mean Curves</h2>
<canvas id="curveChart" width="1120" height="360"></canvas>
</section>

<section class="panel">
<h2>Op-Code Usage at Best Epoch</h2>
<p>OpVQ 没有 query-wise accuracy 字段，因此这里展示 q/code assignment usage，作为 query-wise section 的 OpVQ 版本。</p>
<p class="legend-inline"><span style="color:#0284c7;font-weight:700">blue</span> = code1, <span style="color:#ea580c;font-weight:700">orange</span> = code2</p>
<div class="usage-grid">{code_usage_html(exps)}</div>
</section>

<section class="panel">
<h2>Data-Pair Visualization</h2>
<p>每个实验使用同一个 best epoch；train/eval 由右上角全局按钮切换。点击图片可放大。</p>
{images_html(exps)}
</section>
</main>
<div class="modal" id="modal" onclick="closeModal()"><img id="modal-img" alt="expanded visualization"></div>
<script>
const reportData = {json.dumps(data)};
let currentStage = 'eval';

function setStage(stage) {{
  currentStage = stage;
  document.querySelectorAll('.stage-block').forEach(el => el.classList.toggle('hidden', el.dataset.stage !== stage));
  document.getElementById('btn-train').classList.toggle('active', stage === 'train');
  document.getElementById('btn-eval').classList.toggle('active', stage === 'eval');
  drawBars();
  drawCurves();
}}

function harmonic(row) {{
  return row.harmonic || 0;
}}

function drawAxes(ctx, plot, yMax, maxEpoch) {{
  ctx.strokeStyle = '#cbd5e1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plot.left, plot.top);
  ctx.lineTo(plot.left, plot.bottom);
  ctx.lineTo(plot.right, plot.bottom);
  ctx.stroke();
  ctx.fillStyle = '#64748b';
  ctx.font = '12px Arial';
  for (let i = 0; i <= 4; i++) {{
    const y = plot.bottom - (plot.bottom - plot.top) * i / 4;
    const val = yMax * i / 4;
    ctx.fillText(val.toFixed(2), 8, y + 4);
    ctx.strokeStyle = '#eef2f7';
    ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.right, y); ctx.stroke();
  }}
  for (let i = 0; i <= 4; i++) {{
    const x = plot.left + (plot.right - plot.left) * i / 4;
    const val = Math.round(maxEpoch * i / 4);
    ctx.fillText(String(val), x - 14, plot.bottom + 24);
  }}
}}

function drawBars() {{
  const canvas = document.getElementById('barChart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const plot = {{left:54, right:canvas.width - 30, top:24, bottom:canvas.height - 54}};
  const metrics = [
    ['add_acc', 'add_acc'],
    ['mm21_acc', 'mm21_acc'],
    ['harmonic', 'harmonic mean'],
  ];
  drawAxes(ctx, plot, 1, 1);
  const groupW = (plot.right - plot.left) / metrics.length;
  const barW = Math.min(42, groupW / (reportData.experiments.length + 2));
  metrics.forEach((metric, gi) => {{
    const cx = plot.left + groupW * gi + groupW / 2;
    ctx.fillStyle = '#0f172a';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(metric[1], cx, canvas.height - 16);
    reportData.experiments.forEach((exp, ei) => {{
      const row = exp.best[currentStage];
      const val = metric[0] === 'harmonic' ? harmonic(row) : row[metric[0]];
      const x = cx - (reportData.experiments.length * barW) / 2 + ei * barW;
      const h = (plot.bottom - plot.top) * val;
      ctx.fillStyle = exp.color;
      ctx.fillRect(x, plot.bottom - h, barW * 0.78, h);
    }});
  }});
  const lx = plot.right - 300;
  reportData.experiments.forEach((exp, i) => {{
    const y = plot.top + i * 22;
    ctx.fillStyle = exp.color; ctx.fillRect(lx, y, 12, 12);
    ctx.fillStyle = '#334155'; ctx.textAlign = 'left'; ctx.font = '12px Arial';
    ctx.fillText(exp.label, lx + 18, y + 11);
  }});
}}

function drawCurves() {{
  const canvas = document.getElementById('curveChart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const plot = {{left:54, right:canvas.width - 330, top:24, bottom:canvas.height - 54}};
  const maxEpoch = Math.max(...reportData.experiments.flatMap(exp => exp.records[currentStage].map(r => r.epoch)));
  drawAxes(ctx, plot, 1, maxEpoch);
  reportData.experiments.forEach((exp, i) => {{
    const rows = exp.records[currentStage].filter(r => Number.isFinite(r.harmonic) && r.total_loss < 1e6);
    ctx.strokeStyle = exp.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    rows.forEach((row, idx) => {{
      const x = plot.left + (plot.right - plot.left) * row.epoch / maxEpoch;
      const y = plot.bottom - (plot.bottom - plot.top) * row.harmonic;
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
    const y = plot.top + i * 24;
    ctx.fillStyle = exp.color; ctx.fillRect(plot.right + 24, y, 12, 12);
    ctx.fillStyle = '#334155'; ctx.font = '12px Arial';
    ctx.fillText(exp.label, plot.right + 42, y + 11);
  }});
  ctx.fillStyle = '#64748b';
  ctx.font = '12px Arial';
  ctx.fillText('harmonic mean; rows with total_loss >= 1e6 are omitted from curves', plot.left, canvas.height - 16);
}}

function openModal(src) {{
  document.getElementById('modal-img').src = src;
  document.getElementById('modal').style.display = 'flex';
}}
function closeModal() {{ document.getElementById('modal').style.display = 'none'; }}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});
setStage('eval');
</script>
</body>
</html>
"""


def write_redirect(report_name):
    target = f"{report_name}/index.html"
    redirect_path = ANALYSIS_DIR / f"{report_name}.html"
    redirect_path.write_text(
        f"<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='0; url={target}'>"
        f"<p>Redirecting to <a href='{target}'>{html.escape(target)}</a></p>\n",
        encoding="utf-8",
    )


def main():
    out_dir = ANALYSIS_DIR / REPORT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    exps = [build_experiment(spec, out_dir) for spec in EXPERIMENTS]
    analysis_md = markdown_analysis(exps)
    (out_dir / "analysis.md").write_text(analysis_md, encoding="utf-8")
    (out_dir / "index.html").write_text(html_page(exps, md_to_html(analysis_md)), encoding="utf-8")
    write_redirect(REPORT_NAME)
    print(out_dir / "index.html")
    for exp in exps:
        print(exp["short"], "best_epoch", exp["best_epoch"], "eval_h", fmt(exp["best"]["eval"]["harmonic"]))


if __name__ == "__main__":
    main()
