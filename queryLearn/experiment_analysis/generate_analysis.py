#!/usr/bin/env python3
import argparse
import html
import importlib.util
import json
import re
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
EXPS_DIR = BASE_DIR / "exps"

DEFAULT_REPORT_NAME = "compare_sanity_check_2026-05-10"
DEFAULT_TITLE = "queryLearn sanity-check 对比报告"
DEFAULT_EXPERIMENTS = [
    {
        "short": "mul-concat",
        "exp_name": "2026.4.27_2dimquery_symm1_sanityCheck_fromMulBasedSps_fc_5_1024",
        "label": "2026.4.27 concat / mul-based SPS",
        "color": "#64748b",
    },
    {
        "short": "add-concat",
        "exp_name": "2026.5.08_2dimquery_symm1_sanityCheck_fromAddBasedSps_fc_5_1024",
        "label": "2026.5.08 concat / add-based SPS",
        "color": "#b45309",
    },
    {
        "short": "film-mul",
        "exp_name": "2026.5.10_film_2dimQ_sanityCheck_fromMulBasedSps_fc_5_1024",
        "label": "2026.5.10 FiLM / mul-based SPS",
        "color": "#0f766e",
    },
]


def parse_record(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            epoch_text, payload = line.split("-", 1)
            row = {"epoch": int(epoch_text)}
            for item in payload.split(","):
                key, value = item.split(":", 1)
                row[key] = float(value)
            records.append(row)
    return records


def load_config_summary(path):
    spec = importlib.util.spec_from_file_location(f"analysis_config_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        raise ValueError(f"CONFIG not found in {path}")

    operator = config.get("operator", {})
    query = config.get("query_learner", {})
    operator_mode = operator.get("condition_mode", "concat")
    return {
        "sps": config.get("VQSPS", {}).get("EXP_NAME", "unknown"),
        "operator": (
            f"{operator_mode}, hidden_layers={operator.get('n_hidden_layers', 'unknown')}, "
            f"unit={operator.get('unit', 'unknown')}"
        ),
        "query": (
            f"query_dim={query.get('query_dim', query.get('in_dim', 'default'))}, "
            f"train_queries={query.get('train_queries', False)}"
        ),
        "sanity": str(config.get("sanity_check", False)),
    }


def harmonic(add_acc, mul_acc):
    denom = add_acc + mul_acc
    return 0.0 if denom == 0 else 2 * add_acc * mul_acc / denom


def best_epoch(train_records, eval_records, selector_key):
    eval_epochs = {row["epoch"] for row in eval_records}
    candidates = [row for row in train_records if row["epoch"] in eval_epochs]
    if not candidates:
        raise ValueError("No overlapping epochs between train and eval records")
    return min(candidates, key=lambda row: row[selector_key])["epoch"]


def find_query_image(exp_dir, stage, epoch, query_name):
    result_dir = exp_dir / "1" / ("TrainingResults" if stage == "train" else "EvalResults")
    stage_name = "train" if stage == "train" else "eval"
    pattern = f"query_operation_{stage_name}_epoch_{epoch:06d}_{query_name}_*.png"
    matches = sorted(result_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one image for {exp_dir.name} {stage} epoch {epoch} {query_name}, "
            f"found {len(matches)} with pattern {pattern}"
        )
    return matches[0]


def copy_query_images(exp_dir, out_dir, short, epoch):
    copied = {"train": [], "eval": []}
    asset_dir = out_dir / "assets" / short
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    for stage in ("train", "eval"):
        for query_name in ("q1", "q2"):
            src = find_query_image(exp_dir, stage, epoch, query_name)
            dest_name = f"{stage}_{query_name}.png"
            dest = asset_dir / dest_name
            shutil.copy2(src, dest)
            copied[stage].append(str(Path("assets") / short / dest_name))
    return copied


def row_at(records, epoch):
    for row in records:
        if row["epoch"] == epoch:
            return row
    raise KeyError(epoch)


def build_experiment(spec, out_dir, selector_key):
    exp_dir = EXPS_DIR / spec["exp_name"]
    sub_exp = exp_dir / "1"
    train_records = parse_record(sub_exp / "Train_record.txt")
    eval_records = parse_record(sub_exp / "Eval_record.txt")
    epoch = best_epoch(train_records, eval_records, selector_key)
    images = copy_query_images(exp_dir, out_dir, spec["short"], epoch)
    config = load_config_summary(exp_dir / "config.py")
    return {
        "short": spec["short"],
        "label": spec["label"],
        "color": spec["color"],
        "exp_name": spec["exp_name"],
        "best_epoch": epoch,
        "best_selector": selector_key,
        "setup": config,
        "records": {
            "train": train_records,
            "eval": eval_records,
        },
        "images": images,
    }


def parse_experiment_arg(value):
    parts = value.split("|")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--experiment must be 'short|experiment_dir_name|Display label|#color'"
        )
    short, exp_name, label, color = parts
    return {
        "short": short.strip(),
        "exp_name": exp_name.strip(),
        "label": label.strip(),
        "color": color.strip(),
    }


def render_index(title, report_name, experiments, selector_key):
    payload = json.dumps(
        {
            "title": title,
            "reportName": report_name,
            "selectorKey": selector_key,
            "experiments": experiments,
        },
        ensure_ascii=False,
    )
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_title}</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #607086;
      --line: #d7deea;
      --panel: #f8fafc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #fff;
    }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 28px 28px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 34px 0 12px; font-size: 21px; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; }}
    p, li {{ line-height: 1.55; }}
    code {{ background: #edf2f7; padding: 2px 5px; border-radius: 5px; font-size: 0.92em; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 18px; }}
    .card, .chart-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .chart-panel {{ background: #fff; margin-top: 14px; }}
    .metric {{ font-size: 30px; font-weight: 700; margin: 8px 0 2px; }}
    .metric small {{ font-size: 14px; color: var(--muted); font-weight: 400; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #334155; background: #f8fafc; }}
    canvas {{ width: 100%; height: 340px; display: block; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 12px; }}
    .tab-btn {{ border: 1px solid var(--line); background: #f8fafc; color: #334155; border-radius: 7px; padding: 7px 12px; font-size: 14px; cursor: pointer; }}
    .tab-btn.active {{ background: #172033; color: #fff; border-color: #172033; }}
    .legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 10px 0 0; color: var(--muted); font-size: 13px; }}
    .legend-title {{ color: #334155; font-weight: 700; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; min-height: 18px; }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; }}
    .line-sample {{ display: inline-block; width: 34px; border-top: 3px solid #334155; transform: translateY(-1px); }}
    .line-dashed {{ border-top-style: dashed; }}
    .note {{ border-left: 4px solid #0f766e; background: #f0fdfa; padding: 12px 14px; margin-top: 16px; border-radius: 0 8px 8px 0; }}
    .query-list {{ display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 12px; }}
    .query-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; }}
    .query-card h3 {{ margin-bottom: 4px; }}
    .query-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px; }}
    .query-pair figure {{ margin: 0; }}
    .query-img-btn {{ display: block; width: 100%; padding: 0; border: 0; background: transparent; cursor: zoom-in; }}
    .query-pair img {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; display: block; }}
    .lightbox {{ position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(15, 23, 42, 0.86); z-index: 1000; padding: 28px; }}
    .lightbox.open {{ display: flex; }}
    .lightbox-inner {{ max-width: min(96vw, 1500px); max-height: 94vh; width: 100%; }}
    .lightbox img {{ max-width: 100%; max-height: 86vh; display: block; margin: 0 auto; background: #fff; border-radius: 8px; }}
    .lightbox-caption {{ color: #e2e8f0; text-align: center; margin-top: 10px; font-size: 14px; }}
    .lightbox-close {{ position: fixed; top: 18px; right: 22px; border: 1px solid rgba(255,255,255,0.5); color: #fff; background: rgba(15,23,42,0.4); border-radius: 7px; padding: 7px 11px; cursor: pointer; font-size: 14px; }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-top: 5px; text-align: center; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 3px 8px; background: #e2e8f0; color: #334155; font-size: 12px; margin-top: 3px; }}
    @media (max-width: 900px) {{ .grid, .query-pair {{ grid-template-columns: 1fr; }} main {{ padding: 22px 16px 44px; }} }}
  </style>
</head>
<body>
<main>
  <h1>{escaped_title}</h1>
  <p class="muted">Best epoch 规则：在 Eval_record 可见的 epoch 中，选择对应 Train_record 的 <code>{html.escape(selector_key)}</code> 最低者。图片已复制到本报告目录的 <code>assets/</code> 中，可离线查看。</p>

  <section class="grid">
    <div class="card"><strong>best epoch 对齐</strong><div class="metric">train loss <small>selector</small></div><p class="muted">柱状图、summary 和 query 图片都用同一个 best epoch。</p></div>
    <div class="card"><strong>柱状图结构</strong><div class="metric">3 <small>metric groups</small></div><p class="muted">三组分别是 add_acc、mul_acc、harmonic mean；每组内每个实验一根柱。</p></div>
    <div class="card"><strong>页签切换</strong><div class="metric">train / eval</div><p class="muted">所有结果 section 都可切换 train/eval；best epoch 不随页签变化。</p></div>
  </section>

  <h2>实验设置</h2>
  <table id="setup-table"></table>

  <h2>Metric Curves</h2>
  <section class="chart-panel">
    <div class="tabs" data-tabs></div>
    <h3>add_acc / mul_acc over saved epochs</h3>
    <canvas id="lineChart" width="1120" height="340"></canvas>
    <div class="legend" id="lineLegend"></div>
  </section>

  <h2>Best Epoch Bar Chart</h2>
  <section class="chart-panel">
    <div class="tabs" data-tabs></div>
    <h3>best checkpoint metrics selected by lowest train {html.escape(selector_key)}</h3>
    <canvas id="barChart" width="1120" height="340"></canvas>
    <div class="legend" id="barLegend"></div>
  </section>

  <h2>Query Pairwise Visualization</h2>
  <section class="chart-panel">
    <div class="tabs" data-tabs></div>
    <p class="muted">展示每个实验 best epoch 的 q1 / q2 operation table。图片来自本报告目录下的 assets。</p>
    <div id="queryImages" class="query-list"></div>
  </section>

  <h2>Best Epoch Summary</h2>
  <section class="chart-panel">
    <div class="tabs" data-tabs></div>
    <table id="summary-table"></table>
  </section>

  <h2>Raw Saved-Epoch Data</h2>
  <section class="chart-panel">
    <div class="tabs" data-tabs></div>
    <table id="raw-table"></table>
  </section>

  <h2>Interpretation</h2>
  <div class="note">
    <strong>读取方式</strong>
    <p>柱状图按 train {html.escape(selector_key)} 选出的 checkpoint 读取 train/eval 指标。query pairwise 图片也使用同一 checkpoint，避免图表和图片各自挑最优点。</p>
  </div>
</main>
<div id="lightbox" class="lightbox" aria-hidden="true">
  <button id="lightboxClose" class="lightbox-close" type="button">Close</button>
  <div class="lightbox-inner">
    <img id="lightboxImg" src="" alt="">
    <div id="lightboxCaption" class="lightbox-caption"></div>
  </div>
</div>

<script>
const REPORT = {payload};
const experiments = REPORT.experiments;
let activeStage = "eval";

function harmonic(addAcc, mulAcc) {{
  const denom = addAcc + mulAcc;
  return denom === 0 ? 0 : 2 * addAcc * mulAcc / denom;
}}
function fmt(x) {{ return Number(x).toFixed(3); }}
function rowAt(exp, stage, epoch) {{ return exp.records[stage].find(row => row.epoch === epoch); }}
function setupCanvas(canvas) {{
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return {{ctx, w: rect.width, h: rect.height}};
}}
function drawAxes(ctx, plot, xTicks, yTicks) {{
  ctx.strokeStyle = "#d7deea";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y);
  ctx.lineTo(plot.x, plot.y + plot.h);
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
  ctx.stroke();
  ctx.fillStyle = "#607086";
  ctx.font = "12px Arial";
  yTicks.forEach(t => {{
    const y = plot.y + plot.h - t * plot.h;
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
    ctx.fillText(t.toFixed(1), plot.x - 34, y + 4);
  }});
  xTicks.forEach(t => {{
    const x = plot.x + (t / 50000) * plot.w;
    ctx.fillText(String(t / 1000) + "k", x - 10, plot.y + plot.h + 22);
  }});
}}
function renderTabs() {{
  document.querySelectorAll("[data-tabs]").forEach(el => {{
    el.innerHTML = `
      <button class="tab-btn ${{activeStage === "train" ? "active" : ""}}" data-stage="train">Train</button>
      <button class="tab-btn ${{activeStage === "eval" ? "active" : ""}}" data-stage="eval">Eval</button>
    `;
  }});
  document.querySelectorAll(".tab-btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      activeStage = btn.dataset.stage;
      renderAll();
    }});
  }});
}}
function renderLegends() {{
  const expLegend = experiments.map(exp => `<span class="legend-item"><span class="swatch" style="background:${{exp.color}}"></span>${{exp.short}}</span>`).join("");
  document.getElementById("lineLegend").innerHTML = `
    <span class="legend-title">颜色=实验</span>${{expLegend}}
    <span class="legend-title">线型=指标</span>
    <span class="legend-item"><span class="line-sample"></span>add_acc</span>
    <span class="legend-item"><span class="line-sample line-dashed"></span>mul_acc</span>
  `;
  document.getElementById("barLegend").innerHTML = `<span class="legend-title">颜色=实验</span>${{expLegend}}`;
}}
function renderSetup() {{
  const table = document.getElementById("setup-table");
  table.innerHTML = `<thead><tr><th>experiment</th><th>SPS init</th><th>operator</th><th>query</th><th>best epoch</th></tr></thead>` +
    `<tbody>` + experiments.map(exp => {{
      const trainRow = rowAt(exp, "train", exp.best_epoch);
      return `<tr>
        <td><strong style="color:${{exp.color}}">${{exp.short}}</strong><br><span class="muted">${{exp.label}}</span></td>
        <td>${{exp.setup.sps}}</td>
        <td>${{exp.setup.operator}}</td>
        <td>${{exp.setup.query}}</td>
        <td><span class="pill">epoch ${{exp.best_epoch}}</span><br><span class="muted">train ${{REPORT.selectorKey}}=${{fmt(trainRow[REPORT.selectorKey])}}</span></td>
      </tr>`;
    }}).join("") + `</tbody>`;
}}
function drawLineChart() {{
  const {{ctx, w, h}} = setupCanvas(document.getElementById("lineChart"));
  ctx.clearRect(0, 0, w, h);
  const plot = {{x: 56, y: 18, w: w - 230, h: h - 62}};
  drawAxes(ctx, plot, [0, 10000, 20000, 30000, 40000, 50000], [0, 0.25, 0.5, 0.75, 1.0]);
  const labels = [];
  experiments.forEach(exp => {{
    [["add_acc", "add"], ["mul_acc", "mul"]].forEach(([key, label]) => {{
      ctx.save();
      ctx.strokeStyle = exp.color;
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 2.5;
      ctx.setLineDash(key === "mul_acc" ? [7, 5] : []);
      ctx.beginPath();
      exp.records[activeStage].forEach((row, i) => {{
        const x = plot.x + (row.epoch / 50000) * plot.w;
        const y = plot.y + plot.h - row[key] * plot.h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.restore();
      const last = exp.records[activeStage][exp.records[activeStage].length - 1];
      labels.push({{
        y: plot.y + plot.h - last[key] * plot.h,
        color: exp.color,
        dash: key === "mul_acc" ? [7, 5] : [],
        text: `${{exp.short}} ${{label}}`
      }});
    }});
  }});
  labels.sort((a, b) => a.y - b.y);
  const minGap = 17;
  for (let i = 1; i < labels.length; i++) {{
    if (labels[i].y - labels[i - 1].y < minGap) labels[i].y = labels[i - 1].y + minGap;
  }}
  for (let i = labels.length - 2; i >= 0; i--) {{
    if (labels[i + 1].y > plot.y + plot.h) labels[i + 1].y = plot.y + plot.h;
    if (labels[i + 1].y - labels[i].y < minGap) labels[i].y = labels[i + 1].y - minGap;
  }}
  labels.forEach(label => {{
    const x0 = plot.x + plot.w + 10;
    ctx.save();
    ctx.strokeStyle = label.color;
    ctx.lineWidth = 3;
    ctx.setLineDash(label.dash);
    ctx.beginPath();
    ctx.moveTo(x0, label.y - 4);
    ctx.lineTo(x0 + 28, label.y - 4);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = label.color;
    ctx.font = "12px Arial";
    ctx.fillText(label.text, x0 + 36, label.y);
    ctx.restore();
  }});
}}
function drawBarChart() {{
  const {{ctx, w, h}} = setupCanvas(document.getElementById("barChart"));
  ctx.clearRect(0, 0, w, h);
  const plot = {{x: 64, y: 24, w: w - 100, h: h - 78}};
  drawAxes(ctx, plot, [], [0, 0.25, 0.5, 0.75, 1.0]);
  const groups = [
    {{key: "add_acc", label: "add_acc"}},
    {{key: "mul_acc", label: "mul_acc"}},
    {{key: "harmonic", label: "harmonic mean"}}
  ];
  const groupW = plot.w / groups.length;
  groups.forEach((group, groupIdx) => {{
    const barW = Math.min(54, groupW / 7);
    const gap = 12;
    const totalW = experiments.length * barW + (experiments.length - 1) * gap;
    const start = plot.x + groupIdx * groupW + groupW / 2 - totalW / 2;
    experiments.forEach((exp, expIdx) => {{
      const row = rowAt(exp, activeStage, exp.best_epoch);
      const value = group.key === "harmonic" ? harmonic(row.add_acc, row.mul_acc) : row[group.key];
      const x = start + expIdx * (barW + gap);
      const bh = value * plot.h;
      ctx.fillStyle = exp.color;
      ctx.fillRect(x, plot.y + plot.h - bh, barW, bh);
      ctx.fillStyle = "#172033";
      ctx.font = "12px Arial";
      ctx.fillText(fmt(value), x - 1, plot.y + plot.h - bh - 7);
    }});
    ctx.fillStyle = "#334155";
    ctx.font = "13px Arial";
    ctx.fillText(group.label, plot.x + groupIdx * groupW + groupW / 2 - 38, plot.y + plot.h + 28);
  }});
}}
function renderQueryImages() {{
  const root = document.getElementById("queryImages");
  root.innerHTML = experiments.map(exp => {{
    const row = rowAt(exp, activeStage, exp.best_epoch);
    const imgs = exp.images[activeStage];
    return `<article class="query-card">
      <h3 style="color:${{exp.color}}">${{exp.short}}</h3>
      <div class="muted">best epoch ${{exp.best_epoch}}; ${{activeStage}} add=${{fmt(row.add_acc)}}, mul=${{fmt(row.mul_acc)}}, H=${{fmt(harmonic(row.add_acc, row.mul_acc))}}</div>
      <div class="query-pair">
        <figure>
          <button class="query-img-btn" type="button" data-full-src="${{imgs[0]}}" data-caption="${{exp.short}} ${{activeStage}} q1, epoch ${{exp.best_epoch}}">
            <img src="${{imgs[0]}}" alt="${{exp.short}} ${{activeStage}} q1">
          </button>
          <figcaption>q1 - click to enlarge</figcaption>
        </figure>
        <figure>
          <button class="query-img-btn" type="button" data-full-src="${{imgs[1]}}" data-caption="${{exp.short}} ${{activeStage}} q2, epoch ${{exp.best_epoch}}">
            <img src="${{imgs[1]}}" alt="${{exp.short}} ${{activeStage}} q2">
          </button>
          <figcaption>q2 - click to enlarge</figcaption>
        </figure>
      </div>
    </article>`;
  }}).join("");
  bindLightboxButtons();
}}
function renderSummary() {{
  const table = document.getElementById("summary-table");
  table.innerHTML = `<thead><tr><th>experiment</th><th>best epoch</th><th>train selector</th><th>${{activeStage}} add_acc</th><th>${{activeStage}} mul_acc</th><th>${{activeStage}} harmonic</th></tr></thead>` +
    `<tbody>` + experiments.map(exp => {{
      const trainRow = rowAt(exp, "train", exp.best_epoch);
      const row = rowAt(exp, activeStage, exp.best_epoch);
      return `<tr>
        <td><strong style="color:${{exp.color}}">${{exp.short}}</strong></td>
        <td>${{exp.best_epoch}}</td>
        <td>train ${{REPORT.selectorKey}}=${{fmt(trainRow[REPORT.selectorKey])}}</td>
        <td>${{fmt(row.add_acc)}}</td>
        <td>${{fmt(row.mul_acc)}}</td>
        <td>${{fmt(harmonic(row.add_acc, row.mul_acc))}}</td>
      </tr>`;
    }}).join("") + `</tbody>`;
}}
function renderRaw() {{
  const table = document.getElementById("raw-table");
  const rows = experiments.flatMap(exp => exp.records[activeStage].map(row => ({{exp, row}})));
  table.innerHTML = `<thead><tr><th>experiment</th><th>epoch</th><th>total_loss</th><th>add_acc</th><th>mul_acc</th><th>harmonic</th><th>best?</th></tr></thead>` +
    `<tbody>` + rows.map(({{exp, row}}) => {{
      const isBest = row.epoch === exp.best_epoch;
      return `<tr>
        <td><strong style="color:${{exp.color}}">${{exp.short}}</strong></td>
        <td>${{row.epoch}}</td>
        <td>${{fmt(row.total_loss)}}</td>
        <td>${{fmt(row.add_acc)}}</td>
        <td>${{fmt(row.mul_acc)}}</td>
        <td>${{fmt(harmonic(row.add_acc, row.mul_acc))}}</td>
        <td>${{isBest ? '<span class="pill">selected</span>' : ''}}</td>
      </tr>`;
    }}).join("") + `</tbody>`;
}}
function renderAll() {{
  renderTabs();
  renderLegends();
  renderSetup();
  drawLineChart();
  drawBarChart();
  renderQueryImages();
  renderSummary();
  renderRaw();
}}
function openLightbox(src, caption) {{
  const box = document.getElementById("lightbox");
  const img = document.getElementById("lightboxImg");
  const cap = document.getElementById("lightboxCaption");
  img.src = src;
  img.alt = caption;
  cap.textContent = caption;
  box.classList.add("open");
  box.setAttribute("aria-hidden", "false");
}}
function closeLightbox() {{
  const box = document.getElementById("lightbox");
  const img = document.getElementById("lightboxImg");
  box.classList.remove("open");
  box.setAttribute("aria-hidden", "true");
  img.src = "";
}}
function bindLightboxButtons() {{
  document.querySelectorAll(".query-img-btn").forEach(btn => {{
    btn.addEventListener("click", () => openLightbox(btn.dataset.fullSrc, btn.dataset.caption));
  }});
}}
document.getElementById("lightboxClose").addEventListener("click", closeLightbox);
document.getElementById("lightbox").addEventListener("click", event => {{
  if (event.target.id === "lightbox") closeLightbox();
}});
document.addEventListener("keydown", event => {{
  if (event.key === "Escape") closeLightbox();
}});
window.addEventListener("resize", () => {{
  drawLineChart();
  drawBarChart();
}});
renderAll();
</script>
</body>
</html>
"""


def render_redirect(report_name):
    target = f"{report_name}/index.html"
    escaped = html.escape(target)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="0; url={escaped}" />
  <title>Redirecting to {escaped}</title>
</head>
<body>
  <p>Report moved to <a href="{escaped}">{escaped}</a>.</p>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate self-contained queryLearn analysis reports.")
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--selector-key", default="total_loss")
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment_arg,
        help="Repeatable. Format: 'short|experiment_dir_name|Display label|#color'",
    )
    args = parser.parse_args()

    specs = args.experiment if args.experiment else DEFAULT_EXPERIMENTS
    out_dir = ANALYSIS_DIR / args.report_name
    assets_dir = out_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [build_experiment(spec, out_dir, args.selector_key) for spec in specs]
    index_html = render_index(args.title, args.report_name, experiments, args.selector_key)
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")
    (ANALYSIS_DIR / f"{args.report_name}.html").write_text(render_redirect(args.report_name), encoding="utf-8")

    print(f"Wrote {out_dir / 'index.html'}")
    for exp in experiments:
        print(f"{exp['short']}: best_epoch={exp['best_epoch']}")


if __name__ == "__main__":
    main()
