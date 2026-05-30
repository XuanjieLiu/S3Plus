#!/usr/bin/env python3
import argparse
import csv
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
        "short": "mm21-concat",
        "exp_name": "2026.4.27_2dimquery_symm1_sanityCheck_fromMulBasedSps_fc_5_1024",
        "label": "2026.4.27 concat / mm21-based SPS",
        "color": "#64748b",
    },
    {
        "short": "add-concat",
        "exp_name": "2026.5.08_2dimquery_symm1_sanityCheck_fromAddBasedSps_fc_5_1024",
        "label": "2026.5.08 concat / add-based SPS",
        "color": "#b45309",
    },
    {
        "short": "film-mm21",
        "exp_name": "2026.5.10_film_2dimQ_sanityCheck_fromMulBasedSps_fc_5_1024",
        "label": "2026.5.10 FiLM / mm21-based SPS",
        "color": "#0f766e",
    },
]


def normalize_record_keys(row):
    if "add_acc_q0" in row:
        old_add_q1 = row.get("add_acc_q1")
        row["add_acc_q1"] = row["add_acc_q0"]
        if "add_acc_q2" not in row and old_add_q1 is not None:
            row["add_acc_q2"] = old_add_q1
    elif "add_acc_q2" not in row and "add_acc_q1" in row:
        row["add_acc_q2"] = row["add_acc_q1"]

    if "mul_acc_q0" in row:
        row["mm21_acc_q1"] = row["mul_acc_q0"]
    if "mul_acc_q1" in row:
        row["mm21_acc_q2"] = row["mul_acc_q1"]
    if "mul_acc" in row:
        row["mm21_acc"] = row["mul_acc"]
    if "mul_total" in row:
        row["mm21_total"] = row["mul_total"]

    for legacy_key in ("add_acc_q0", "mul_acc_q0", "mul_acc_q1", "mul_acc", "mul_total"):
        row.pop(legacy_key, None)
    return row


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
            normalize_record_keys(row)
            records.append(row)
    return records


def parse_optional_record(path):
    if not path.exists():
        return []
    return parse_record(path)


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
    optimizer = config.get("optimizer", "adam")
    init_queries = query.get("init_queries", "default")
    return {
        "sps": config.get("VQSPS", {}).get("EXP_NAME", "unknown"),
        "operator": (
            f"{operator_mode}, hidden_layers={operator.get('n_hidden_layers', 'unknown')}, "
            f"unit={operator.get('unit', 'unknown')}"
        ),
        "query": (
            f"query_dim={query.get('query_dim', query.get('in_dim', 'default'))}, "
            f"train_queries={query.get('train_queries', False)}, init={init_queries}"
        ),
        "training": (
            f"lr={config.get('learning_rate', 'unknown')}, optimizer={optimizer}, "
            f"weight_decay={config.get('weight_decay', 0)}, grad_clip={config.get('grad_clip_norm', 'none')}, "
            f"symm_loss={config.get('symm_loss_scalar', 'unknown')}"
        ),
        "sanity": str(config.get("sanity_check", False)),
    }


def harmonic(add_acc, mm21_acc):
    denom = add_acc + mm21_acc
    return 0.0 if denom == 0 else 2 * add_acc * mm21_acc / denom


def best_epoch(train_records, eval_records, selector_key, allowed_epochs=None):
    eval_epochs = {row["epoch"] for row in eval_records}
    candidates = [row for row in train_records if row["epoch"] in eval_epochs] if eval_epochs else train_records
    if allowed_epochs is not None:
        candidates = [row for row in candidates if row["epoch"] in allowed_epochs]
    if not candidates:
        raise ValueError("No train records available for best epoch selection")
    return min(candidates, key=lambda row: row[selector_key])["epoch"]


def find_operation_image(exp_dir, stage, epoch, operation, sub_exp_id="1"):
    result_dir = exp_dir / str(sub_exp_id) / ("TrainingResults" if stage == "train" else "EvalResults")
    stage_name = "train" if stage == "train" else "eval"
    pattern = f"query_operation_{stage_name}_epoch_{epoch:06d}_{operation}.*"
    matches = sorted(result_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one image for {exp_dir.name} {stage} epoch {epoch} {operation}, "
            f"found {len(matches)} with pattern {pattern}"
        )
    return matches[0]


def operation_image_epochs(exp_dir, stage, sub_exp_id="1"):
    result_dir = exp_dir / str(sub_exp_id) / ("TrainingResults" if stage == "train" else "EvalResults")
    stage_name = "train" if stage == "train" else "eval"
    per_operation = []
    for operation in ("add", "mm21"):
        epochs = set()
        pattern = f"query_operation_{stage_name}_epoch_*_{operation}.*"
        for path in result_dir.glob(pattern):
            match = re.search(r"_epoch_(\d+)_" + re.escape(operation) + r"\.", path.name)
            if match:
                epochs.add(int(match.group(1)))
        per_operation.append(epochs)
    if not per_operation or not all(per_operation):
        return set()
    return set.intersection(*per_operation)


def available_operation_image_epochs(exp_dir, stages, sub_exp_id="1"):
    stage_epochs = [operation_image_epochs(exp_dir, stage, sub_exp_id) for stage in stages]
    if not stage_epochs or not all(stage_epochs):
        return set()
    return set.intersection(*stage_epochs)


def copy_operation_images(exp_dir, out_dir, short, epoch, sub_exp_id="1", nested=False, stages=("train", "eval")):
    copied = {"train": [], "eval": []}
    asset_dir = out_dir / "assets" / short
    if nested:
        asset_dir = asset_dir / f"sub{sub_exp_id}"
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        for operation in ("add", "mm21"):
            src = find_operation_image(exp_dir, stage, epoch, operation, sub_exp_id)
            dest_name = f"{stage}_{operation}{src.suffix.lower()}"
            dest = asset_dir / dest_name
            shutil.copy2(src, dest)
            if nested:
                rel_path = Path("assets") / short / f"sub{sub_exp_id}" / dest_name
            else:
                rel_path = Path("assets") / short / dest_name
            copied[stage].append(str(rel_path))
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
    eval_records = parse_optional_record(sub_exp / "Eval_record.txt")
    stages = ("train", "eval") if eval_records else ("train",)
    allowed_epochs = available_operation_image_epochs(exp_dir, stages, "1")
    epoch = best_epoch(train_records, eval_records, selector_key, allowed_epochs or None)
    images = copy_operation_images(exp_dir, out_dir, spec["short"], epoch, stages=stages)
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
        "available_stages": list(stages),
        "images": images,
    }


def parse_sub_exps(value):
    sub_exps = [item.strip() for item in value.split(",") if item.strip()]
    if not sub_exps:
        raise argparse.ArgumentTypeError("--sub-exps must contain at least one sub-exp id")
    return sub_exps


def row_harmonic(row):
    return harmonic(row["add_acc"], row["mm21_acc"])


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def build_run(exp_dir, out_dir, short, sub_exp_id, selector_key):
    sub_exp = exp_dir / str(sub_exp_id)
    train_records = parse_record(sub_exp / "Train_record.txt")
    eval_records = parse_optional_record(sub_exp / "Eval_record.txt")
    stages = ("train", "eval") if eval_records else ("train",)
    allowed_epochs = available_operation_image_epochs(exp_dir, stages, sub_exp_id)
    epoch = best_epoch(train_records, eval_records, selector_key, allowed_epochs or None)
    images = copy_operation_images(exp_dir, out_dir, short, epoch, sub_exp_id, nested=True, stages=stages)
    best_eval_h_row = max(eval_records, key=row_harmonic) if eval_records else None
    return {
        "sub_id": str(sub_exp_id),
        "best_epoch": epoch,
        "best_selector": selector_key,
        "best_eval_h_epoch": best_eval_h_row["epoch"] if best_eval_h_row else None,
        "best_eval_h": row_harmonic(best_eval_h_row) if best_eval_h_row else None,
        "records": {
            "train": train_records,
            "eval": eval_records,
        },
        "available_stages": list(stages),
        "images": images,
    }


def summarize_runs(runs):
    summary = {}
    for stage in ("train", "eval"):
        if not any(run["records"][stage] for run in runs):
            continue
        summary[stage] = {}
        for key in (
                "add_acc",
                "mm21_acc",
                "harmonic",
                "add_acc_q1",
                "add_acc_q2",
                "mm21_acc_q1",
                "mm21_acc_q2"):
            values = []
            for run in runs:
                row = row_at(run["records"][stage], run["best_epoch"])
                values.append(row_harmonic(row) if key == "harmonic" else row[key])
            summary[stage][key] = {
                "mean": mean(values),
                "std": std(values),
                "values": values,
            }
    return summary


def build_repeated_experiment(spec, out_dir, selector_key, sub_exp_ids):
    exp_dir = EXPS_DIR / spec["exp_name"]
    runs = [build_run(exp_dir, out_dir, spec["short"], sub_id, selector_key) for sub_id in sub_exp_ids]
    available_stages = ["train"]
    if all(run["records"]["eval"] for run in runs):
        available_stages.append("eval")
    return {
        "short": spec["short"],
        "label": spec["label"],
        "color": spec["color"],
        "exp_name": spec["exp_name"],
        "setup": load_config_summary(exp_dir / "config.py"),
        "runs": runs,
        "summary": summarize_runs(runs),
        "available_stages": available_stages,
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


def parse_critical_pair_arg(value):
    parts = value.split("|")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--critical-pair must be 'short|experiment_dir_name|sub_exp_id|Display label'"
        )
    short, exp_name, sub_exp_id, label = parts
    return {
        "short": short.strip(),
        "exp_name": exp_name.strip(),
        "sub_exp_id": sub_exp_id.strip(),
        "label": label.strip(),
    }


def parse_pair_risk_arg(value):
    parts = value.split("|")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--pair-risk must be 'short|experiment_dir_name|sub_exp_id|Display label'"
        )
    short, exp_name, sub_exp_id, label = parts
    return {
        "short": short.strip(),
        "exp_name": exp_name.strip(),
        "sub_exp_id": sub_exp_id.strip(),
        "label": label.strip(),
    }


def _parse_critical_pair_row(row):
    int_keys = (
        "epoch",
        "a",
        "b",
        "add_target",
        "mm21_target",
        "total_epochs",
        "q1_exclusive_count",
        "q2_exclusive_count",
        "split_count",
        "tie_or_missing_count",
    )
    float_keys = (
        "q1_exclusive_rate",
        "q2_exclusive_rate",
        "split_rate",
        "tie_or_missing_rate",
    )
    parsed = {}
    for key in int_keys:
        parsed[key] = int(row[key])
    for key in float_keys:
        parsed[key] = float(row[key])
    return parsed


def _parse_pair_risk_row(row):
    int_keys = (
        "epoch",
        "a",
        "b",
        "add_target",
        "mm21_target",
        "total_epochs",
        "q1_only_count",
        "q2_only_count",
        "mixed_count",
        "tie_or_missing_count",
        "q1_exclusive_count",
        "q2_exclusive_count",
        "split_count",
        "mixed_or_missing_count",
    )
    float_keys = (
        "risk_score",
        "q1_only_rate",
        "q2_only_rate",
        "mixed_rate",
        "tie_or_missing_rate",
        "competition_rate",
        "q1_exclusive_rate",
        "q2_exclusive_rate",
        "split_rate",
        "mixed_or_missing_rate",
        "dominance_rate",
    )
    parsed = {
        "pair_type": row["pair_type"],
        "present_ops": row["present_ops"],
        "target_op": row["target_op"],
        "dominant_query": row["dominant_query"],
    }
    for key in int_keys:
        parsed[key] = int(row[key])
    for key in float_keys:
        parsed[key] = float(row[key])
    return parsed


def _dominant_query(q1_rate, q2_rate):
    if q1_rate > q2_rate:
        return "q1"
    if q2_rate > q1_rate:
        return "q2"
    return "tie"


def _critical_pair_item(a, b, add_target, mm21_target, total_epochs, q1_count, q2_count, split_count, tie_or_missing_count):
    q1_rate = 0.0 if total_epochs == 0 else q1_count / total_epochs
    q2_rate = 0.0 if total_epochs == 0 else q2_count / total_epochs
    split_rate = 0.0 if total_epochs == 0 else split_count / total_epochs
    tie_or_missing_rate = 0.0 if total_epochs == 0 else tie_or_missing_count / total_epochs
    return {
        "a": a,
        "b": b,
        "add_target": add_target,
        "mm21_target": mm21_target,
        "total_epochs": total_epochs,
        "q1_exclusive_rate": q1_rate,
        "q2_exclusive_rate": q2_rate,
        "split_rate": split_rate,
        "tie_or_missing_rate": tie_or_missing_rate,
        "q1_exclusive_count": q1_count,
        "q2_exclusive_count": q2_count,
        "split_count": split_count,
        "tie_or_missing_count": tie_or_missing_count,
        "dominant": _dominant_query(q1_rate, q2_rate),
        "dominant_rate": max(q1_rate, q2_rate),
    }


def _single_pair_risk_item(row_or_acc):
    total_epochs = row_or_acc["total_epochs"]
    q1_only_count = row_or_acc["q1_only_count"]
    q2_only_count = row_or_acc["q2_only_count"]
    mixed_count = row_or_acc["mixed_count"]
    tie_or_missing_count = row_or_acc["tie_or_missing_count"]
    q1_rate = 0.0 if total_epochs == 0 else q1_only_count / total_epochs
    q2_rate = 0.0 if total_epochs == 0 else q2_only_count / total_epochs
    mixed_rate = 0.0 if total_epochs == 0 else mixed_count / total_epochs
    tie_rate = 0.0 if total_epochs == 0 else tie_or_missing_count / total_epochs
    competition_rate = mixed_rate + min(q1_rate, q2_rate)
    return {
        "a": row_or_acc["a"],
        "b": row_or_acc["b"],
        "pair_type": row_or_acc["pair_type"],
        "target_op": row_or_acc["target_op"],
        "present_ops": row_or_acc["present_ops"],
        "add_target": row_or_acc["add_target"],
        "mm21_target": row_or_acc["mm21_target"],
        "total_epochs": total_epochs,
        "risk_score": competition_rate,
        "q1_only_rate": q1_rate,
        "q2_only_rate": q2_rate,
        "mixed_rate": mixed_rate,
        "tie_or_missing_rate": tie_rate,
        "q1_only_count": q1_only_count,
        "q2_only_count": q2_only_count,
        "mixed_count": mixed_count,
        "tie_or_missing_count": tie_or_missing_count,
        "dominant_query": _dominant_query(q1_rate, q2_rate),
        "competition_rate": competition_rate,
    }


def _dual_pair_risk_item(row_or_acc):
    total_epochs = row_or_acc["total_epochs"]
    q1_count = row_or_acc["q1_exclusive_count"]
    q2_count = row_or_acc["q2_exclusive_count"]
    split_count = row_or_acc["split_count"]
    mixed_or_missing_count = row_or_acc["mixed_or_missing_count"]
    q1_rate = 0.0 if total_epochs == 0 else q1_count / total_epochs
    q2_rate = 0.0 if total_epochs == 0 else q2_count / total_epochs
    split_rate = 0.0 if total_epochs == 0 else split_count / total_epochs
    mixed_or_missing_rate = 0.0 if total_epochs == 0 else mixed_or_missing_count / total_epochs
    dominance_rate = q1_rate + q2_rate
    return {
        "a": row_or_acc["a"],
        "b": row_or_acc["b"],
        "pair_type": row_or_acc["pair_type"],
        "present_ops": row_or_acc["present_ops"],
        "add_target": row_or_acc["add_target"],
        "mm21_target": row_or_acc["mm21_target"],
        "total_epochs": total_epochs,
        "risk_score": dominance_rate,
        "q1_exclusive_rate": q1_rate,
        "q2_exclusive_rate": q2_rate,
        "split_rate": split_rate,
        "mixed_or_missing_rate": mixed_or_missing_rate,
        "q1_exclusive_count": q1_count,
        "q2_exclusive_count": q2_count,
        "split_count": split_count,
        "mixed_or_missing_count": mixed_or_missing_count,
        "dominant_query": _dominant_query(q1_rate, q2_rate),
        "dominance_rate": dominance_rate,
    }


def build_critical_pair_report(spec, top_n=20):
    csv_path = EXPS_DIR / spec["exp_name"] / str(spec["sub_exp_id"]) / "CriticalPairStats_record.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Critical pair CSV not found: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(_parse_critical_pair_row(row))
    if not rows:
        raise ValueError(f"Critical pair CSV is empty: {csv_path}")

    by_pair = {}
    for row in rows:
        key = (row["a"], row["b"], row["add_target"], row["mm21_target"])
        acc = by_pair.setdefault(
            key,
            {
                "total_epochs": 0,
                "q1_exclusive_count": 0,
                "q2_exclusive_count": 0,
                "split_count": 0,
                "tie_or_missing_count": 0,
            },
        )
        acc["total_epochs"] += row["total_epochs"]
        acc["q1_exclusive_count"] += row["q1_exclusive_count"]
        acc["q2_exclusive_count"] += row["q2_exclusive_count"]
        acc["split_count"] += row["split_count"]
        acc["tie_or_missing_count"] += row["tie_or_missing_count"]

    aggregate_items = [
        _critical_pair_item(
            a,
            b,
            add_target,
            mm21_target,
            acc["total_epochs"],
            acc["q1_exclusive_count"],
            acc["q2_exclusive_count"],
            acc["split_count"],
            acc["tie_or_missing_count"],
        )
        for (a, b, add_target, mm21_target), acc in by_pair.items()
    ]
    aggregate_items.sort(key=lambda item: item["dominant_rate"], reverse=True)

    last_epoch = max(row["epoch"] for row in rows)
    last_items = []
    for row in rows:
        if row["epoch"] != last_epoch:
            continue
        last_items.append(
            _critical_pair_item(
                row["a"],
                row["b"],
                row["add_target"],
                row["mm21_target"],
                row["total_epochs"],
                row["q1_exclusive_count"],
                row["q2_exclusive_count"],
                row["split_count"],
                row["tie_or_missing_count"],
            )
        )
    last_items.sort(key=lambda item: item["dominant_rate"], reverse=True)

    return {
        "short": spec["short"],
        "label": spec["label"],
        "exp_name": spec["exp_name"],
        "sub_exp_id": str(spec["sub_exp_id"]),
        "source": str(Path("exps") / spec["exp_name"] / str(spec["sub_exp_id"]) / "CriticalPairStats_record.csv"),
        "critical_pair_count": len(by_pair),
        "row_count": len(rows),
        "last_epoch": last_epoch,
        "top_all": aggregate_items[:top_n],
        "top_last": last_items[:top_n],
    }


def build_pair_risk_report(spec, top_n=20):
    csv_path = EXPS_DIR / spec["exp_name"] / str(spec["sub_exp_id"]) / "PairRiskStats_record.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Pair risk CSV not found: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(_parse_pair_risk_row(row))
    if not rows:
        raise ValueError(f"Pair risk CSV is empty: {csv_path}")

    by_pair = {}
    count_keys = (
        "q1_only_count",
        "q2_only_count",
        "mixed_count",
        "tie_or_missing_count",
        "q1_exclusive_count",
        "q2_exclusive_count",
        "split_count",
        "mixed_or_missing_count",
    )
    for row in rows:
        key = (row["a"], row["b"], row["pair_type"], row["target_op"])
        acc = by_pair.setdefault(
            key,
            {
                "a": row["a"],
                "b": row["b"],
                "pair_type": row["pair_type"],
                "present_ops": row["present_ops"],
                "target_op": row["target_op"],
                "add_target": row["add_target"],
                "mm21_target": row["mm21_target"],
                "total_epochs": 0,
                **{count_key: 0 for count_key in count_keys},
            },
        )
        acc["total_epochs"] += row["total_epochs"]
        for count_key in count_keys:
            acc[count_key] += row[count_key]

    single_items = [
        _single_pair_risk_item(acc)
        for acc in by_pair.values()
        if acc["pair_type"] in {"single_add", "single_mm21"}
    ]
    dual_items = [
        _dual_pair_risk_item(acc)
        for acc in by_pair.values()
        if acc["pair_type"] == "dual_distinct"
    ]
    single_items.sort(key=lambda item: item["risk_score"], reverse=True)
    dual_items.sort(key=lambda item: item["risk_score"], reverse=True)

    last_epoch = max(row["epoch"] for row in rows)
    last_single_items = [
        _single_pair_risk_item(row)
        for row in rows
        if row["epoch"] == last_epoch and row["pair_type"] in {"single_add", "single_mm21"}
    ]
    last_dual_items = [
        _dual_pair_risk_item(row)
        for row in rows
        if row["epoch"] == last_epoch and row["pair_type"] == "dual_distinct"
    ]
    last_single_items.sort(key=lambda item: item["risk_score"], reverse=True)
    last_dual_items.sort(key=lambda item: item["risk_score"], reverse=True)

    return {
        "short": spec["short"],
        "label": spec["label"],
        "exp_name": spec["exp_name"],
        "sub_exp_id": str(spec["sub_exp_id"]),
        "source": str(Path("exps") / spec["exp_name"] / str(spec["sub_exp_id"]) / "PairRiskStats_record.csv"),
        "pair_count": len(by_pair),
        "single_pair_count": len(single_items),
        "dual_pair_count": len(dual_items),
        "row_count": len(rows),
        "last_epoch": last_epoch,
        "top_single_all": single_items[:top_n],
        "top_single_last": last_single_items[:top_n],
        "top_dual_all": dual_items[:top_n],
        "top_dual_last": last_dual_items[:top_n],
    }


def render_markdownish(text):
    lines = text.strip().splitlines()
    if not lines:
        return "<p>No experiment analysis was provided for this report.</p>"

    parts = []
    paragraph = []
    in_list = False

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            parts.append(f"<p>{html.escape(' '.join(paragraph))}</p>")
            paragraph = []

    def close_list():
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            close_list()
            continue
        if line.startswith("## "):
            flush_paragraph()
            close_list()
            parts.append(f"<h3>{html.escape(line[3:].strip())}</h3>")
        elif line.startswith("# "):
            flush_paragraph()
            close_list()
            parts.append(f"<h3>{html.escape(line[2:].strip())}</h3>")
        elif line.startswith("- "):
            flush_paragraph()
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{html.escape(line[2:].strip())}</li>")
        else:
            paragraph.append(line)

    flush_paragraph()
    close_list()
    return "\n".join(parts)


def build_default_analysis(experiments, selector_key):
    report_stage = "eval" if all(exp["records"]["eval"] for exp in experiments) else "train"
    rows = []
    for exp in experiments:
        row = row_at(exp["records"][report_stage], exp["best_epoch"])
        rows.append((exp, row, harmonic(row["add_acc"], row["mm21_acc"])))
    rows.sort(key=lambda item: item[2], reverse=True)
    leader, leader_row, leader_h = rows[0]
    selection_note = (
        f"Best epoch is selected by lowest train {selector_key} among epochs visible in Eval_record and query operation images."
        if report_stage == "eval"
        else f"No Eval_record is present, so best epoch is selected by lowest train {selector_key} among epochs with query operation images."
    )
    return (
        "## Key Takeaways\n"
        f"- {selection_note}\n"
        f"- On {report_stage}, {leader['short']} has the highest harmonic mean: H={leader_h:.3f}, "
        f"add={leader_row['add_acc']:.3f}, mm21={leader_row['mm21_acc']:.3f}.\n"
        "- Add a custom --analysis-file when the report needs experiment-specific causal hypotheses and next-step recommendations.\n"
    )


def render_index(title, report_name, experiments, selector_key, analysis_text=""):
    payload = json.dumps(
        {
            "title": title,
            "reportName": report_name,
            "selectorKey": selector_key,
            "experiments": experiments,
            "availableStages": ["train", "eval"] if all(exp["records"]["eval"] for exp in experiments) else ["train"],
        },
        ensure_ascii=False,
    )
    escaped_title = html.escape(title)
    analysis_source = analysis_text.strip() or build_default_analysis(experiments, selector_key)
    analysis_html = render_markdownish(analysis_source)
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
    main {{ max-width: 1220px; margin: 0 auto; padding: 84px 28px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 34px 0 12px; font-size: 21px; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; }}
    p, li {{ line-height: 1.55; }}
    code {{ background: #edf2f7; padding: 2px 5px; border-radius: 5px; font-size: 0.92em; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 18px; }}
    .card, .chart-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .chart-panel {{ background: #fff; margin-top: 14px; }}
    .analysis-panel {{ background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px; margin-top: 16px; }}
    .analysis-panel h3 {{ margin: 18px 0 8px; font-size: 17px; }}
    .analysis-panel h3:first-child {{ margin-top: 0; }}
    .analysis-panel ul {{ margin: 8px 0 0; padding-left: 22px; }}
    .analysis-panel li {{ margin: 5px 0; }}
    .metric {{ font-size: 30px; font-weight: 700; margin: 8px 0 2px; }}
    .metric small {{ font-size: 14px; color: var(--muted); font-weight: 400; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #334155; background: #f8fafc; }}
    canvas {{ width: 100%; height: 340px; display: block; }}
    .stage-switch {{ position: fixed; top: 16px; right: 18px; z-index: 900; display: flex; align-items: center; gap: 8px; padding: 8px; border: 1px solid var(--line); border-radius: 8px; background: rgba(255, 255, 255, 0.94); box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12); backdrop-filter: blur(8px); }}
    .stage-label {{ color: var(--muted); font-size: 13px; padding: 0 4px; }}
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
    .query-wise-list {{ display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 12px; }}
    .query-wise-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; }}
    .query-wise-card h3 {{ margin-bottom: 4px; }}
    .query-wise-card canvas {{ height: 280px; margin-top: 8px; }}
    .lightbox {{ position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(15, 23, 42, 0.86); z-index: 1000; padding: 28px; }}
    .lightbox.open {{ display: flex; }}
    .lightbox-inner {{ max-width: min(96vw, 1500px); max-height: 94vh; width: 100%; }}
    .lightbox img {{ max-width: 100%; max-height: 86vh; display: block; margin: 0 auto; background: #fff; border-radius: 8px; }}
    .lightbox-caption {{ color: #e2e8f0; text-align: center; margin-top: 10px; font-size: 14px; }}
    .lightbox-close {{ position: fixed; top: 18px; right: 22px; border: 1px solid rgba(255,255,255,0.5); color: #fff; background: rgba(15,23,42,0.4); border-radius: 7px; padding: 7px 11px; cursor: pointer; font-size: 14px; }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-top: 5px; text-align: center; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 3px 8px; background: #e2e8f0; color: #334155; font-size: 12px; margin-top: 3px; }}
    @media (max-width: 900px) {{ .grid, .query-pair {{ grid-template-columns: 1fr; }} main {{ padding: 86px 16px 44px; }} .stage-switch {{ left: 16px; right: 16px; justify-content: flex-end; }} }}
  </style>
</head>
<body>
<div id="stageSwitch" class="stage-switch" aria-label="Train and eval view switch"></div>
<main>
  <h1>{escaped_title}</h1>
  <section class="analysis-panel">
    {analysis_html}
  </section>

  <h2>实验设置</h2>
  <table id="setup-table"></table>

  <h2>Metric Curves</h2>
  <section class="chart-panel">
    <h3>add_acc / mm21_acc over saved epochs</h3>
    <canvas id="lineChart" width="1120" height="340"></canvas>
    <div class="legend" id="lineLegend"></div>
  </section>

  <h2>Best Epoch Bar Chart</h2>
  <section class="chart-panel">
    <h3>best checkpoint metrics selected by lowest train {html.escape(selector_key)}</h3>
    <canvas id="barChart" width="1120" height="340"></canvas>
    <div class="legend" id="barLegend"></div>
  </section>

  <h2>Query-wise Best Epoch Bar Chart</h2>
  <section class="chart-panel">
    <p class="muted">每行一个实验；每个实验含 add_acc 和 mm21_acc 两组柱，每组分别展示 q1、q2 和 overall。</p>
    <div class="legend" id="queryWiseLegend"></div>
    <div id="queryWiseBars" class="query-wise-list"></div>
  </section>

  <h2>Data-Pair Visualization</h2>
  <section class="chart-panel">
    <p class="muted">展示每个实验 best epoch 的 add / mm21 data-pair table。图片来自本报告目录下的 assets。</p>
    <div id="queryImages" class="query-list"></div>
  </section>

  <h2>Automated Findings</h2>
  <section class="chart-panel">
    <div id="findings"></div>
  </section>

  <h2>Best Epoch Summary</h2>
  <section class="chart-panel">
    <table id="summary-table"></table>
  </section>

  <h2>Raw Saved-Epoch Data</h2>
  <section class="chart-panel">
    <table id="raw-table"></table>
  </section>

  <h2>Report Notes</h2>
  <div class="note">
    <strong>Checkpoint rule</strong>
    <p>{"Best epoch is selected from epochs visible in Eval_record and query operation images by the lowest corresponding Train_record" if all(exp["records"]["eval"] for exp in experiments) else "No Eval_record is present, so best epoch is selected by the lowest Train_record among epochs with query operation images"} <code>{html.escape(selector_key)}</code>. All best-epoch tables, bars, and query images use that same epoch.</p>
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
const availableStages = REPORT.availableStages || ["train", "eval"];
let activeStage = availableStages.includes("eval") ? "eval" : "train";

function harmonic(addAcc, mm21Acc) {{
  const denom = addAcc + mm21Acc;
  return denom === 0 ? 0 : 2 * addAcc * mm21Acc / denom;
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
function drawAxes(ctx, plot, xTicks, yTicks, xMax = 50000) {{
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
    const x = plot.x + (t / xMax) * plot.w;
    ctx.fillText(String(t / 1000) + "k", x - 10, plot.y + plot.h + 22);
  }});
}}
function renderStageSwitch() {{
  const el = document.getElementById("stageSwitch");
  el.innerHTML = `
      <span class="stage-label">View</span>
      ${{availableStages.map(stage => `<button class="tab-btn ${{activeStage === stage ? "active" : ""}}" data-stage="${{stage}}">${{stage === "train" ? "Train" : "Eval"}}</button>`).join("")}}
    `;
  el.querySelectorAll(".tab-btn").forEach(btn => {{
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
    <span class="legend-item"><span class="line-sample line-dashed"></span>mm21_acc</span>
  `;
  document.getElementById("barLegend").innerHTML = `<span class="legend-title">颜色=实验</span>${{expLegend}}`;
  document.getElementById("queryWiseLegend").innerHTML = `
    <span class="legend-title">颜色=query scope</span>
    <span class="legend-item"><span class="swatch" style="background:#2563eb"></span>q1_acc</span>
    <span class="legend-item"><span class="swatch" style="background:#dc2626"></span>q2_acc</span>
    <span class="legend-item"><span class="swatch" style="background:#111827"></span>overall</span>
  `;
}}
function renderSetup() {{
  const table = document.getElementById("setup-table");
  table.innerHTML = `<thead><tr><th>experiment</th><th>SPS init</th><th>operator</th><th>query</th><th>training</th><th>best epoch</th></tr></thead>` +
    `<tbody>` + experiments.map(exp => {{
      const trainRow = rowAt(exp, "train", exp.best_epoch);
      return `<tr>
        <td><strong style="color:${{exp.color}}">${{exp.short}}</strong><br><span class="muted">${{exp.label}}</span></td>
        <td>${{exp.setup.sps}}</td>
        <td>${{exp.setup.operator}}</td>
        <td>${{exp.setup.query}}</td>
        <td>${{exp.setup.training}}</td>
        <td><span class="pill">epoch ${{exp.best_epoch}}</span><br><span class="muted">train ${{REPORT.selectorKey}}=${{fmt(trainRow[REPORT.selectorKey])}}</span></td>
      </tr>`;
    }}).join("") + `</tbody>`;
}}
function drawLineChart() {{
  const {{ctx, w, h}} = setupCanvas(document.getElementById("lineChart"));
  ctx.clearRect(0, 0, w, h);
  const plot = {{x: 56, y: 18, w: w - 230, h: h - 62}};
  const maxEpoch = Math.max(1, ...experiments.flatMap(exp => exp.records[activeStage].map(row => row.epoch)));
  const tickStep = maxEpoch <= 20000 ? 5000 : 10000;
  const xTicks = [];
  for (let t = 0; t <= maxEpoch; t += tickStep) xTicks.push(t);
  if (xTicks[xTicks.length - 1] !== maxEpoch) xTicks.push(maxEpoch);
  drawAxes(ctx, plot, xTicks, [0, 0.25, 0.5, 0.75, 1.0], maxEpoch);
  const labels = [];
  experiments.forEach(exp => {{
    [["add_acc", "add"], ["mm21_acc", "mm21"]].forEach(([key, label]) => {{
      ctx.save();
      ctx.strokeStyle = exp.color;
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 2.5;
      ctx.setLineDash(key === "mm21_acc" ? [7, 5] : []);
      ctx.beginPath();
      exp.records[activeStage].forEach((row, i) => {{
        const x = plot.x + (row.epoch / maxEpoch) * plot.w;
        const y = plot.y + plot.h - row[key] * plot.h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.restore();
      const last = exp.records[activeStage][exp.records[activeStage].length - 1];
      labels.push({{
        y: plot.y + plot.h - last[key] * plot.h,
        color: exp.color,
        dash: key === "mm21_acc" ? [7, 5] : [],
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
    {{key: "mm21_acc", label: "mm21_acc"}},
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
      const value = group.key === "harmonic" ? harmonic(row.add_acc, row.mm21_acc) : row[group.key];
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
function drawSingleQueryWiseChart(canvas, exp) {{
  const {{ctx, w, h}} = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const row = rowAt(exp, activeStage, exp.best_epoch);
  const plot = {{x: 58, y: 24, w: w - 96, h: h - 74}};
  drawAxes(ctx, plot, [], [0, 0.25, 0.5, 0.75, 1.0]);
  const groups = [
    {{
      label: "add_acc",
      values: [
        {{label: "q1", value: row.add_acc_q1, color: "#2563eb"}},
        {{label: "q2", value: row.add_acc_q2, color: "#dc2626"}},
        {{label: "overall", value: row.add_acc, color: "#111827"}}
      ]
    }},
    {{
      label: "mm21_acc",
      values: [
        {{label: "q1", value: row.mm21_acc_q1, color: "#2563eb"}},
        {{label: "q2", value: row.mm21_acc_q2, color: "#dc2626"}},
        {{label: "overall", value: row.mm21_acc, color: "#111827"}}
      ]
    }}
  ];
  const groupW = plot.w / groups.length;
  groups.forEach((group, groupIdx) => {{
    const barW = Math.min(62, groupW / 8);
    const gap = 16;
    const totalW = group.values.length * barW + (group.values.length - 1) * gap;
    const start = plot.x + groupIdx * groupW + groupW / 2 - totalW / 2;
    group.values.forEach((item, itemIdx) => {{
      const x = start + itemIdx * (barW + gap);
      const bh = item.value * plot.h;
      ctx.fillStyle = item.color;
      ctx.fillRect(x, plot.y + plot.h - bh, barW, bh);
      ctx.fillStyle = "#172033";
      ctx.font = "12px Arial";
      ctx.fillText(fmt(item.value), x - 2, plot.y + plot.h - bh - 7);
      ctx.fillStyle = "#607086";
      ctx.fillText(item.label, x + Math.max(0, barW / 2 - 18), plot.y + plot.h + 20);
    }});
    ctx.fillStyle = "#334155";
    ctx.font = "13px Arial";
    ctx.fillText(group.label, plot.x + groupIdx * groupW + groupW / 2 - 26, plot.y + plot.h + 44);
  }});
}}
function renderQueryWiseBars() {{
  const root = document.getElementById("queryWiseBars");
  root.innerHTML = experiments.map(exp => {{
    const row = rowAt(exp, activeStage, exp.best_epoch);
    return `<article class="query-wise-card">
      <h3 style="color:${{exp.color}}">${{exp.short}}</h3>
      <div class="muted">best epoch ${{exp.best_epoch}}; ${{activeStage}} q1(add=${{fmt(row.add_acc_q1)}}, mm21=${{fmt(row.mm21_acc_q1)}}), q2(add=${{fmt(row.add_acc_q2)}}, mm21=${{fmt(row.mm21_acc_q2)}}), overall(add=${{fmt(row.add_acc)}}, mm21=${{fmt(row.mm21_acc)}})</div>
      <canvas class="query-wise-canvas" data-exp-short="${{exp.short}}" width="1120" height="280"></canvas>
    </article>`;
  }}).join("");
  document.querySelectorAll(".query-wise-canvas").forEach(canvas => {{
    const exp = experiments.find(item => item.short === canvas.dataset.expShort);
    drawSingleQueryWiseChart(canvas, exp);
  }});
}}
function renderQueryImages() {{
  const root = document.getElementById("queryImages");
  root.innerHTML = experiments.map(exp => {{
    const row = rowAt(exp, activeStage, exp.best_epoch);
    const imgs = exp.images[activeStage];
    return `<article class="query-card">
      <h3 style="color:${{exp.color}}">${{exp.short}}</h3>
      <div class="muted">best epoch ${{exp.best_epoch}}; ${{activeStage}} add=${{fmt(row.add_acc)}}, mm21=${{fmt(row.mm21_acc)}}, H=${{fmt(harmonic(row.add_acc, row.mm21_acc))}}</div>
      <div class="query-pair">
        <figure>
          <button class="query-img-btn" type="button" data-full-src="${{imgs[0]}}" data-caption="${{exp.short}} ${{activeStage}} add set, epoch ${{exp.best_epoch}}">
            <img src="${{imgs[0]}}" alt="${{exp.short}} ${{activeStage}} add set">
          </button>
          <figcaption>add set - click to enlarge</figcaption>
        </figure>
        <figure>
          <button class="query-img-btn" type="button" data-full-src="${{imgs[1]}}" data-caption="${{exp.short}} ${{activeStage}} mm21 set, epoch ${{exp.best_epoch}}">
            <img src="${{imgs[1]}}" alt="${{exp.short}} ${{activeStage}} mm21 set">
          </button>
          <figcaption>mm21 set - click to enlarge</figcaption>
        </figure>
      </div>
    </article>`;
  }}).join("");
  bindLightboxButtons();
}}
function queryPreference(addValue, mm21Value) {{
  if (Math.abs(addValue - mm21Value) < 0.02) return "balanced";
  return addValue > mm21Value ? "add-biased" : "mm21-biased";
}}
function renderFindings() {{
  const root = document.getElementById("findings");
  const rows = experiments.map(exp => {{
    const trainRow = rowAt(exp, "train", exp.best_epoch);
    const row = rowAt(exp, activeStage, exp.best_epoch);
    return {{
      exp,
      trainRow,
      row,
      h: harmonic(row.add_acc, row.mm21_acc),
      q1Pref: queryPreference(row.add_acc_q1, row.mm21_acc_q1),
      q2Pref: queryPreference(row.add_acc_q2, row.mm21_acc_q2)
    }};
  }}).sort((a, b) => b.h - a.h);
  const leader = rows[0];
  const comparison = rows.length === 2
    ? `<p class="muted">相对差异：${{rows[0].exp.short}} 比 ${{rows[1].exp.short}} 高 ${{fmt(rows[0].h - rows[1].h)}} harmonic；add_acc 差 ${{fmt(rows[0].row.add_acc - rows[1].row.add_acc)}}，mm21_acc 差 ${{fmt(rows[0].row.mm21_acc - rows[1].row.mm21_acc)}}。</p>`
    : "";
  root.innerHTML = `
    <p><strong style="color:${{leader.exp.color}}">${{leader.exp.short}}</strong> 在当前 ${{activeStage}} 页签下 best-epoch harmonic 最高：H=${{fmt(leader.h)}}，add=${{fmt(leader.row.add_acc)}}，mm21=${{fmt(leader.row.mm21_acc)}}。</p>
    ${{comparison}}
    <table>
      <thead><tr><th>experiment</th><th>best epoch</th><th>${{activeStage}} overall</th><th>q1 behavior</th><th>q2 behavior</th></tr></thead>
      <tbody>${{rows.map(item => `<tr>
        <td><strong style="color:${{item.exp.color}}">${{item.exp.short}}</strong></td>
        <td>${{item.exp.best_epoch}}<br><span class="muted">train ${{REPORT.selectorKey}}=${{fmt(item.trainRow[REPORT.selectorKey])}}</span></td>
        <td>add=${{fmt(item.row.add_acc)}}<br>mm21=${{fmt(item.row.mm21_acc)}}<br>H=${{fmt(item.h)}}</td>
        <td>${{item.q1Pref}}<br><span class="muted">add=${{fmt(item.row.add_acc_q1)}}, mm21=${{fmt(item.row.mm21_acc_q1)}}</span></td>
        <td>${{item.q2Pref}}<br><span class="muted">add=${{fmt(item.row.add_acc_q2)}}, mm21=${{fmt(item.row.mm21_acc_q2)}}</span></td>
      </tr>`).join("")}}</tbody>
    </table>
  `;
}}
function renderSummary() {{
  const table = document.getElementById("summary-table");
  table.innerHTML = `<thead><tr><th>experiment</th><th>best epoch</th><th>train selector</th><th>${{activeStage}} add_acc</th><th>${{activeStage}} mm21_acc</th><th>${{activeStage}} harmonic</th></tr></thead>` +
    `<tbody>` + experiments.map(exp => {{
      const trainRow = rowAt(exp, "train", exp.best_epoch);
      const row = rowAt(exp, activeStage, exp.best_epoch);
      return `<tr>
        <td><strong style="color:${{exp.color}}">${{exp.short}}</strong></td>
        <td>${{exp.best_epoch}}</td>
        <td>train ${{REPORT.selectorKey}}=${{fmt(trainRow[REPORT.selectorKey])}}</td>
        <td>${{fmt(row.add_acc)}}</td>
        <td>${{fmt(row.mm21_acc)}}</td>
        <td>${{fmt(harmonic(row.add_acc, row.mm21_acc))}}</td>
      </tr>`;
    }}).join("") + `</tbody>`;
}}
function renderRaw() {{
  const table = document.getElementById("raw-table");
  const rows = experiments.flatMap(exp => exp.records[activeStage].map(row => ({{exp, row}})));
  table.innerHTML = `<thead><tr><th>experiment</th><th>epoch</th><th>total_loss</th><th>add_acc</th><th>mm21_acc</th><th>harmonic</th><th>best?</th></tr></thead>` +
    `<tbody>` + rows.map(({{exp, row}}) => {{
      const isBest = row.epoch === exp.best_epoch;
      return `<tr>
        <td><strong style="color:${{exp.color}}">${{exp.short}}</strong></td>
        <td>${{row.epoch}}</td>
        <td>${{fmt(row.total_loss)}}</td>
        <td>${{fmt(row.add_acc)}}</td>
        <td>${{fmt(row.mm21_acc)}}</td>
        <td>${{fmt(harmonic(row.add_acc, row.mm21_acc))}}</td>
        <td>${{isBest ? '<span class="pill">selected</span>' : ''}}</td>
      </tr>`;
    }}).join("") + `</tbody>`;
}}
function renderAll() {{
  renderStageSwitch();
  renderLegends();
  renderSetup();
  drawLineChart();
  drawBarChart();
  renderQueryWiseBars();
  renderQueryImages();
  renderFindings();
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
  document.querySelectorAll(".query-wise-canvas").forEach(canvas => {{
    const exp = experiments.find(item => item.short === canvas.dataset.expShort);
    drawSingleQueryWiseChart(canvas, exp);
  }});
}});
renderAll();
</script>
</body>
</html>
"""


def render_repeated_index(
        title,
        report_name,
        experiments,
        selector_key,
        analysis_text="",
        critical_pair_reports=None,
        pair_risk_reports=None):
    critical_pair_reports = critical_pair_reports or []
    pair_risk_reports = pair_risk_reports or []
    payload = json.dumps(
        {
            "title": title,
            "reportName": report_name,
            "selectorKey": selector_key,
            "experiments": experiments,
            "criticalPairReports": critical_pair_reports,
            "pairRiskReports": pair_risk_reports,
            "availableStages": ["train", "eval"] if all("eval" in exp.get("available_stages", []) for exp in experiments) else ["train"],
        },
        ensure_ascii=False,
    )
    escaped_title = html.escape(title)
    analysis_source = analysis_text.strip() or build_default_repeated_analysis(experiments, selector_key)
    analysis_html = render_markdownish(analysis_source)
    checkpoint_note = (
        f"Each sub-exp selects the epoch visible in Eval_record and query operation images with the lowest corresponding Train_record <code>{html.escape(selector_key)}</code>. Aggregate means/stds are computed over the selected sub-exp checkpoints."
        if all("eval" in exp.get("available_stages", []) for exp in experiments)
        else f"No Eval_record is present for this report, so each sub-exp selects the epoch with the lowest Train_record <code>{html.escape(selector_key)}</code> among epochs with query operation images. Aggregate means/stds are computed over the selected train checkpoints."
    )
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root { --ink:#172033; --muted:#607086; --line:#d7deea; --panel:#f8fafc; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Arial, Helvetica, sans-serif; color:var(--ink); background:#fff; }
    main { max-width:1240px; margin:0 auto; padding:84px 28px 56px; }
    h1 { margin:0 0 8px; font-size:28px; }
    h2 { margin:34px 0 12px; font-size:21px; }
    h3 { margin:0 0 10px; font-size:16px; }
    p, li { line-height:1.55; }
    code { background:#edf2f7; padding:2px 5px; border-radius:5px; font-size:.92em; }
    table { width:100%; border-collapse:collapse; margin-top:12px; font-size:14px; }
    th, td { border-bottom:1px solid var(--line); padding:9px 8px; text-align:left; vertical-align:top; }
    th { color:#334155; background:#f8fafc; }
    canvas { width:100%; height:340px; display:block; }
    .muted { color:var(--muted); }
    .analysis-panel, .chart-panel { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px; margin-top:14px; }
    .analysis-panel { padding:18px 20px; }
    .analysis-panel h3 { margin:18px 0 8px; font-size:17px; }
    .analysis-panel h3:first-child { margin-top:0; }
    .analysis-panel ul { margin:8px 0 0; padding-left:22px; }
    .stage-switch { position:fixed; top:16px; right:18px; z-index:900; display:flex; align-items:center; gap:12px; padding:8px; border:1px solid var(--line); border-radius:8px; background:rgba(255,255,255,.94); box-shadow:0 8px 24px rgba(15,23,42,.12); backdrop-filter:blur(8px); }
    .switch-group { display:flex; align-items:center; gap:7px; }
    .stage-label { color:var(--muted); font-size:13px; padding:0 4px; }
    .tab-btn { border:1px solid var(--line); background:#f8fafc; color:#334155; border-radius:7px; padding:7px 12px; font-size:14px; cursor:pointer; }
    .tab-btn.active { background:#172033; color:#fff; border-color:#172033; }
    .legend { display:flex; gap:14px; flex-wrap:wrap; margin:10px 0 0; color:var(--muted); font-size:13px; }
    .legend-title { color:#334155; font-weight:700; }
    .legend-item { display:inline-flex; align-items:center; gap:6px; min-height:18px; }
    .swatch { display:inline-block; width:12px; height:12px; border-radius:3px; }
    .line-sample { display:inline-block; width:34px; border-top:3px solid #334155; transform:translateY(-1px); }
    .line-dashed { border-top-style:dashed; }
    .pill { display:inline-block; border-radius:999px; padding:3px 8px; background:#e2e8f0; color:#334155; font-size:12px; margin-top:3px; }
    .summary-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; margin-top:12px; }
    .summary-card, .run-card { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; }
    .metric-row { display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; margin-top:10px; }
    .metric { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
    .metric strong { display:block; font-size:22px; margin-top:5px; }
    .critical-report { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; margin-top:12px; }
    details.critical-report > summary { cursor:pointer; font-weight:700; color:#334155; }
    details.critical-report[open] > summary { margin-bottom:8px; }
    .critical-grid { display:grid; grid-template-columns:1fr; gap:16px; margin-top:10px; }
    .dominant-q1 { color:#2563eb; font-weight:700; }
    .dominant-q2 { color:#dc2626; font-weight:700; }
    .dominant-tie { color:#64748b; font-weight:700; }
    .run-list { display:grid; grid-template-columns:1fr; gap:16px; margin-top:12px; }
    .subrun-grid { display:grid; grid-template-columns:1fr; gap:12px; margin-top:10px; }
    .image-row { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; }
    .image-row-head { display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
    .image-pair { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:8px; }
    figure { margin:0; }
    figcaption { color:var(--muted); font-size:12px; margin-top:5px; text-align:center; }
    .img-btn { display:block; width:100%; padding:0; border:0; background:transparent; cursor:zoom-in; }
    .img-btn img { width:100%; border:1px solid var(--line); border-radius:6px; background:#fff; display:block; }
    .lightbox { position:fixed; inset:0; display:none; align-items:center; justify-content:center; background:rgba(15,23,42,.86); z-index:1000; padding:28px; }
    .lightbox.open { display:flex; }
    .lightbox-inner { max-width:min(96vw,1500px); max-height:94vh; width:100%; }
    .lightbox img { max-width:100%; max-height:86vh; display:block; margin:0 auto; background:#fff; border-radius:8px; }
    .lightbox-caption { color:#e2e8f0; text-align:center; margin-top:10px; font-size:14px; }
    .lightbox-close { position:fixed; top:18px; right:22px; border:1px solid rgba(255,255,255,.5); color:#fff; background:rgba(15,23,42,.4); border-radius:7px; padding:7px 11px; cursor:pointer; font-size:14px; }
    @media (max-width:900px) { main { padding:128px 16px 44px; } .summary-grid, .metric-row, .image-pair { grid-template-columns:1fr; } .stage-switch { left:16px; right:16px; justify-content:flex-end; flex-wrap:wrap; } }
  </style>
</head>
<body>
<div id="stageSwitch" class="stage-switch" aria-label="Train and eval view switch"></div>
<main>
  <h1>__TITLE__</h1>
  <section class="analysis-panel">__ANALYSIS__</section>

  <h2 id="setupTitle">实验设置</h2>
  <section id="setupSection" class="chart-panel"><table id="setupTable"></table></section>

  <h2 id="aggregateTitle">Aggregate Summary</h2>
  <section id="aggregateSection" class="chart-panel">
    <div id="aggregateCards" class="summary-grid"></div>
    <canvas id="aggregateBars" width="1120" height="340"></canvas>
    <div class="legend" id="aggregateLegend"></div>
  </section>

  <h2 id="metricTitle">Metric Curves</h2>
  <section id="metricSection" class="chart-panel">
    <canvas id="lineChart" width="1120" height="360"></canvas>
    <div class="legend" id="lineLegend"></div>
  </section>

  <h2 id="queryWiseTitle">Query-wise Best Epoch Accuracy</h2>
  <section id="queryWiseSection" class="chart-panel">
    <p class="muted">Overall 视图按实验展示 mean +/- sample std；具体实验视图按 sub-exp 展示 q1/q2/overall。</p>
    <div id="queryWiseTable"></div>
    <div class="legend" id="queryWiseLegend"></div>
  </section>

  <h2 id="subexpTitle">Sub-exp Details</h2>
  <section id="subexpSection" class="chart-panel"><table id="subexpTable"></table></section>

  <h2 id="criticalPairTitle">Critical Pair Exclusive-Dominance</h2>
  <section id="criticalPairSection" class="chart-panel">
    <p class="muted">只分析训练集上同时存在 add target 和 mm21 target 的关键 (a,b) pair。dominant rate 表示同一个 query 同时赢走 add/mm21 两个 target 的比例。</p>
    <div id="criticalPairReports"></div>
  </section>

  <h2 id="pairRiskTitle">Pair Risk Diagnostics</h2>
  <section id="pairRiskSection" class="chart-panel">
    <p class="muted">同时分析 single-op pair 的 query 竞争风险，以及 dual-op pair 的同 query 独占风险。special ambiguous pair 不进入风险表。</p>
    <div id="pairRiskReports"></div>
  </section>

  <h2 id="dataPairTitle">Data-Pair Visualization</h2>
  <section id="dataPairSection" class="chart-panel">
    <p class="muted">每个 sub-exp 独占一行；行内并排展示 add 和 mm21 两张 query operation 图，每张图可点击放大。</p>
    <div id="imageRuns" class="run-list"></div>
  </section>

  <h2>Report Notes</h2>
  <div class="chart-panel">
    <strong>Checkpoint rule</strong>
    <p>__CHECKPOINT_NOTE__</p>
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
const REPORT = __PAYLOAD__;
const experiments = REPORT.experiments;
const availableStages = REPORT.availableStages || ["train", "eval"];
let activeStage = availableStages.includes("eval") ? "eval" : "train";
let activeSubView = "overall";

function fmt(x) { return Number(x).toFixed(3); }
function harmonic(addAcc, mm21Acc) {
  const denom = addAcc + mm21Acc;
  return denom === 0 ? 0 : 2 * addAcc * mm21Acc / denom;
}
function rowAt(run, stage, epoch) { return run.records[stage].find(row => row.epoch === epoch); }
function bestRow(run, stage) { return rowAt(run, stage, run.best_epoch); }
function subExpIds() {
  return Array.from(new Set(experiments.flatMap(exp => exp.runs.map(run => run.sub_id)))).sort((a, b) => Number(a) - Number(b));
}
function visibleRuns(exp) {
  return activeSubView === "overall" ? exp.runs : exp.runs.filter(run => run.sub_id === activeSubView);
}
function valuesFor(exp, key, stage = activeStage) {
  return visibleRuns(exp).map(run => {
    const row = bestRow(run, stage);
    return key === "harmonic" ? harmonic(row.add_acc, row.mm21_acc) : row[key];
  });
}
function setSectionVisible(titleId, sectionId, visible) {
  document.getElementById(titleId).style.display = visible ? "" : "none";
  document.getElementById(sectionId).style.display = visible ? "" : "none";
}
function avg(values) { return values.reduce((a, b) => a + b, 0) / values.length; }
function sampleStd(values) {
  if (values.length <= 1) return 0;
  const m = avg(values);
  return Math.sqrt(values.reduce((acc, value) => acc + Math.pow(value - m, 2), 0) / (values.length - 1));
}
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return {ctx, w: rect.width, h: rect.height};
}
function drawAxes(ctx, plot, xTicks, yTicks, xMax = 1) {
  ctx.strokeStyle = "#d7deea";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y);
  ctx.lineTo(plot.x, plot.y + plot.h);
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
  ctx.stroke();
  ctx.fillStyle = "#607086";
  ctx.font = "12px Arial";
  yTicks.forEach(t => {
    const y = plot.y + plot.h - t * plot.h;
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
    ctx.fillText(t.toFixed(1), plot.x - 34, y + 4);
  });
  xTicks.forEach(t => {
    const x = plot.x + (t / xMax) * plot.w;
    ctx.fillText(String(t / 1000) + "k", x - 10, plot.y + plot.h + 22);
  });
}
function drawScaledAxes(ctx, plot, xTicks, yTicks, yMin, yMax, xMax = 1) {
  ctx.strokeStyle = "#d7deea";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y);
  ctx.lineTo(plot.x, plot.y + plot.h);
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h);
  ctx.stroke();
  ctx.fillStyle = "#607086";
  ctx.font = "12px Arial";
  yTicks.forEach(value => {
    const frac = (value - yMin) / (yMax - yMin);
    const y = plot.y + plot.h - frac * plot.h;
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(2), plot.x - 42, y + 4);
  });
  xTicks.forEach(t => {
    const x = plot.x + (t / xMax) * plot.w;
    ctx.fillText(String(t / 1000) + "k", x - 10, plot.y + plot.h + 22);
  });
}
function niceMetricRange(rows) {
  const values = rows
    .filter(row => row.epoch > 0)
    .flatMap(row => [row.add_acc, row.mm21_acc])
    .filter(value => Number.isFinite(value));
  if (!values.length) return {min: 0, max: 1};
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const pad = Math.max(0.02, (hi - lo) * 0.12);
  lo = Math.max(0, Math.floor((lo - pad) * 20) / 20);
  hi = Math.min(1, Math.ceil((hi + pad) * 20) / 20);
  if (hi - lo < 0.1) {
    lo = Math.max(0, hi - 0.1);
  }
  return {min: lo, max: hi};
}
function yTicksForRange(min, max) {
  const ticks = [];
  for (let i = 0; i <= 4; i++) ticks.push(min + (max - min) * i / 4);
  return ticks;
}
function renderStageSwitch() {
  const el = document.getElementById("stageSwitch");
  el.innerHTML = `
    <div class="switch-group">
      <span class="stage-label">Stage</span>
      ${availableStages.map(stage => `<button class="tab-btn ${activeStage === stage ? "active" : ""}" data-stage="${stage}">${stage === "train" ? "Train" : "Eval"}</button>`).join("")}
    </div>
    <div class="switch-group">
      <span class="stage-label">Sub-exp</span>
      <button class="tab-btn ${activeSubView === "overall" ? "active" : ""}" data-sub-view="overall">Overall</button>
      ${subExpIds().map(id => `<button class="tab-btn ${activeSubView === id ? "active" : ""}" data-sub-view="${id}">exp${id}</button>`).join("")}
    </div>`;
  el.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (btn.dataset.stage) activeStage = btn.dataset.stage;
      if (btn.dataset.subView) activeSubView = btn.dataset.subView;
      renderAll();
    });
  });
}
function renderSetup() {
  document.getElementById("setupTable").innerHTML =
    `<thead><tr><th>experiment</th><th>SPS init</th><th>operator</th><th>query</th><th>training</th><th>runs</th></tr></thead><tbody>` +
    experiments.map(exp => `<tr>
      <td><strong style="color:${exp.color}">${exp.short}</strong><br><span class="muted">${exp.label}</span></td>
      <td>${exp.setup.sps}</td><td>${exp.setup.operator}</td><td>${exp.setup.query}</td><td>${exp.setup.training}</td>
      <td>${visibleRuns(exp).map(run => "sub" + run.sub_id).join(", ")}</td>
    </tr>`).join("") + `</tbody>`;
}
function renderAggregateCards() {
  const root = document.getElementById("aggregateCards");
  root.innerHTML = experiments.map(exp => {
    const addVals = valuesFor(exp, "add_acc");
    const mm21Vals = valuesFor(exp, "mm21_acc");
    const hVals = valuesFor(exp, "harmonic");
    return `<article class="summary-card">
      <h3 style="color:${exp.color}">${exp.short}</h3>
      <div class="muted">${activeStage}; ${activeSubView === "overall" ? `mean +/- sample std over ${exp.runs.length} sub-exp checkpoints` : `selected sub-exp ${activeSubView}`}</div>
      <div class="metric-row">
        <div class="metric">add_acc<strong>${fmt(avg(addVals))}</strong><span class="muted">std ${fmt(sampleStd(addVals))}</span></div>
        <div class="metric">mm21_acc<strong>${fmt(avg(mm21Vals))}</strong><span class="muted">std ${fmt(sampleStd(mm21Vals))}</span></div>
        <div class="metric">harmonic<strong>${fmt(avg(hVals))}</strong><span class="muted">std ${fmt(sampleStd(hVals))}</span></div>
      </div>
    </article>`;
  }).join("");
}
function drawAggregateBars() {
  const {ctx, w, h} = setupCanvas(document.getElementById("aggregateBars"));
  ctx.clearRect(0, 0, w, h);
  const plot = {x: 64, y: 28, w: w - 108, h: h - 82};
  drawAxes(ctx, plot, [], [0, .25, .5, .75, 1]);
  const groups = [
    {key: "add_acc", label: "add_acc"},
    {key: "mm21_acc", label: "mm21_acc"},
    {key: "harmonic", label: "harmonic"}
  ];
  const groupW = plot.w / groups.length;
  groups.forEach((group, groupIdx) => {
    const barW = Math.min(58, groupW / 6);
    const gap = 16;
      const exps = experiments;
      const totalW = exps.length * barW + (exps.length - 1) * gap;
      const start = plot.x + groupIdx * groupW + groupW / 2 - totalW / 2;
    exps.forEach((exp, expIdx) => {
      const vals = valuesFor(exp, group.key);
      const value = avg(vals);
      const err = sampleStd(vals);
      const x = start + expIdx * (barW + gap);
      const bh = value * plot.h;
      ctx.fillStyle = exp.color;
      ctx.globalAlpha = .9;
      ctx.fillRect(x, plot.y + plot.h - bh, barW, bh);
      ctx.globalAlpha = 1;
      const yMean = plot.y + plot.h - bh;
      const yErrTop = plot.y + plot.h - Math.min(1, value + err) * plot.h;
      const yErrBottom = plot.y + plot.h - Math.max(0, value - err) * plot.h;
      ctx.strokeStyle = "#172033";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x + barW / 2, yErrTop);
      ctx.lineTo(x + barW / 2, yErrBottom);
      ctx.moveTo(x + barW / 2 - 6, yErrTop);
      ctx.lineTo(x + barW / 2 + 6, yErrTop);
      ctx.moveTo(x + barW / 2 - 6, yErrBottom);
      ctx.lineTo(x + barW / 2 + 6, yErrBottom);
      ctx.stroke();
      ctx.fillStyle = "#172033";
      ctx.font = "12px Arial";
      ctx.fillText(fmt(value), x - 1, yMean - 7);
    });
    ctx.fillStyle = "#334155";
    ctx.font = "13px Arial";
    ctx.fillText(group.label, plot.x + groupIdx * groupW + groupW / 2 - 34, plot.y + plot.h + 30);
  });
}
function drawLineChart() {
  const {ctx, w, h} = setupCanvas(document.getElementById("lineChart"));
  ctx.clearRect(0, 0, w, h);
  const exps = experiments;
  const plot = {x: 64, y: 18, w: w - 250, h: h - 64};
  const allRows = exps.flatMap(exp => visibleRuns(exp).flatMap(run => run.records[activeStage]));
  const maxEpoch = Math.max(1, ...allRows.map(row => row.epoch));
  const yRange = niceMetricRange(allRows);
  const tickStep = maxEpoch <= 20000 ? 5000 : 10000;
  const xTicks = [];
  for (let t = 0; t <= maxEpoch; t += tickStep) xTicks.push(t);
  if (xTicks[xTicks.length - 1] !== maxEpoch) xTicks.push(maxEpoch);
  drawScaledAxes(ctx, plot, xTicks, yTicksForRange(yRange.min, yRange.max), yRange.min, yRange.max, maxEpoch);
  exps.forEach(exp => {
    visibleRuns(exp).forEach((run, runIdx) => {
      [["add_acc", []], ["mm21_acc", [7, 5]]].forEach(([key, dash]) => {
        ctx.save();
        ctx.strokeStyle = exp.color;
        ctx.globalAlpha = activeSubView === "overall" ? .35 + runIdx * .2 : .9;
        ctx.lineWidth = 2;
        ctx.setLineDash(dash);
        ctx.beginPath();
        run.records[activeStage].forEach((row, idx) => {
          const x = plot.x + (row.epoch / maxEpoch) * plot.w;
          const y = plot.y + plot.h - ((row[key] - yRange.min) / (yRange.max - yRange.min)) * plot.h;
          if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.restore();
      });
    });
  });
}
function meanStdText(values) {
  return `${fmt(avg(values))} +/- ${fmt(sampleStd(values))}`;
}
function renderQueryWiseTable() {
  const root = document.getElementById("queryWiseTable");
  const cols = ["add_acc_q1", "add_acc_q2", "add_acc", "mm21_acc_q1", "mm21_acc_q2", "mm21_acc"];
  const show = activeSubView !== "overall";
  setSectionVisible("queryWiseTitle", "queryWiseSection", show);
  if (!show) {
    root.innerHTML = "";
    return;
  }
  root.innerHTML = `<table>
    <thead><tr><th>experiment</th><th>sub-exp</th><th>add q1</th><th>add q2</th><th>add overall</th><th>mm21 q1</th><th>mm21 q2</th><th>mm21 overall</th></tr></thead>
    <tbody>${experiments.map(exp => {
      const run = visibleRuns(exp)[0];
      const row = bestRow(run, activeStage);
      return `<tr>
        <td><strong style="color:${exp.color}">${exp.short}</strong></td>
        <td>sub${run.sub_id}<br><span class="muted">epoch ${run.best_epoch}</span></td>
        ${cols.map(key => `<td>${fmt(row[key])}</td>`).join("")}
      </tr>`;
    }).join("")}
    </tbody>
  </table>`;
}
function renderSubexpTable() {
  const rows = experiments.flatMap(exp => visibleRuns(exp).map(run => ({exp, run, row: bestRow(run, activeStage)})));
  document.getElementById("subexpTable").innerHTML =
    `<thead><tr><th>experiment</th><th>sub-exp</th><th>best epoch</th><th>${activeStage} overall</th><th>q1</th><th>q2</th><th>best eval-H epoch</th></tr></thead><tbody>` +
    rows.map(({exp, run, row}) => `<tr>
      <td><strong style="color:${exp.color}">${exp.short}</strong></td>
      <td>sub${run.sub_id}</td>
      <td>${run.best_epoch}<br><span class="muted">train ${REPORT.selectorKey}=${fmt(bestRow(run, "train")[REPORT.selectorKey])}</span></td>
      <td>add=${fmt(row.add_acc)}<br>mm21=${fmt(row.mm21_acc)}<br>H=${fmt(harmonic(row.add_acc, row.mm21_acc))}</td>
      <td>add=${fmt(row.add_acc_q1)}<br>mm21=${fmt(row.mm21_acc_q1)}</td>
      <td>add=${fmt(row.add_acc_q2)}<br>mm21=${fmt(row.mm21_acc_q2)}</td>
      <td>${run.best_eval_h_epoch == null ? '<span class="muted">train-only</span>' : `${run.best_eval_h_epoch}<br><span class="muted">H=${fmt(run.best_eval_h)}</span>`}</td>
    </tr>`).join("") + `</tbody>`;
}
function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
function pct(value) { return (Number(value) * 100).toFixed(1) + "%"; }
function dominantClass(item) {
  if (item.dominant === "q1") return "dominant-q1";
  if (item.dominant === "q2") return "dominant-q2";
  return "dominant-tie";
}
function criticalPairRows(items) {
  return items.map(item => `<tr>
    <td>(${item.a}, ${item.b})</td>
    <td>add=${item.add_target}<br>mm21=${item.mm21_target}</td>
    <td><span class="${dominantClass(item)}">${item.dominant}</span><br><span class="muted">${pct(item.dominant_rate)}</span></td>
    <td>${pct(item.q1_exclusive_rate)}<br><span class="muted">${item.q1_exclusive_count}/${item.total_epochs}</span></td>
    <td>${pct(item.q2_exclusive_rate)}<br><span class="muted">${item.q2_exclusive_count}/${item.total_epochs}</span></td>
    <td>${pct(item.split_rate)}<br><span class="muted">${item.split_count}/${item.total_epochs}</span></td>
    <td>${pct(item.tie_or_missing_rate)}<br><span class="muted">${item.tie_or_missing_count}/${item.total_epochs}</span></td>
  </tr>`).join("");
}
function criticalPairTable(title, items) {
  return `<div>
    <h3>${escapeHtml(title)}</h3>
    <table>
      <thead><tr><th>pair</th><th>targets</th><th>dominant</th><th>q1 exclusive</th><th>q2 exclusive</th><th>split</th><th>tie/missing</th></tr></thead>
      <tbody>${criticalPairRows(items)}</tbody>
    </table>
  </div>`;
}
function renderCriticalPairs() {
  const reports = activeSubView === "overall" ? [] : (REPORT.criticalPairReports || []).filter(report => report.sub_exp_id === activeSubView);
  const section = document.getElementById("criticalPairSection");
  const title = document.getElementById("criticalPairTitle");
  if (!reports.length) {
    section.style.display = "none";
    title.style.display = "none";
    return;
  }
  section.style.display = "";
  title.style.display = "";
  document.getElementById("criticalPairReports").innerHTML = reports.map(report => `<details class="critical-report">
    <summary>${escapeHtml(report.label)} - critical pairs=${report.critical_pair_count}; last epoch=${report.last_epoch}</summary>
    <div class="muted">source=${escapeHtml(report.source)}; critical pairs=${report.critical_pair_count}; rows=${report.row_count}; last epoch=${report.last_epoch}</div>
    <div class="critical-grid">
      ${criticalPairTable("Top dominated pairs over all intervals", report.top_all)}
      ${criticalPairTable("Top dominated pairs in last interval", report.top_last)}
    </div>
  </details>`).join("");
}
function pairRiskSingleRows(items) {
  return items.map(item => `<tr>
    <td>(${item.a}, ${item.b})</td>
    <td>${escapeHtml(item.pair_type)}<br><span class="muted">target=${escapeHtml(item.target_op)}</span></td>
    <td>add=${item.add_target}<br>mm21=${item.mm21_target}</td>
    <td>${pct(item.risk_score)}</td>
    <td><span class="${dominantClass({dominant: item.dominant_query})}">${escapeHtml(item.dominant_query)}</span></td>
    <td>${pct(item.q1_only_rate)}<br><span class="muted">${item.q1_only_count}/${item.total_epochs}</span></td>
    <td>${pct(item.q2_only_rate)}<br><span class="muted">${item.q2_only_count}/${item.total_epochs}</span></td>
    <td>${pct(item.mixed_rate)}<br><span class="muted">${item.mixed_count}/${item.total_epochs}</span></td>
    <td>${pct(item.tie_or_missing_rate)}<br><span class="muted">${item.tie_or_missing_count}/${item.total_epochs}</span></td>
  </tr>`).join("");
}
function pairRiskDualRows(items) {
  return items.map(item => `<tr>
    <td>(${item.a}, ${item.b})</td>
    <td>add=${item.add_target}<br>mm21=${item.mm21_target}</td>
    <td>${pct(item.risk_score)}</td>
    <td><span class="${dominantClass({dominant: item.dominant_query})}">${escapeHtml(item.dominant_query)}</span></td>
    <td>${pct(item.q1_exclusive_rate)}<br><span class="muted">${item.q1_exclusive_count}/${item.total_epochs}</span></td>
    <td>${pct(item.q2_exclusive_rate)}<br><span class="muted">${item.q2_exclusive_count}/${item.total_epochs}</span></td>
    <td>${pct(item.split_rate)}<br><span class="muted">${item.split_count}/${item.total_epochs}</span></td>
    <td>${pct(item.mixed_or_missing_rate)}<br><span class="muted">${item.mixed_or_missing_count}/${item.total_epochs}</span></td>
  </tr>`).join("");
}
function pairRiskSingleTable(title, items) {
  return `<div>
    <h3>${escapeHtml(title)}</h3>
    <table>
      <thead><tr><th>pair</th><th>type</th><th>targets</th><th>risk</th><th>dominant</th><th>q1 only</th><th>q2 only</th><th>mixed</th><th>tie/missing</th></tr></thead>
      <tbody>${pairRiskSingleRows(items)}</tbody>
    </table>
  </div>`;
}
function pairRiskDualTable(title, items) {
  return `<div>
    <h3>${escapeHtml(title)}</h3>
    <table>
      <thead><tr><th>pair</th><th>targets</th><th>risk</th><th>dominant</th><th>q1 exclusive</th><th>q2 exclusive</th><th>split</th><th>mixed/missing</th></tr></thead>
      <tbody>${pairRiskDualRows(items)}</tbody>
    </table>
  </div>`;
}
function renderPairRisks() {
  const reports = activeSubView === "overall" ? [] : (REPORT.pairRiskReports || []).filter(report => report.sub_exp_id === activeSubView);
  const section = document.getElementById("pairRiskSection");
  const title = document.getElementById("pairRiskTitle");
  if (!reports.length) {
    section.style.display = "none";
    title.style.display = "none";
    return;
  }
  section.style.display = "";
  title.style.display = "";
  document.getElementById("pairRiskReports").innerHTML = reports.map(report => `<details class="critical-report">
    <summary>${escapeHtml(report.label)} - pairs=${report.pair_count}; single=${report.single_pair_count}; dual=${report.dual_pair_count}; last epoch=${report.last_epoch}</summary>
    <div class="muted">source=${escapeHtml(report.source)}; pairs=${report.pair_count}; single=${report.single_pair_count}; dual=${report.dual_pair_count}; rows=${report.row_count}; last epoch=${report.last_epoch}</div>
    <div class="critical-grid">
      ${pairRiskSingleTable("Single-op competition risk over all intervals", report.top_single_all)}
      ${pairRiskSingleTable("Single-op competition risk in last interval", report.top_single_last)}
      ${pairRiskDualTable("Dual-op same-query dominance over all intervals", report.top_dual_all)}
      ${pairRiskDualTable("Dual-op same-query dominance in last interval", report.top_dual_last)}
    </div>
  </details>`).join("");
}
function renderImages() {
  const root = document.getElementById("imageRuns");
  const show = activeSubView !== "overall";
  setSectionVisible("dataPairTitle", "dataPairSection", show);
  if (!show) {
    root.innerHTML = "";
    return;
  }
  root.innerHTML = experiments.map(exp => `<article class="run-card">
    <h3 style="color:${exp.color}">${exp.short}</h3>
    <div class="subrun-grid">${visibleRuns(exp).map(run => {
      const row = bestRow(run, activeStage);
      const imgs = run.images[activeStage];
      return `<section class="image-row">
        <div class="image-row-head">
          <strong>sub${run.sub_id}</strong>
          <span class="muted">epoch ${run.best_epoch}; ${activeStage} add=${fmt(row.add_acc)}, mm21=${fmt(row.mm21_acc)}, H=${fmt(harmonic(row.add_acc, row.mm21_acc))}</span>
        </div>
        <div class="image-pair">
          <figure><button class="img-btn" type="button" data-full-src="${imgs[0]}" data-caption="${exp.short} sub${run.sub_id} ${activeStage} add set"><img src="${imgs[0]}" alt="${exp.short} sub${run.sub_id} ${activeStage} add set"></button><figcaption>add set</figcaption></figure>
          <figure><button class="img-btn" type="button" data-full-src="${imgs[1]}" data-caption="${exp.short} sub${run.sub_id} ${activeStage} mm21 set"><img src="${imgs[1]}" alt="${exp.short} sub${run.sub_id} ${activeStage} mm21 set"></button><figcaption>mm21 set</figcaption></figure>
        </div>
      </section>`;
    }).join("")}</div>
  </article>`).join("");
  bindLightboxButtons();
}
function renderLegends() {
  const exps = experiments;
  const expLegend = exps.map(exp => `<span class="legend-item"><span class="swatch" style="background:${exp.color}"></span>${exp.short}</span>`).join("");
  document.getElementById("aggregateLegend").innerHTML = `<span class="legend-title">颜色=实验</span>${expLegend}<span class="legend-title">误差线=sample std</span>`;
  document.getElementById("lineLegend").innerHTML = `<span class="legend-title">颜色=实验</span>${expLegend}<span class="legend-item"><span class="line-sample"></span>add_acc</span><span class="legend-item"><span class="line-sample line-dashed"></span>mm21_acc</span><span class="legend-title">透明度=sub-exp</span>`;
  document.getElementById("queryWiseLegend").innerHTML = `<span class="legend-title">表格列名已经标明 q1 / q2 / overall 和 add / mm21</span>`;
}
function openLightbox(src, caption) {
  const box = document.getElementById("lightbox");
  const img = document.getElementById("lightboxImg");
  const cap = document.getElementById("lightboxCaption");
  img.src = src;
  img.alt = caption;
  cap.textContent = caption;
  box.classList.add("open");
  box.setAttribute("aria-hidden", "false");
}
function closeLightbox() {
  const box = document.getElementById("lightbox");
  const img = document.getElementById("lightboxImg");
  box.classList.remove("open");
  box.setAttribute("aria-hidden", "true");
  img.src = "";
}
function bindLightboxButtons() {
  document.querySelectorAll(".img-btn").forEach(btn => {
    btn.addEventListener("click", () => openLightbox(btn.dataset.fullSrc, btn.dataset.caption));
  });
}
function renderAll() {
  renderStageSwitch();
  renderSetup();
  renderAggregateCards();
  renderLegends();
  drawAggregateBars();
  drawLineChart();
  renderQueryWiseTable();
  renderSubexpTable();
  renderCriticalPairs();
  renderPairRisks();
  renderImages();
}
document.getElementById("lightboxClose").addEventListener("click", closeLightbox);
document.getElementById("lightbox").addEventListener("click", event => { if (event.target.id === "lightbox") closeLightbox(); });
document.addEventListener("keydown", event => { if (event.key === "Escape") closeLightbox(); });
window.addEventListener("resize", () => { drawAggregateBars(); drawLineChart(); });
renderAll();
</script>
</body>
</html>"""
    return (
        template
        .replace("__TITLE__", escaped_title)
        .replace("__ANALYSIS__", analysis_html)
        .replace("__PAYLOAD__", payload)
        .replace("__SELECTOR__", html.escape(selector_key))
        .replace("__CHECKPOINT_NOTE__", checkpoint_note)
    )


def build_default_repeated_analysis(experiments, selector_key):
    report_stage = "eval" if all("eval" in exp.get("summary", {}) for exp in experiments) else "train"
    ranked = []
    for exp in experiments:
        h_mean = exp["summary"][report_stage]["harmonic"]["mean"]
        add_mean = exp["summary"][report_stage]["add_acc"]["mean"]
        mm21_mean = exp["summary"][report_stage]["mm21_acc"]["mean"]
        ranked.append((h_mean, add_mean, mm21_mean, exp))
    ranked.sort(reverse=True, key=lambda item: item[0])
    h_mean, add_mean, mm21_mean, leader = ranked[0]
    selection_note = (
        f"Best epoch is selected per sub-exp by lowest train {selector_key} among epochs visible in Eval_record and query operation images."
        if report_stage == "eval"
        else f"No Eval_record is present, so best epoch is selected per sub-exp by lowest train {selector_key} among epochs with query operation images."
    )
    return (
        "## Key Takeaways\n"
        f"- {selection_note}\n"
        f"- On {report_stage}, {leader['short']} has the highest mean harmonic over sub-exps: "
        f"H={h_mean:.3f}, add={add_mean:.3f}, mm21={mm21_mean:.3f}.\n"
        "- This repeated-run report should be read by mean/std first, then by sub-exp details for instability.\n"
    )


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
        "--analysis-file",
        help="Optional plain text/markdown-ish file rendered as the top experiment analysis section.",
    )
    parser.add_argument(
        "--sub-exps",
        type=parse_sub_exps,
        default=["1"],
        help="Comma-separated sub-exp ids to aggregate. Use '1,2,3' for repeated-run reports.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment_arg,
        help="Repeatable. Format: 'short|experiment_dir_name|Display label|#color'",
    )
    parser.add_argument(
        "--critical-pair",
        action="append",
        type=parse_critical_pair_arg,
        help="Repeatable. Format: 'short|experiment_dir_name|sub_exp_id|Display label'",
    )
    parser.add_argument(
        "--pair-risk",
        action="append",
        type=parse_pair_risk_arg,
        help="Repeatable. Format: 'short|experiment_dir_name|sub_exp_id|Display label'",
    )
    parser.add_argument(
        "--critical-pair-top-n",
        type=int,
        default=20,
        help="Number of dominated critical pairs to show per critical-pair table.",
    )
    args = parser.parse_args()

    specs = args.experiment if args.experiment else DEFAULT_EXPERIMENTS
    analysis_text = ""
    if args.analysis_file:
        analysis_text = Path(args.analysis_file).read_text(encoding="utf-8")

    out_dir = ANALYSIS_DIR / args.report_name
    assets_dir = out_dir / "assets"
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(args.sub_exps) == 1 and args.sub_exps[0] == "1":
        experiments = [build_experiment(spec, out_dir, args.selector_key) for spec in specs]
        index_html = render_index(args.title, args.report_name, experiments, args.selector_key, analysis_text)
    else:
        experiments = [
            build_repeated_experiment(spec, out_dir, args.selector_key, args.sub_exps)
            for spec in specs
        ]
        critical_pair_reports = [
            build_critical_pair_report(spec, top_n=args.critical_pair_top_n)
            for spec in (args.critical_pair or [])
        ]
        pair_risk_reports = [
            build_pair_risk_report(spec, top_n=args.critical_pair_top_n)
            for spec in (args.pair_risk or [])
        ]
        index_html = render_repeated_index(
            args.title,
            args.report_name,
            experiments,
            args.selector_key,
            analysis_text,
            critical_pair_reports,
            pair_risk_reports,
        )
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")
    (ANALYSIS_DIR / f"{args.report_name}.html").write_text(render_redirect(args.report_name), encoding="utf-8")

    print(f"Wrote {out_dir / 'index.html'}")
    for exp in experiments:
        if "runs" in exp:
            best_epochs = ", ".join(f"sub{run['sub_id']}={run['best_epoch']}" for run in exp["runs"])
            print(f"{exp['short']}: {best_epochs}")
        else:
            print(f"{exp['short']}: best_epoch={exp['best_epoch']}")
    for report in (critical_pair_reports if "critical_pair_reports" in locals() else []):
        print(
            f"{report['short']}: critical_pairs={report['critical_pair_count']}, "
            f"last_epoch={report['last_epoch']}"
        )
    for report in (pair_risk_reports if "pair_risk_reports" in locals() else []):
        print(
            f"{report['short']}: pair_risks={report['pair_count']}, "
            f"last_epoch={report['last_epoch']}"
        )


if __name__ == "__main__":
    main()
