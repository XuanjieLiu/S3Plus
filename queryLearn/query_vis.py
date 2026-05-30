import os


OPERATIONS = ("add", "mm21")
Q1_BG = "#dbeafe"
Q2_BG = "#fef3c7"
TIE_BG = "#f8fafc"
MISSING_BG = "#ffffff"
Q1_TEXT = "#1d4ed8"
Q2_TEXT = "#d97706"
MISS_TEXT = "#dc2626"
GRID_LINE = "#d8dee8"
SPECIAL_BORDER = "#172033"


def record_dir(record_path):
    record_dir_path = os.path.dirname(record_path)
    return record_dir_path if record_dir_path else "."


def _is_true(values, idx):
    value = values[idx]
    return bool(value.item()) if hasattr(value, "item") else bool(value)


def _to_float(values, idx):
    value = values[idx]
    return float(value.item()) if hasattr(value, "item") else float(value)


def _is_add(a, b, c):
    return c == a + b


def _is_mm21(a, b, c):
    return c == (a * b) % 21


def _is_operation_sample(operation, a, b, c):
    if operation == "add":
        return _is_add(a, b, c)
    if operation == "mm21":
        return _is_mm21(a, b, c)
    raise ValueError(f"Unsupported operation: {operation}")


def _is_special_pair(a, b):
    return (a + b) == ((a * b) % 21)


def distance_winner(q1_dist, q2_dist, tie_eps=1e-8):
    if q1_dist < q2_dist - tie_eps:
        return "q1"
    if q2_dist < q1_dist - tie_eps:
        return "q2"
    return "tie"


def operation_cells(
        label_a,
        label_b,
        label_c,
        q1_correct,
        q2_correct,
        q1_target_dist,
        q2_target_dist,
        operation):
    cells = {}
    for idx, (a, b, c) in enumerate(zip(label_a, label_b, label_c)):
        if not _is_operation_sample(operation, a, b, c):
            continue
        pair = (a, b)
        if pair in cells:
            continue

        q1_dist = _to_float(q1_target_dist, idx)
        q2_dist = _to_float(q2_target_dist, idx)
        cells[pair] = {
            "in_set": True,
            "q1": _is_true(q1_correct, idx),
            "q2": _is_true(q2_correct, idx),
            "winner": distance_winner(q1_dist, q2_dist),
            "special": _is_special_pair(a, b),
        }
    return cells


def operation_cell_text(cell):
    if not cell["in_set"]:
        return "/"
    if cell["q1"] and cell["q2"]:
        return "12"
    if cell["q1"]:
        return "1"
    if cell["q2"]:
        return "2"
    return "×"


def operation_cell_draw_items(cell):
    text = operation_cell_text(cell)
    if text == "12":
        return [
            {"text": "1", "x_offset": -0.09, "color": Q1_TEXT, "fontsize": 10, "fontweight": "bold"},
            {"text": "2", "x_offset": 0.09, "color": Q2_TEXT, "fontsize": 10, "fontweight": "bold"},
        ]
    if text == "1":
        return [{"text": "1", "x_offset": 0.0, "color": Q1_TEXT, "fontsize": 10, "fontweight": "bold"}]
    if text == "2":
        return [{"text": "2", "x_offset": 0.0, "color": Q2_TEXT, "fontsize": 10, "fontweight": "bold"}]
    if text == "×":
        return [{"text": "×", "x_offset": 0.0, "color": MISS_TEXT, "fontsize": 16, "fontweight": "bold"}]
    return [{"text": "/", "x_offset": 0.0, "color": "#94a3b8", "fontsize": 9, "fontweight": "normal"}]


def operation_table_grid(rows, cols, cells):
    cell_grid = []
    bg_grid = []
    color_idx = {
        "missing": 0,
        "tie": 1,
        "q1": 2,
        "q2": 3,
    }
    for row in rows:
        cell_row = []
        bg_row = []
        for col in cols:
            cell = cells.get((row, col), {"in_set": False, "q1": False, "q2": False, "winner": "missing"})
            cell_row.append(cell)
            bg_row.append(color_idx[cell["winner"] if cell["in_set"] else "missing"])
        cell_grid.append(cell_row)
        bg_grid.append(bg_row)
    return cell_grid, bg_grid


def operation_stats(cells):
    total = sum(1 for cell in cells.values() if cell["in_set"])
    q1 = sum(1 for cell in cells.values() if cell["in_set"] and cell["q1"])
    q2 = sum(1 for cell in cells.values() if cell["in_set"] and cell["q2"])
    either = sum(1 for cell in cells.values() if cell["in_set"] and (cell["q1"] or cell["q2"]))
    return {
        "total": total,
        "q1_acc": q1 / total if total else 0.0,
        "q2_acc": q2 / total if total else 0.0,
        "overall": either / total if total else 0.0,
    }


def save_operation_table_plot(output_path, operation, stage, epoch, rows, cols, cells):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cell_grid, color_grid = operation_table_grid(rows, cols, cells)
    stats = operation_stats(cells)
    fig_w = max(6, 0.42 * len(cols) + 1.2)
    fig_h = max(5, 0.34 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    cmap = ListedColormap([MISSING_BG, TIE_BG, Q1_BG, Q2_BG])
    ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_title(
        (
            f"{operation} set {stage} epoch {epoch} "
            f"q1={stats['q1_acc']:.2f} q2={stats['q2_acc']:.2f} overall={stats['overall']:.2f}"
        ),
        pad=12,
    )
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    ax.set_xlabel("label b")
    ax.set_ylabel("label a")
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=GRID_LINE, linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for y, cell_row in enumerate(cell_grid):
        for x, cell in enumerate(cell_row):
            for item in operation_cell_draw_items(cell):
                ax.text(
                    x + item["x_offset"],
                    y,
                    item["text"],
                    ha="center",
                    va="center",
                    color=item["color"],
                    fontsize=item["fontsize"],
                    fontweight=item["fontweight"],
                )
            if _is_special_pair(rows[y], cols[x]):
                ax.add_patch(
                    patches.Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=SPECIAL_BORDER,
                        linewidth=2.2,
                    )
                )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_query_operation_tables(
        output_dir,
        stage,
        epoch,
        label_a,
        label_b,
        label_c,
        q1_correct,
        q2_correct,
        q1_target_dist,
        q2_target_dist,
        file_format):
    file_format = file_format.lower().lstrip(".")
    if file_format not in {"png", "svg"}:
        raise ValueError(f"Unsupported query_vis_format: {file_format}")

    os.makedirs(output_dir, exist_ok=True)
    rows = sorted(set(label_a))
    cols = sorted(set(label_b))
    for operation in OPERATIONS:
        cells = operation_cells(
            label_a,
            label_b,
            label_c,
            q1_correct,
            q2_correct,
            q1_target_dist,
            q2_target_dist,
            operation,
        )
        file_name = f"query_operation_{stage}_epoch_{epoch:06d}_{operation}.{file_format}"
        save_operation_table_plot(
            os.path.join(output_dir, file_name),
            operation,
            stage,
            epoch,
            rows,
            cols,
            cells,
        )
