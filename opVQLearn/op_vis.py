import os


OPERATIONS = ("add", "mm21")


def _as_python(value):
    return value.item() if hasattr(value, "item") else value


def _is_true(values, idx):
    return bool(_as_python(values[idx]))


def _code_label(values, idx):
    return str(int(_as_python(values[idx])) + 1)


def _is_add(a, b, c):
    return c == a + b


def _is_mm21(a, b, c):
    return c == (a * b) % 21


def _is_special_pair(a, b):
    return (a + b) == ((a * b) % 21)


def _is_operation_sample(operation, a, b, c):
    if operation == "add":
        return _is_add(a, b, c)
    if operation == "mm21":
        return _is_mm21(a, b, c)
    raise ValueError(f"Unsupported operation: {operation}")


def assignment_cells(label_a, label_b, label_c, op_indices, pred_correct, operation):
    cells = {}
    for idx, (a, b, c) in enumerate(zip(label_a, label_b, label_c)):
        if not _is_operation_sample(operation, a, b, c):
            continue
        cell = cells.setdefault((a, b), {"seen": False, "codes": set(), "all_correct": True, "count": 0})
        cell["seen"] = True
        cell["codes"].add(_code_label(op_indices, idx))
        cell["all_correct"] = cell["all_correct"] and _is_true(pred_correct, idx)
        cell["count"] += 1
    return cells


def cell_text(cell):
    if not cell["seen"]:
        return ""
    return ",".join(sorted(cell["codes"]))


def table_grids(rows, cols, cells):
    text_grid = []
    color_grid = []
    for row in rows:
        text_row = []
        color_row = []
        for col in cols:
            cell = cells.get((row, col), {"seen": False, "codes": set(), "all_correct": False})
            text_row.append(cell_text(cell))
            if not cell["seen"]:
                color_row.append(0)
            elif cell["all_correct"]:
                color_row.append(1)
            else:
                color_row.append(2)
        text_grid.append(text_row)
        color_grid.append(color_row)
    return text_grid, color_grid


def assignment_stats(cells):
    seen = [cell for cell in cells.values() if cell["seen"]]
    total = len(seen)
    correct = sum(1 for cell in seen if cell["all_correct"])
    code1 = sum(1 for cell in seen if "1" in cell["codes"])
    code2 = sum(1 for cell in seen if "2" in cell["codes"])
    return {
        "total": total,
        "correct": correct / total if total else 0.0,
        "code1": code1 / total if total else 0.0,
        "code2": code2 / total if total else 0.0,
    }


def save_assignment_table_plot(output_path, operation, stage, epoch, rows, cols, cells):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    text_grid, color_grid = table_grids(rows, cols, cells)
    stats = assignment_stats(cells)
    fig_w = max(6, 0.42 * len(cols) + 1.2)
    fig_h = max(5, 0.34 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    cmap = ListedColormap(["#ffffff", "#dcfce7", "#fee2e2"])
    ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_title(
        (
            f"{operation} assignment {stage} epoch {epoch} "
            f"cell_acc={stats['correct']:.2f} code1={stats['code1']:.2f} code2={stats['code2']:.2f}"
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
    ax.grid(which="minor", color="#d8dee8", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for y, row in enumerate(rows):
        for x, col in enumerate(cols):
            if _is_special_pair(row, col):
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="#111827", linewidth=2.0))
            text = text_grid[y][x]
            if text:
                ax.text(x, y, text, ha="center", va="center", color="#172033", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_op_assignment_tables(
        output_dir,
        stage,
        epoch,
        label_a,
        label_b,
        label_c,
        op_indices,
        pred_correct,
        file_format):
    file_format = file_format.lower().lstrip(".")
    if file_format not in {"png", "svg"}:
        raise ValueError(f"Unsupported op_vis_format: {file_format}")

    os.makedirs(output_dir, exist_ok=True)
    rows = sorted(set(label_a))
    cols = sorted(set(label_b))
    for operation in OPERATIONS:
        cells = assignment_cells(label_a, label_b, label_c, op_indices, pred_correct, operation)
        file_name = f"op_assignment_{stage}_epoch_{epoch:06d}_{operation}.{file_format}"
        save_assignment_table_plot(
            os.path.join(output_dir, file_name),
            operation,
            stage,
            epoch,
            rows,
            cols,
            cells,
        )
