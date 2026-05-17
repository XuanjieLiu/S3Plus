import os


OPERATIONS = ("add", "mm21")


def record_dir(record_path):
    record_dir_path = os.path.dirname(record_path)
    return record_dir_path if record_dir_path else "."


def _is_true(values, idx):
    value = values[idx]
    return bool(value.item()) if hasattr(value, "item") else bool(value)


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


def operation_cells(label_a, label_b, label_c, q1_correct, q2_correct, operation):
    cells = {}
    for idx, (a, b, c) in enumerate(zip(label_a, label_b, label_c)):
        if not _is_operation_sample(operation, a, b, c):
            continue

        cell = cells.setdefault((a, b), {"in_set": False, "q1": False, "q2": False})
        cell["in_set"] = True
        cell["q1"] = cell["q1"] or _is_true(q1_correct, idx)
        cell["q2"] = cell["q2"] or _is_true(q2_correct, idx)
    return cells


def operation_cell_text(cell):
    if not cell["in_set"]:
        return "/"
    if cell["q1"] and cell["q2"]:
        return "1,2"
    if cell["q1"]:
        return "1"
    if cell["q2"]:
        return "2"
    return "×"


def operation_table_grid(rows, cols, cells):
    text_grid = []
    color_grid = []
    color_idx = {
        "default": 0,
        "special": 1,
    }
    for row in rows:
        text_row = []
        color_row = []
        for col in cols:
            cell = cells.get((row, col), {"in_set": False, "q1": False, "q2": False})
            text_row.append(operation_cell_text(cell))
            color_row.append(color_idx["special" if _is_special_pair(row, col) else "default"])
        text_grid.append(text_row)
        color_grid.append(color_row)
    return text_grid, color_grid


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
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    text_grid, color_grid = operation_table_grid(rows, cols, cells)
    stats = operation_stats(cells)
    fig_w = max(6, 0.42 * len(cols) + 1.2)
    fig_h = max(5, 0.34 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    cmap = ListedColormap(["#f8fafc", "#efe5ff"])
    ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
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
    ax.grid(which="minor", color="#d8dee8", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for y, text_row in enumerate(text_grid):
        for x, text in enumerate(text_row):
            color = "#c62828" if text == "×" else "#172033"
            ax.text(x, y, text, ha="center", va="center", color=color, fontweight="bold")
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
        file_format):
    file_format = file_format.lower().lstrip(".")
    if file_format not in {"png", "svg"}:
        raise ValueError(f"Unsupported query_vis_format: {file_format}")

    os.makedirs(output_dir, exist_ok=True)
    rows = sorted(set(label_a))
    cols = sorted(set(label_b))
    for operation in OPERATIONS:
        cells = operation_cells(label_a, label_b, label_c, q1_correct, q2_correct, operation)
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
