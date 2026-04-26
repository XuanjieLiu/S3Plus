import os


def record_dir(record_path):
    record_dir_path = os.path.dirname(record_path)
    return record_dir_path if record_dir_path else '.'


def operation_cells(label_a, label_b, label_c, q_correct):
    cells = {}
    for a, b in zip(label_a, label_b):
        cells.setdefault((a, b), {'add': False, 'mul': False})

    for i, (a, b, c) in enumerate(zip(label_a, label_b, label_c)):
        if not q_correct[i].item():
            continue
        cell = cells.setdefault((a, b), {'add': False, 'mul': False})
        if c == a + b:
            cell['add'] = True
        if c == (a * b) % 21:
            cell['mul'] = True
    return cells


def operation_cell_text(cell):
    if cell['add'] and cell['mul']:
        return '+/*m'
    if cell['add']:
        return '+'
    if cell['mul']:
        return '*m'
    return '?'


def operation_table_grid(rows, cols, cells):
    text_grid = []
    color_grid = []
    color_idx = {
        '?': 0,
        '+': 1,
        '*m': 2,
        '+/*m': 3,
    }
    for row in rows:
        text_row = []
        color_row = []
        for col in cols:
            text = operation_cell_text(cells.get((row, col), {'add': False, 'mul': False}))
            text_row.append(text)
            color_row.append(color_idx[text])
        text_grid.append(text_row)
        color_grid.append(color_row)
    return text_grid, color_grid


def save_operation_table_plot(output_path, query_name, stage, epoch, rows, cols, cells, add_acc, mul_acc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    text_grid, color_grid = operation_table_grid(rows, cols, cells)
    fig_w = max(6, 0.42 * len(cols) + 1.2)
    fig_h = max(5, 0.34 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    cmap = ListedColormap(['#f4f5f7', '#dff3e6', '#dfe9fb', '#eadff7'])
    ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=3, aspect='auto')
    ax.set_title(f'{query_name} {stage} epoch {epoch} add_acc={add_acc:.2f} mul_acc={mul_acc:.2f}', pad=12)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    ax.set_xlabel('label b')
    ax.set_ylabel('label a')
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which='minor', color='#d8dee8', linewidth=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)
    text_colors = {
        '?': '#777777',
        '+': '#176b35',
        '*m': '#1b4f9c',
        '+/*m': '#63308f',
    }
    for y, text_row in enumerate(text_grid):
        for x, text in enumerate(text_row):
            ax.text(x, y, text, ha='center', va='center', color=text_colors[text], fontweight='bold')
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
        query_specs,
        file_format):
    file_format = file_format.lower().lstrip('.')
    if file_format not in {'png', 'svg'}:
        raise ValueError(f"Unsupported query_vis_format: {file_format}")

    os.makedirs(output_dir, exist_ok=True)
    rows = sorted(set(label_a))
    cols = sorted(set(label_b))
    for query_name, query_correct, add_acc, mul_acc in query_specs:
        cells = operation_cells(label_a, label_b, label_c, query_correct)
        file_name = (
            f'query_operation_{stage}_epoch_{epoch:06d}_{query_name}'
            f'_add{add_acc:.2f}_mul{mul_acc:.2f}.{file_format}'
        )
        save_operation_table_plot(
            os.path.join(output_dir, file_name),
            query_name,
            stage,
            epoch,
            rows,
            cols,
            cells,
            add_acc,
            mul_acc,
        )
