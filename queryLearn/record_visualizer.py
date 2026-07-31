import html
import os
from pathlib import Path


DEFAULT_RECORD_PLOT_METRICS = {
    'accuracy': [
        'add_acc_q1',
        'add_acc_q2',
        'mm21_acc_q1',
        'mm21_acc_q2',
        'add_acc',
        'mm21_acc',
    ],
    'loss': [
        'oper_loss',
        'hard_min_loss',
        'symm_loss',
        'total_loss',
    ],
}


def parse_record_file(record_path):
    records = []
    if record_path is None or not os.path.exists(record_path):
        return records
    with open(record_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            epoch_text, payload = line.split('-', 1)
            row = {'epoch': int(epoch_text)}
            for item in payload.split(','):
                key, value = item.split(':', 1)
                row[key] = float(value)
            records.append(row)
    return records


def best_record_metric(record_path, metric_name):
    values = [
        row[metric_name]
        for row in parse_record_file(record_path)
        if metric_name in row
    ]
    return min(values) if values else None


def _safe_name(name):
    return ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in str(name))


def _normalize_metric_groups(metric_groups):
    if metric_groups is None:
        return DEFAULT_RECORD_PLOT_METRICS
    if isinstance(metric_groups, (list, tuple)):
        return {'metrics': list(metric_groups)}
    return {
        str(group_name): list(metrics)
        for group_name, metrics in metric_groups.items()
    }


class RecordVisualizer:
    def __init__(self, config, train_record_path, eval_record_path=None):
        vis_config = config.get('record_visualizer', {})
        if isinstance(vis_config, bool):
            vis_config = {'enabled': vis_config}
        self.enabled = vis_config.get('enabled', True)
        self.train_record_path = train_record_path
        self.eval_record_path = eval_record_path
        self.output_dir = vis_config.get('output_dir', 'RecordPlots')
        self.file_format = vis_config.get('format', 'png').lower().lstrip('.')
        self.metric_groups = _normalize_metric_groups(vis_config.get('metrics', None))
        self.dpi = vis_config.get('dpi', 150)
        self.max_points = vis_config.get('max_points', None)
        self._warned = False

    def refresh(self):
        if not self.enabled:
            return
        try:
            train_records = parse_record_file(self.train_record_path)
            eval_records = parse_record_file(self.eval_record_path)
            if not train_records and not eval_records:
                return

            os.makedirs(self.output_dir, exist_ok=True)
            written = []
            for group_name, metrics in self.metric_groups.items():
                output_path = os.path.join(
                    self.output_dir,
                    f'{_safe_name(group_name)}.{self.file_format}',
                )
                if self._plot_metric_group(output_path, group_name, metrics, train_records, eval_records):
                    written.append(output_path)
            self._write_index(written)
        except Exception as exc:
            if not self._warned:
                print(f"WARNING: record visualization failed: {exc}")
                self._warned = True

    def _plot_metric_group(self, output_path, group_name, metrics, train_records, eval_records):
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        series = []
        for stage_name, records, linestyle in [
                ('train', train_records, '-'),
                ('eval', eval_records, '--')]:
            if self.max_points is not None and len(records) > self.max_points:
                records = records[-int(self.max_points):]
            for metric in metrics:
                points = [
                    (row['epoch'], row[metric])
                    for row in records
                    if metric in row
                ]
                if points:
                    series.append((stage_name, metric, linestyle, points))

        if not series:
            return False

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=self.dpi)
        for stage_name, metric, linestyle, points in series:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax.plot(xs, ys, linestyle=linestyle, linewidth=1.8, label=f'{stage_name}/{metric}')

        ax.set_title(str(group_name))
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        return True

    def _write_index(self, image_paths):
        index_path = Path(self.output_dir) / 'index.html'
        lines = [
            '<!doctype html>',
            '<meta charset="utf-8">',
            '<title>queryLearn record plots</title>',
            '<style>',
            'body{font-family:system-ui,sans-serif;margin:24px;background:#f8fafc;color:#111827}',
            'section{margin-bottom:28px}',
            'img{max-width:100%;height:auto;background:white;border:1px solid #d1d5db}',
            '</style>',
            '<h1>queryLearn record plots</h1>',
        ]
        for path in image_paths:
            name = Path(path).name
            lines.extend([
                '<section>',
                f'<h2>{html.escape(Path(name).stem)}</h2>',
                f'<img src="{html.escape(name)}" alt="{html.escape(name)}">',
                '</section>',
            ])
        index_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
