import os
from bisect import bisect_right

from VQ.common_func import parse_label


OP_ADD = 'add'
OP_MM21 = 'mm21'
PAIR_TYPE_SINGLE_ADD = 'single_add'
PAIR_TYPE_SINGLE_MM21 = 'single_mm21'
PAIR_TYPE_DUAL_DISTINCT = 'dual_distinct'
PAIR_TYPE_SPECIAL_AMBIGUOUS = 'special_ambiguous'
PAIR_RISK_TYPES = {PAIR_TYPE_SINGLE_ADD, PAIR_TYPE_SINGLE_MM21, PAIR_TYPE_DUAL_DISTINCT}

CRITICAL_PAIR_RECORD_COLUMNS = [
    'epoch',
    'a',
    'b',
    'add_target',
    'mm21_target',
    'total_epochs',
    'q1_exclusive_rate',
    'q2_exclusive_rate',
    'split_rate',
    'tie_or_missing_rate',
    'q1_exclusive_count',
    'q2_exclusive_count',
    'split_count',
    'tie_or_missing_count',
]

PAIR_RISK_RECORD_COLUMNS = [
    'epoch',
    'a',
    'b',
    'pair_type',
    'present_ops',
    'add_target',
    'mm21_target',
    'total_epochs',
    'risk_score',
    'target_op',
    'q1_only_rate',
    'q2_only_rate',
    'mixed_rate',
    'tie_or_missing_rate',
    'dominant_query',
    'competition_rate',
    'q1_exclusive_rate',
    'q2_exclusive_rate',
    'split_rate',
    'mixed_or_missing_rate',
    'dominance_rate',
    'q1_only_count',
    'q2_only_count',
    'mixed_count',
    'tie_or_missing_count',
    'q1_exclusive_count',
    'q2_exclusive_count',
    'split_count',
    'mixed_or_missing_count',
]


def add_target(a, b):
    return a + b


def mm21_target(a, b):
    return (a * b) % 21


def is_special_pair(a, b):
    return add_target(a, b) == mm21_target(a, b)


def infer_operation(a, b, c):
    if c == add_target(a, b):
        return OP_ADD
    if c == mm21_target(a, b):
        return OP_MM21
    return None


def build_pair_taxonomy(label_triples):
    pair_ops = {}
    special_pairs = {}
    for a, b, c in label_triples:
        pair = (a, b)
        pair_add_target = add_target(a, b)
        pair_mm21_target = mm21_target(a, b)
        if pair_add_target == pair_mm21_target:
            if c == pair_add_target:
                special_pairs[pair] = {
                    'pair_type': PAIR_TYPE_SPECIAL_AMBIGUOUS,
                    'present_ops': (),
                    'add_target': pair_add_target,
                    'mm21_target': pair_mm21_target,
                    'target_op': '',
                }
            continue

        op = infer_operation(a, b, c)
        if op is None:
            continue
        pair_ops.setdefault(pair, set()).add(op)

    taxonomy = dict(special_pairs)
    for pair, ops in pair_ops.items():
        a, b = pair
        if ops == {OP_ADD, OP_MM21}:
            pair_type = PAIR_TYPE_DUAL_DISTINCT
            target_op = ''
        elif ops == {OP_ADD}:
            pair_type = PAIR_TYPE_SINGLE_ADD
            target_op = OP_ADD
        elif ops == {OP_MM21}:
            pair_type = PAIR_TYPE_SINGLE_MM21
            target_op = OP_MM21
        else:
            continue
        taxonomy[pair] = {
            'pair_type': pair_type,
            'present_ops': tuple(sorted(ops)),
            'add_target': add_target(a, b),
            'mm21_target': mm21_target(a, b),
            'target_op': target_op,
        }
    return taxonomy


def count_pair_types(pair_taxonomy):
    counts = {
        PAIR_TYPE_SINGLE_ADD: 0,
        PAIR_TYPE_SINGLE_MM21: 0,
        PAIR_TYPE_DUAL_DISTINCT: 0,
        PAIR_TYPE_SPECIAL_AMBIGUOUS: 0,
    }
    for info in pair_taxonomy.values():
        counts[info['pair_type']] = counts.get(info['pair_type'], 0) + 1
    return counts


def labels_to_triple(labels):
    return tuple(parse_label(x) for x in labels)


def labels_from_dataset_name(data_name):
    parts = data_name.split('-')
    if len(parts) < 3:
        raise ValueError(f"Cannot parse dataset item name as operation sample: {data_name}")
    a = int(parts[0])
    b = int(parts[1])
    op = parts[-1]
    if op == OP_ADD:
        c = add_target(a, b)
    elif op in {'mul', OP_MM21}:
        c = mm21_target(a, b)
    else:
        raise ValueError(f"Cannot infer operation from dataset item name: {data_name}")
    return a, b, c


def dataset_label_triple(dataset, index):
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        return dataset_label_triple(dataset.dataset, int(dataset.indices[index]))

    if hasattr(dataset, 'datasets') and hasattr(dataset, 'cumulative_sizes'):
        dataset_idx = bisect_right(dataset.cumulative_sizes, index)
        prev_size = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        return dataset_label_triple(dataset.datasets[dataset_idx], index - prev_size)

    if hasattr(dataset, 'f_list'):
        base_index = index % len(dataset.f_list)
        data_list = getattr(dataset, 'data_list', None)
        if data_list:
            return labels_to_triple(data_list[base_index][1])
        if hasattr(dataset, 'read_a_data_from_disk'):
            _, labels = dataset.read_a_data_from_disk(dataset.f_list[base_index], apply_transform=False)
            return labels_to_triple(labels)
        return labels_from_dataset_name(dataset.f_list[base_index])

    _, labels = dataset[index]
    return labels_to_triple(labels)


def iter_dataset_label_triples(dataset):
    for index in range(len(dataset)):
        yield dataset_label_triple(dataset, index)


def critical_pairs_from_taxonomy(pair_taxonomy):
    return {
        pair: {
            'add_target': pair_info['add_target'],
            'mm21_target': pair_info['mm21_target'],
        }
        for pair, pair_info in pair_taxonomy.items()
        if pair_info['pair_type'] == PAIR_TYPE_DUAL_DISTINCT
    }


def pair_risk_pairs_from_taxonomy(pair_taxonomy):
    return {
        pair: pair_info
        for pair, pair_info in pair_taxonomy.items()
        if pair_info['pair_type'] in PAIR_RISK_TYPES
    }


def query_winner(q1_loss, q2_loss, tie_eps=1e-8):
    if q1_loss < q2_loss - tie_eps:
        return 'q1'
    if q2_loss < q1_loss - tie_eps:
        return 'q2'
    return 'tie'


def values_to_list(values):
    if hasattr(values, 'detach'):
        return values.detach().cpu().tolist()
    return list(values)


def init_critical_pair_interval_stats(critical_pairs):
    return {
        pair: {
            'total_epochs': 0,
            'q1_exclusive': 0,
            'q2_exclusive': 0,
            'split': 0,
            'tie_or_missing': 0,
        }
        for pair in critical_pairs
    }


def new_critical_pair_epoch_stats(critical_pairs):
    return {
        pair: {
            OP_ADD: {'q1': 0, 'q2': 0, 'tie': 0},
            OP_MM21: {'q1': 0, 'q2': 0, 'tie': 0},
        }
        for pair in critical_pairs
    }


def update_critical_pair_epoch_stats(
        critical_pairs,
        epoch_stats,
        per_loss_q1,
        per_loss_q2,
        label_a,
        label_b,
        label_c,
        tie_eps=1e-8):
    if not epoch_stats:
        return
    q1_losses = values_to_list(per_loss_q1)
    q2_losses = values_to_list(per_loss_q2)
    for a, b, c, q1_loss, q2_loss in zip(label_a, label_b, label_c, q1_losses, q2_losses):
        pair = (a, b)
        pair_info = critical_pairs.get(pair)
        if pair_info is None:
            continue
        if c == pair_info['add_target']:
            op = OP_ADD
        elif c == pair_info['mm21_target']:
            op = OP_MM21
        else:
            continue
        winner = query_winner(q1_loss, q2_loss, tie_eps)
        epoch_stats[pair][op][winner] += 1


def majority_query_winner(winner_counts):
    if sum(winner_counts.values()) == 0:
        return None
    q1_count = winner_counts['q1']
    q2_count = winner_counts['q2']
    tie_count = winner_counts['tie']
    if q1_count > q2_count and q1_count > tie_count:
        return 'q1'
    if q2_count > q1_count and q2_count > tie_count:
        return 'q2'
    return 'tie'


def accumulate_critical_pair_epoch_stats(critical_pairs, interval_stats, epoch_stats):
    for pair, pair_epoch_stats in epoch_stats.items():
        stats = interval_stats[pair]
        stats['total_epochs'] += 1
        add_winner = majority_query_winner(pair_epoch_stats[OP_ADD])
        mm21_winner = majority_query_winner(pair_epoch_stats[OP_MM21])
        if add_winner is None or mm21_winner is None or add_winner == 'tie' or mm21_winner == 'tie':
            stats['tie_or_missing'] += 1
        elif add_winner == 'q1' and mm21_winner == 'q1':
            stats['q1_exclusive'] += 1
        elif add_winner == 'q2' and mm21_winner == 'q2':
            stats['q2_exclusive'] += 1
        else:
            stats['split'] += 1


def critical_pair_rows(epoch, critical_pairs, interval_stats):
    rows = []
    for pair in sorted(critical_pairs):
        stats = interval_stats[pair]
        total_epochs = stats['total_epochs']
        if total_epochs == 0:
            continue
        pair_info = critical_pairs[pair]

        def rate(key):
            return stats[key] / total_epochs

        rows.append([
            epoch,
            pair[0],
            pair[1],
            pair_info['add_target'],
            pair_info['mm21_target'],
            total_epochs,
            round(rate('q1_exclusive'), 6),
            round(rate('q2_exclusive'), 6),
            round(rate('split'), 6),
            round(rate('tie_or_missing'), 6),
            stats['q1_exclusive'],
            stats['q2_exclusive'],
            stats['split'],
            stats['tie_or_missing'],
        ])
    return rows


def init_pair_risk_interval_stats(pair_risk_pairs):
    return {
        pair: {
            'total_epochs': 0,
            'q1_only': 0,
            'q2_only': 0,
            'mixed': 0,
            'tie_or_missing': 0,
            'q1_exclusive': 0,
            'q2_exclusive': 0,
            'split': 0,
            'mixed_or_missing': 0,
        }
        for pair in pair_risk_pairs
    }


def new_pair_epoch_stats(pair_risk_pairs):
    return {
        pair: {
            op: {'q1': 0, 'q2': 0, 'tie': 0}
            for op in pair_info['present_ops']
        }
        for pair, pair_info in pair_risk_pairs.items()
    }


def update_pair_epoch_stats(
        pair_risk_pairs,
        epoch_stats,
        per_loss_q1,
        per_loss_q2,
        label_a,
        label_b,
        label_c,
        tie_eps=1e-8):
    if not epoch_stats:
        return
    q1_losses = values_to_list(per_loss_q1)
    q2_losses = values_to_list(per_loss_q2)
    for a, b, c, q1_loss, q2_loss in zip(label_a, label_b, label_c, q1_losses, q2_losses):
        pair = (a, b)
        pair_info = pair_risk_pairs.get(pair)
        if pair_info is None:
            continue
        op = infer_operation(a, b, c)
        if op not in epoch_stats[pair]:
            continue
        winner = query_winner(q1_loss, q2_loss, tie_eps)
        epoch_stats[pair][op][winner] += 1


def classify_single_epoch(winner_counts):
    if not winner_counts or winner_counts.get('tie', 0) > 0:
        return 'tie_or_missing'
    q1_count = winner_counts.get('q1', 0)
    q2_count = winner_counts.get('q2', 0)
    if q1_count > 0 and q2_count > 0:
        return 'mixed'
    if q1_count > 0:
        return 'q1_only'
    if q2_count > 0:
        return 'q2_only'
    return 'tie_or_missing'


def classify_dual_op_epoch(winner_counts):
    if not winner_counts or winner_counts.get('tie', 0) > 0:
        return 'mixed_or_missing'
    q1_count = winner_counts.get('q1', 0)
    q2_count = winner_counts.get('q2', 0)
    if q1_count > 0 and q2_count > 0:
        return 'mixed_or_missing'
    if q1_count > 0:
        return 'q1'
    if q2_count > 0:
        return 'q2'
    return 'mixed_or_missing'


def accumulate_pair_risk_epoch_stats(pair_risk_pairs, interval_stats, epoch_stats):
    for pair, pair_epoch_stats in epoch_stats.items():
        pair_info = pair_risk_pairs[pair]
        stats = interval_stats[pair]
        stats['total_epochs'] += 1
        if pair_info['pair_type'] in {PAIR_TYPE_SINGLE_ADD, PAIR_TYPE_SINGLE_MM21}:
            state = classify_single_epoch(pair_epoch_stats.get(pair_info['target_op']))
            stats[state] += 1
            continue

        add_state = classify_dual_op_epoch(pair_epoch_stats.get(OP_ADD))
        mm21_state = classify_dual_op_epoch(pair_epoch_stats.get(OP_MM21))
        if add_state == 'mixed_or_missing' or mm21_state == 'mixed_or_missing':
            stats['mixed_or_missing'] += 1
        elif add_state == 'q1' and mm21_state == 'q1':
            stats['q1_exclusive'] += 1
        elif add_state == 'q2' and mm21_state == 'q2':
            stats['q2_exclusive'] += 1
        else:
            stats['split'] += 1


def dominant_query(q1_rate, q2_rate):
    if q1_rate > q2_rate:
        return 'q1'
    if q2_rate > q1_rate:
        return 'q2'
    return 'tie'


def rate_from_stats(stats, key):
    total = stats['total_epochs']
    return 0.0 if total == 0 else stats[key] / total


def pair_risk_rows(epoch, pair_risk_pairs, interval_stats):
    rows = []
    for pair in sorted(pair_risk_pairs):
        pair_info = pair_risk_pairs[pair]
        stats = interval_stats[pair]
        total_epochs = stats['total_epochs']
        if total_epochs == 0:
            continue

        q1_only_rate = rate_from_stats(stats, 'q1_only')
        q2_only_rate = rate_from_stats(stats, 'q2_only')
        mixed_rate = rate_from_stats(stats, 'mixed')
        tie_or_missing_rate = rate_from_stats(stats, 'tie_or_missing')
        q1_exclusive_rate = rate_from_stats(stats, 'q1_exclusive')
        q2_exclusive_rate = rate_from_stats(stats, 'q2_exclusive')
        split_rate = rate_from_stats(stats, 'split')
        mixed_or_missing_rate = rate_from_stats(stats, 'mixed_or_missing')

        if pair_info['pair_type'] in {PAIR_TYPE_SINGLE_ADD, PAIR_TYPE_SINGLE_MM21}:
            competition_rate = mixed_rate + min(q1_only_rate, q2_only_rate)
            dominance_rate = 0.0
            risk_score = competition_rate
            query_name = dominant_query(q1_only_rate, q2_only_rate)
        else:
            competition_rate = 0.0
            dominance_rate = q1_exclusive_rate + q2_exclusive_rate
            risk_score = dominance_rate
            query_name = dominant_query(q1_exclusive_rate, q2_exclusive_rate)

        rows.append([
            epoch,
            pair[0],
            pair[1],
            pair_info['pair_type'],
            '|'.join(pair_info['present_ops']),
            pair_info['add_target'],
            pair_info['mm21_target'],
            total_epochs,
            round(risk_score, 6),
            pair_info['target_op'],
            round(q1_only_rate, 6),
            round(q2_only_rate, 6),
            round(mixed_rate, 6),
            round(tie_or_missing_rate, 6),
            query_name,
            round(competition_rate, 6),
            round(q1_exclusive_rate, 6),
            round(q2_exclusive_rate, 6),
            round(split_rate, 6),
            round(mixed_or_missing_rate, 6),
            round(dominance_rate, 6),
            stats['q1_only'],
            stats['q2_only'],
            stats['mixed'],
            stats['tie_or_missing'],
            stats['q1_exclusive'],
            stats['q2_exclusive'],
            stats['split'],
            stats['mixed_or_missing'],
        ])
    return rows


class PairDiagnosticsRecorder:
    def __init__(
            self,
            train_dataset,
            critical_pair_record_path='CriticalPairStats_record.csv',
            pair_risk_record_path='PairRiskStats_record.csv'):
        self.critical_pair_record_path = critical_pair_record_path
        self.pair_risk_record_path = pair_risk_record_path
        self.pair_taxonomy = build_pair_taxonomy(iter_dataset_label_triples(train_dataset))
        self.pair_type_counts = count_pair_types(self.pair_taxonomy)
        self.critical_pairs = critical_pairs_from_taxonomy(self.pair_taxonomy)
        self.pair_risk_pairs = pair_risk_pairs_from_taxonomy(self.pair_taxonomy)
        self.critical_pair_interval_stats = init_critical_pair_interval_stats(self.critical_pairs)
        self.pair_risk_interval_stats = init_pair_risk_interval_stats(self.pair_risk_pairs)
        self._ensure_header(self.critical_pair_record_path, CRITICAL_PAIR_RECORD_COLUMNS)
        self._ensure_header(self.pair_risk_record_path, PAIR_RISK_RECORD_COLUMNS)

    @staticmethod
    def _ensure_header(record_path, columns):
        if os.path.exists(record_path) and os.path.getsize(record_path) > 0:
            return
        with open(record_path, 'w', encoding='utf-8') as f:
            f.write(','.join(columns) + '\n')

    @staticmethod
    def _append_rows(record_path, rows):
        if not rows:
            return
        with open(record_path, 'a', encoding='utf-8') as f:
            for row in rows:
                f.write(','.join(str(item) for item in row) + '\n')

    def summary_text(self):
        return (
            "Train pair taxonomy: "
            f"single_add={self.pair_type_counts[PAIR_TYPE_SINGLE_ADD]}, "
            f"single_mm21={self.pair_type_counts[PAIR_TYPE_SINGLE_MM21]}, "
            f"dual_distinct={self.pair_type_counts[PAIR_TYPE_DUAL_DISTINCT]}, "
            f"special_ambiguous={self.pair_type_counts[PAIR_TYPE_SPECIAL_AMBIGUOUS]}"
        )

    def new_epoch_stats(self):
        return {
            'critical': (
                new_critical_pair_epoch_stats(self.critical_pairs)
                if self.critical_pairs
                else None
            ),
            'pair_risk': (
                new_pair_epoch_stats(self.pair_risk_pairs)
                if self.pair_risk_pairs
                else None
            ),
        }

    def update_epoch_stats(self, epoch_stats, per_loss_q1, per_loss_q2, label_a, label_b, label_c):
        if epoch_stats is None:
            return
        if epoch_stats['critical'] is not None:
            update_critical_pair_epoch_stats(
                self.critical_pairs,
                epoch_stats['critical'],
                per_loss_q1,
                per_loss_q2,
                label_a,
                label_b,
                label_c,
            )
        if epoch_stats['pair_risk'] is not None:
            update_pair_epoch_stats(
                self.pair_risk_pairs,
                epoch_stats['pair_risk'],
                per_loss_q1,
                per_loss_q2,
                label_a,
                label_b,
                label_c,
            )

    def accumulate_epoch_stats(self, epoch_stats):
        if epoch_stats is None:
            return
        if epoch_stats['critical'] is not None:
            accumulate_critical_pair_epoch_stats(
                self.critical_pairs,
                self.critical_pair_interval_stats,
                epoch_stats['critical'],
            )
        if epoch_stats['pair_risk'] is not None:
            accumulate_pair_risk_epoch_stats(
                self.pair_risk_pairs,
                self.pair_risk_interval_stats,
                epoch_stats['pair_risk'],
            )

    def record_interval(self, epoch):
        critical_rows = critical_pair_rows(epoch, self.critical_pairs, self.critical_pair_interval_stats)
        pair_risk_rows_ = pair_risk_rows(epoch, self.pair_risk_pairs, self.pair_risk_interval_stats)
        self._ensure_header(self.critical_pair_record_path, CRITICAL_PAIR_RECORD_COLUMNS)
        self._ensure_header(self.pair_risk_record_path, PAIR_RISK_RECORD_COLUMNS)
        self._append_rows(self.critical_pair_record_path, critical_rows)
        self._append_rows(self.pair_risk_record_path, pair_risk_rows_)
        self.critical_pair_interval_stats = init_critical_pair_interval_stats(self.critical_pairs)
        self.pair_risk_interval_stats = init_pair_risk_interval_stats(self.pair_risk_pairs)
