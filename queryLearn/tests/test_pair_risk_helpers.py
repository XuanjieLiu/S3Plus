import sys
import tempfile
import unittest
from pathlib import Path


S3PLUS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S3PLUS_ROOT))

from queryLearn.pair_diagnostics import (  # noqa: E402
    CRITICAL_PAIR_RECORD_COLUMNS,
    PAIR_RISK_RECORD_COLUMNS,
    PAIR_RISK_TYPES,
    PAIR_TYPE_DUAL_DISTINCT,
    PAIR_TYPE_SINGLE_ADD,
    PAIR_TYPE_SINGLE_MM21,
    PAIR_TYPE_SPECIAL_AMBIGUOUS,
    accumulate_pair_risk_epoch_stats,
    build_pair_taxonomy,
    init_pair_risk_interval_stats,
    iter_dataset_label_triples,
    new_pair_epoch_stats,
    pair_risk_rows,
    update_pair_epoch_stats,
)
from queryLearn.experiment_analysis import generate_analysis  # noqa: E402


def risk_pairs_from_taxonomy(taxonomy):
    return {
        pair: info
        for pair, info in taxonomy.items()
        if info["pair_type"] in PAIR_RISK_TYPES
    }


def row_dict(row):
    return dict(zip(PAIR_RISK_RECORD_COLUMNS, row))


def stats_row(label_triples, epochs):
    taxonomy = build_pair_taxonomy(label_triples)
    pair_risk_pairs = risk_pairs_from_taxonomy(taxonomy)
    interval_stats = init_pair_risk_interval_stats(pair_risk_pairs)
    for labels, q1_losses, q2_losses in epochs:
        epoch_stats = new_pair_epoch_stats(pair_risk_pairs)
        label_a, label_b, label_c = zip(*labels)
        update_pair_epoch_stats(
            pair_risk_pairs,
            epoch_stats,
            q1_losses,
            q2_losses,
            label_a,
            label_b,
            label_c,
        )
        accumulate_pair_risk_epoch_stats(pair_risk_pairs, interval_stats, epoch_stats)
    rows = pair_risk_rows(0, pair_risk_pairs, interval_stats)
    return row_dict(rows[0])


class FakeLeafDataset:
    def __init__(self, label_sets):
        self.f_list = [f"sample-{idx}" for idx in range(len(label_sets))]
        self.data_list = [(None, labels) for labels in label_sets]

    def __len__(self):
        return len(self.f_list)


class FakeConcatDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        total = 0
        self.cumulative_sizes = []
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]


class FakeSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)


class PairRiskHelpersTest(unittest.TestCase):
    def test_single_add_stable_q1_has_zero_risk(self):
        row = stats_row(
            [(1, 2, 3)],
            [([(1, 2, 3)], [0.1], [0.2])],
        )
        self.assertEqual(row["pair_type"], PAIR_TYPE_SINGLE_ADD)
        self.assertEqual(row["risk_score"], 0.0)
        self.assertEqual(row["q1_only_rate"], 1.0)

    def test_single_add_alternating_queries_has_competition_risk(self):
        row = stats_row(
            [(1, 2, 3)],
            [
                ([(1, 2, 3)], [0.1], [0.2]),
                ([(1, 2, 3)], [0.3], [0.1]),
            ],
        )
        self.assertEqual(row["pair_type"], PAIR_TYPE_SINGLE_ADD)
        self.assertEqual(row["risk_score"], 0.5)
        self.assertEqual(row["competition_rate"], 0.5)

    def test_single_mm21_same_epoch_mixed_winners_has_full_risk(self):
        row = stats_row(
            [(2, 3, 6)],
            [([(2, 3, 6), (2, 3, 6)], [0.1, 0.3], [0.2, 0.1])],
        )
        self.assertEqual(row["pair_type"], PAIR_TYPE_SINGLE_MM21)
        self.assertEqual(row["mixed_rate"], 1.0)
        self.assertEqual(row["risk_score"], 1.0)

    def test_dual_pair_split_has_zero_risk(self):
        row = stats_row(
            [(1, 2, 3), (1, 2, 2)],
            [([(1, 2, 3), (1, 2, 2)], [0.1, 0.3], [0.2, 0.1])],
        )
        self.assertEqual(row["pair_type"], PAIR_TYPE_DUAL_DISTINCT)
        self.assertEqual(row["split_rate"], 1.0)
        self.assertEqual(row["risk_score"], 0.0)

    def test_dual_pair_same_query_has_full_risk(self):
        row = stats_row(
            [(1, 2, 3), (1, 2, 2)],
            [([(1, 2, 3), (1, 2, 2)], [0.1, 0.1], [0.2, 0.2])],
        )
        self.assertEqual(row["pair_type"], PAIR_TYPE_DUAL_DISTINCT)
        self.assertEqual(row["q1_exclusive_rate"], 1.0)
        self.assertEqual(row["risk_score"], 1.0)

    def test_special_pair_is_not_a_risk_row(self):
        taxonomy = build_pair_taxonomy([(0, 0, 0)])
        self.assertEqual(taxonomy[(0, 0)]["pair_type"], PAIR_TYPE_SPECIAL_AMBIGUOUS)
        self.assertEqual(risk_pairs_from_taxonomy(taxonomy), {})

    def test_fake_train_dataset_taxonomy_uses_actual_subset_indices(self):
        add_leaf = FakeLeafDataset([
            ["a-1.png", "b-2.png", "c-3.png"],
            ["a-4.png", "b-5.png", "c-9.png"],
        ])
        mm21_leaf = FakeLeafDataset([
            ["a-1.png", "b-2.png", "c-2.png"],
            ["a-2.png", "b-3.png", "c-6.png"],
            ["a-0.png", "b-0.png", "c-0.png"],
        ])
        concat = FakeConcatDataset([add_leaf, mm21_leaf])
        subset = FakeSubset(concat, [0, 2, 3, 4])

        taxonomy = build_pair_taxonomy(iter_dataset_label_triples(subset))

        self.assertEqual(taxonomy[(1, 2)]["pair_type"], PAIR_TYPE_DUAL_DISTINCT)
        self.assertEqual(taxonomy[(2, 3)]["pair_type"], PAIR_TYPE_SINGLE_MM21)
        self.assertEqual(taxonomy[(0, 0)]["pair_type"], PAIR_TYPE_SPECIAL_AMBIGUOUS)
        self.assertNotIn((4, 5), taxonomy)

    def test_analysis_reads_legacy_critical_pair_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_exps_dir = generate_analysis.EXPS_DIR
            generate_analysis.EXPS_DIR = Path(tmp_dir)
            try:
                csv_dir = Path(tmp_dir) / "exp" / "1"
                csv_dir.mkdir(parents=True)
                row = [
                    10,
                    1,
                    2,
                    3,
                    2,
                    5,
                    0.6,
                    0.0,
                    0.4,
                    0.0,
                    3,
                    0,
                    2,
                    0,
                ]
                (csv_dir / "CriticalPairStats_record.csv").write_text(
                    ",".join(CRITICAL_PAIR_RECORD_COLUMNS) + "\n"
                    + ",".join(str(item) for item in row) + "\n",
                    encoding="utf-8",
                )

                report = generate_analysis.build_critical_pair_report(
                    {"short": "legacy", "exp_name": "exp", "sub_exp_id": "1", "label": "Legacy"}
                )
            finally:
                generate_analysis.EXPS_DIR = old_exps_dir

        self.assertEqual(report["critical_pair_count"], 1)
        self.assertEqual(report["top_all"][0]["dominant"], "q1")
        self.assertEqual(report["top_all"][0]["dominant_rate"], 0.6)

    def test_analysis_reads_new_pair_risk_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_exps_dir = generate_analysis.EXPS_DIR
            generate_analysis.EXPS_DIR = Path(tmp_dir)
            try:
                csv_dir = Path(tmp_dir) / "exp" / "1"
                csv_dir.mkdir(parents=True)
                single_row = [
                    10, 1, 2, PAIR_TYPE_SINGLE_ADD, "add", 3, 2, 4, 0.25, "add",
                    0.75, 0.25, 0.0, 0.0, "q1", 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                    3, 1, 0, 0, 0, 0, 0, 0,
                ]
                dual_row = [
                    10, 3, 4, PAIR_TYPE_DUAL_DISTINCT, "add|mm21", 7, 12, 4, 0.5, "",
                    0.0, 0.0, 0.0, 0.0, "q2", 0.0, 0.0, 0.5, 0.5, 0.0, 0.5,
                    0, 0, 0, 0, 0, 2, 2, 0,
                ]
                (csv_dir / "PairRiskStats_record.csv").write_text(
                    ",".join(PAIR_RISK_RECORD_COLUMNS) + "\n"
                    + ",".join(str(item) for item in single_row) + "\n"
                    + ",".join(str(item) for item in dual_row) + "\n",
                    encoding="utf-8",
                )

                report = generate_analysis.build_pair_risk_report(
                    {"short": "risk", "exp_name": "exp", "sub_exp_id": "1", "label": "Risk"}
                )
            finally:
                generate_analysis.EXPS_DIR = old_exps_dir

        self.assertEqual(report["pair_count"], 2)
        self.assertEqual(report["single_pair_count"], 1)
        self.assertEqual(report["dual_pair_count"], 1)
        self.assertEqual(report["top_single_all"][0]["risk_score"], 0.25)
        self.assertEqual(report["top_dual_all"][0]["risk_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
