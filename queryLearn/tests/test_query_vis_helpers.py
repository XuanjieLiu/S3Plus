import sys
import unittest
from pathlib import Path


S3PLUS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S3PLUS_ROOT))

from queryLearn.query_vis import (  # noqa: E402
    operation_cell_draw_items,
    operation_cell_text,
    operation_cells,
    operation_table_grid,
)


class QueryVisHelpersTest(unittest.TestCase):
    def test_duplicate_pair_uses_first_observation_only(self):
        cells = operation_cells(
            label_a=[1, 1],
            label_b=[2, 2],
            label_c=[3, 3],
            q1_correct=[True, False],
            q2_correct=[False, True],
            q1_target_dist=[0.1, 0.9],
            q2_target_dist=[0.4, 0.2],
            operation="add",
        )

        self.assertEqual(cells[(1, 2)]["q1"], True)
        self.assertEqual(cells[(1, 2)]["q2"], False)
        self.assertEqual(cells[(1, 2)]["winner"], "q1")

    def test_background_winner_classes(self):
        cells = operation_cells(
            label_a=[1, 2, 3],
            label_b=[2, 3, 4],
            label_c=[3, 5, 7],
            q1_correct=[False, False, False],
            q2_correct=[False, False, False],
            q1_target_dist=[0.1, 0.7, 0.5],
            q2_target_dist=[0.3, 0.2, 0.5],
            operation="add",
        )

        _, bg_grid = operation_table_grid([1, 2, 3], [2, 3, 4], cells)

        self.assertEqual(bg_grid[0][0], 2)  # q1 closer
        self.assertEqual(bg_grid[1][1], 3)  # q2 closer
        self.assertEqual(bg_grid[2][2], 1)  # tie
        self.assertEqual(bg_grid[0][1], 0)  # missing cell

    def test_text_states_and_draw_items(self):
        q1_cell = {"in_set": True, "q1": True, "q2": False, "winner": "q1"}
        q2_cell = {"in_set": True, "q1": False, "q2": True, "winner": "q2"}
        both_cell = {"in_set": True, "q1": True, "q2": True, "winner": "tie"}
        miss_cell = {"in_set": True, "q1": False, "q2": False, "winner": "q1"}

        self.assertEqual(operation_cell_text(q1_cell), "1")
        self.assertEqual(operation_cell_text(q2_cell), "2")
        self.assertEqual(operation_cell_text(both_cell), "12")
        self.assertEqual(operation_cell_text(miss_cell), "×")

        both_items = operation_cell_draw_items(both_cell)
        self.assertEqual([item["text"] for item in both_items], ["1", "2"])
        self.assertNotEqual(both_items[0]["color"], both_items[1]["color"])

        miss_items = operation_cell_draw_items(miss_cell)
        self.assertEqual(miss_items[0]["text"], "×")
        self.assertEqual(miss_items[0]["fontweight"], "bold")

    def test_special_pair_is_marked_and_belongs_to_both_operations(self):
        add_cells = operation_cells(
            label_a=[3],
            label_b=[12],
            label_c=[15],
            q1_correct=[True],
            q2_correct=[False],
            q1_target_dist=[0.1],
            q2_target_dist=[0.2],
            operation="add",
        )
        mm21_cells = operation_cells(
            label_a=[3],
            label_b=[12],
            label_c=[15],
            q1_correct=[True],
            q2_correct=[False],
            q1_target_dist=[0.1],
            q2_target_dist=[0.2],
            operation="mm21",
        )

        self.assertTrue(add_cells[(3, 12)]["special"])
        self.assertTrue(mm21_cells[(3, 12)]["special"])


if __name__ == "__main__":
    unittest.main()
