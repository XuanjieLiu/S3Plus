import numpy as np
from scipy.optimize import linear_sum_assignment

# profit = np.array([
#     [5, 7, 9],
#     [6, 4, 3],
#     [2, 2, 2],
#     [8, 5, 6],
# ])
# profit = np.array([
#     [0, 6, 0, 0],
#     [0, 0, 6, 0],
#     [0, 6, 0, 0],
# ])
#
# C = profit.max() - profit
# row_ind, col_ind = linear_sum_assignment(C)
#
# total_profit = profit[row_ind, col_ind].sum()
# min_total_cost = C[row_ind, col_ind].sum()  # 等价于 3 * profit.max() - total_profit
# max_profit = profit.sum()
#
# print("匹配对：", list(zip(row_ind, col_ind)))
# print("总收益：", total_profit)
# print("总收益率", total_profit / max_profit)
# print("最小成本（等价于最大收益的转换损失）：", min_total_cost)

def solve_label_emb_one2one_matching(num_emb_idx, num_labels):
    """
    Solve the one-to-one matching problem for label-embedding pairs using the Hungarian algorithm.
    :param num_emb_idx: List of numerical indices for embeddings.
    :param num_labels: List of numerical labels corresponding to the embeddings.
    :return: A tuple containing the mapping of labels to embeddings and the km_score.
    1. The mapping is a list of tuples where each tuple contains a label and its corresponding embedding index.
    2. The km_score is the ratio of total profit to maximum profit.
    """
    # Assemble a label embedding matrix based on the provided embedding indices and labels.
    all_embs = list(set(num_emb_idx))
    emb_idx2col_idx = {emb: i for i, emb in enumerate(all_embs)}
    all_labels = list(set(num_labels))
    label_idx2row_idx = {label: i for i, label in enumerate(all_labels)}
    label_emb_matrix = np.zeros((len(all_labels), len(all_embs)))
    for label, emb in zip(num_labels, num_emb_idx):
        label_midx = label_idx2row_idx[label]
        emb_midx = emb_idx2col_idx[emb]
        label_emb_matrix[label_midx, emb_midx] += 1
    print("Label-Embedding Matrix:\n", label_emb_matrix)

    # Solve the assignment problem using the Hungarian algorithm.
    profit = label_emb_matrix
    C = profit.max() - profit
    row_ind, col_ind = linear_sum_assignment(C)

    # Return the mapping from matrix indices to embedding and label indices, remove profit of 0.
    non_zero_label_emb_pairs = []
    for i, j in zip(row_ind, col_ind):
        if profit[i, j] > 0:
            non_zero_label_emb_pairs.append((i, j))
    col_idx2emb_idx = {i: emb for i, emb in enumerate(all_embs)}
    row_idx2label_idx = {i: label for i, label in enumerate(all_labels)}
    mapping = [(row_idx2label_idx[i], col_idx2emb_idx[j]) for i, j in non_zero_label_emb_pairs]

    # Calculate km_score, the ratio of total profit to max profit.
    total_profit = profit[row_ind, col_ind].sum()
    max_profit = profit.sum()
    km_score = total_profit / max_profit

    return mapping, km_score

if __name__ == "__main__":
    num_emb_idx = [0, 1, 2, 0, 1, 2, 3, 3,]
    num_labels = [1, 2, 3, 1, 2, 3, 1, 1,]
    mapping, km_score = solve_label_emb_one2one_matching(num_emb_idx, num_labels)
    print("Label-Embedding Mapping:", mapping)
    print("KM Score:", km_score)
