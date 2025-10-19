

import torch

def compute_batchwise_transition_matrix(indices, n_atoms, normalize=True):
    """
    indices: (B, T) long tensor
    n_atoms: number of discrete states
    normalize: whether to row-normalize
    returns: (n_atoms, n_atoms) averaged transition matrix
    """
    B, T = indices.shape
    transition = torch.zeros(n_atoms, n_atoms, device=indices.device)
    
    for b in range(B):
        seq = indices[b]
        src = seq[:-1]
        dst = seq[1:]
        
        # count transitions in this batch
        mat = torch.zeros(n_atoms, n_atoms, device=indices.device)
        mat.index_add_(0, src, torch.nn.functional.one_hot(dst, n_atoms).float())
        
        transition += mat

    # average across batches
    transition /= B

    if normalize:
        transition = transition / (transition.sum(dim=-1, keepdim=True) + 1e-8)


    return transition


if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟参数
    B, T, n_atoms = 4, 6, 5

    # 随机生成离散状态序列
    indices = torch.randint(0, n_atoms, (B, T))
    print("随机生成的 indices:")
    print(indices)

    # 计算转移矩阵
    A = compute_batchwise_transition_matrix(indices, n_atoms, normalize=True)

    print("\n平均转移矩阵 A:")
    print(A)

    # 验证每行是否归一化
    print("\n每行和（应接近 1）:")
    print(A.sum(dim=-1))

    # 可选：可视化（需要 matplotlib）
    try:
        import matplotlib.pyplot as plt
        plt.imshow(A.cpu(), cmap='viridis')
        plt.title("Averaged Transition Matrix")
        plt.colorbar(label='Transition Probability')
        plt.xlabel("To state")
        plt.ylabel("From state")
        plt.show()
    except ImportError:
        print("\n未安装 matplotlib，跳过可视化。")
