import json
from typing import List, Dict, Any

def find_best_epoch(file_path: str, metric_name: str, larger_is_better: bool) -> int:
    """
    从训练日志 txt 文件中找到指定指标的最佳 epoch。

    参数:
        file_path (str): 日志文件路径
        metric_name (str): 指标名称，例如 "accuracy"、"all_loss"
        larger_is_better (bool): True 表示指标越大越好，False 表示指标越小越好

    返回:
        int: 最佳指标对应的 epoch num
    """
    best_epoch = None
    best_value = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()
            # 按 '-' 拆开
            split_parts = line.split("-")
            # 第一个元素是 epoch 号
            epoch_num = int(split_parts[0])
            # 跳过 epoch 号为 0 的行（通常是初始状态，不具有代表性）
            if epoch_num == 0:
                continue
            # 其余部分拼回去，保证科学计数法里的 '-' 不会被拆散
            rest_line = "-".join(split_parts[1:])
            # 再按逗号分割
            parts = rest_line.split(",")

            # 提取所有指标
            metrics = {}
            for p in parts:
                if ":" in p:
                    key, value = p.split(":")
                    metrics[key] = float(value)

            # 如果该行包含目标指标
            if metric_name in metrics:
                value = metrics[metric_name]
                if best_value is None:
                    best_value = value
                    best_epoch = epoch_num
                else:
                    if (larger_is_better and value > best_value) or (not larger_is_better and value < best_value):
                        best_value = value
                        best_epoch = epoch_num

    if best_epoch is None:
        raise ValueError(f"日志中没有找到指标: {metric_name}")

    return best_epoch


def get_metrics_by_epoch(record_path: str, epoch_num: int, key_list: List[str]) -> List[float]:
    """
    在指定日志文件中查找某个 epoch 对应的多个指标值。

    参数:
        record_path (str): 日志文件路径
        epoch_num (int): 目标 epoch
        key_list (List[str]): 要查询的指标名称列表

    返回:
        List[Dict[str, Any]]: JSON 风格的结果，每个 key 对应一条记录
    """
    with open(record_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 先按 '-' 分割，取出 epoch
            split_parts = line.split("-")
            try:
                current_epoch = int(split_parts[0])
            except ValueError:
                continue

            if current_epoch != epoch_num:
                continue

            # 把后面拼回去，避免科学计数法被拆开
            rest_line = "-".join(split_parts[1:])
            parts = rest_line.split(",")

            # 解析成 dict
            metrics = {}
            for p in parts:
                if ":" in p:
                    key, value = p.split(":")
                    metrics[key] = float(value)

            # 按 key_list 组织结果
            result = []
            for key in key_list:
                if key in metrics:
                    result.append(metrics[key])
            return result

    # 如果没找到对应 epoch
    raise ValueError(f"日志文件 {record_path} 中未找到 epoch {epoch_num}")

# 使用示例
if __name__ == "__main__":
    # test find_best_epoch
    file_path = "D:\Projects\Gus Xia\S3Plus\LM_align\exp/2025.8.30\Train_record.txt"
    metric_name = "plus_loss"
    larger_is_better = False  # loss 越小越好，accuracy 越大越好
    best_epoch = find_best_epoch(file_path, metric_name, larger_is_better)
    print(f"{metric_name} 最佳的 epoch 是: {best_epoch}")

    # test get_metrics_by_epoch
    record_path = "D:\Projects\Gus Xia\S3Plus\LM_align\exp/2025.8.30\Train_record.txt"
    epoch_num = best_epoch
    key_list = ["all_loss", "accuracy", "collapse_loss"]

    result = get_metrics_by_epoch(record_path, epoch_num, key_list)
    print(result)
