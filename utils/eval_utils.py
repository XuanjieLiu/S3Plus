import os
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_curve


def mscatter_3d(x, y, z, ax=None, m=None, **kw):
    """
    Adapted from https://stackoverflow.com/questions/52303660/iterating-markers-in-plots/52303895#52303895
    3d scatter plot with different markers
    """
    import matplotlib.markers as mmarkers

    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, z, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def precision_recall_at_k(emb, labels, k_list=[1, 2, 5, 10, 20, 50, 100, 200, 500]):
    """
    Find the nearest k samples to the input sample
    Compute the precision & recall at k metrics using a list of positive samples
    Also compute the F1 score
    emb: the embeddings of the samples (n_samples, emb_dim)
    labels: the labels of the samples (n_samples,)
    k_list: the list of k values

    Return: the precision, recall, F1 score at k
    """
    n_samples = emb.shape[0]

    all_precisions = []
    all_recalls = []
    all_f1s = []

    for i in range(n_samples):
        # find all the positive samples
        pos_indices = []
        for j in range(n_samples):
            if labels[j] == labels[i] and j != i:
                pos_indices.append(j)

        # compute distances from the input to all the samples
        all_distances = {}
        for j in range(n_samples):
            d = np.linalg.norm(emb[i] - emb[j])
            all_distances[j] = d
        all_distances = sorted(all_distances.items(), key=lambda x: x[1])

        # compute the precision & recall at k metrics
        precision_list = []
        recall_list = []
        f1_list = []
        for k in k_list:
            # find the nearest k samples
            nearest_k = all_distances[:k]
            nearest_k = [x[0] for x in nearest_k]

            # compute the precision and recall at k metric
            precision = 0
            recall = 0
            for j in nearest_k:
                if j in pos_indices:
                    precision += 1
                    recall += 1
            precision /= k
            recall /= len(pos_indices)
            precision_list.append(precision)
            recall_list.append(recall)

            # compute the f1 score
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_list.append(f1)

        precision_list = np.array(precision_list)
        recall_list = np.array(recall_list)
        f1_list = np.array(f1_list)

        all_precisions.append(precision_list)
        all_recalls.append(recall_list)
        all_f1s.append(f1_list)

    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    all_f1s = np.array(all_f1s)

    all_precisions = np.mean(all_precisions, axis=0)
    all_recalls = np.mean(all_recalls, axis=0)
    all_f1s = np.mean(all_f1s, axis=0)

    return all_precisions, all_recalls, all_f1s


def area_under_prcurve(r_list, p_list):
    """
    Compute the area under the precision-recall curve
    r_list: the list of recall values (should have been sorted in ascending order)
    p_list: the list of precision values (should have been sorted in ascending order)
    """
    area = 0
    for i in range(1, len(r_list)):
        area += (r_list[i] - r_list[i - 1]) * p_list[i]

    return area


def get_confusion_mtx(n_pred, n_labels, preds, labels):
    """
    n_pred: the number of predicted classes (might be more than the true number of classes)
    n_labels: the number of true classes
    preds: the predicted labels
    labels: the true labels
    returns the confusion matrix and the permutation of the predicted classes
    """
    confusion_mtx = np.zeros((n_pred, n_labels))
    for i in range(len(preds)):
        label = int(labels[i])
        pred = int(preds[i])
        confusion_mtx[pred, label] += 1
    confusion_mtx = confusion_mtx / (confusion_mtx.sum(axis=1, keepdims=True) + 1e-7)
    assignments = np.argmax(confusion_mtx, axis=1)
    perm = np.argsort(assignments)
    confusion_mtx = confusion_mtx[perm]

    return confusion_mtx, perm


def plot_confusion_mtx(mtx, title="Confusion Matrix"):
    confusion, ax = plt.subplots()
    ax = sns.heatmap(mtx, cmap="Purples", vmin=0, vmax=1, cbar=False)
    ax.set_xlabel("Content Index", fontsize=16)
    ax.set_ylabel("Codebook Index", fontsize=16)
    ax.set_title(title, fontsize=20)

    confusion.add_axes(ax)
    confusion.canvas.draw()
    gcf = plt.gcf()
    plt.close()

    return gcf


def confusion_mtx_acc(mtx):
    """
    Overall accuracy of a confusion matrix
    mtx: [n_predicted_classes, n_labels]
    """
    acc = 0
    for i in range(mtx.shape[0]):
        max_entry = mtx[i, :].max()
        acc += max_entry
    acc /= mtx.sum()

    return acc


def confusion_mtx_std(confusion_matrices):
    """
    For a list of confusion matrices (of the same shape, of course), compute the consensus score. They have to have the same permutation.
    horizonally: the real labels
    vertically: the codebook atoms
    """
    if isinstance(confusion_matrices, list):
        confusion_matrices = np.array(confusion_matrices)
    std = np.std(confusion_matrices, axis=0)
    std = np.mean(std)

    return std


def pairwise_d(x):
    """
    x is a tensor of shape (n, d)
    """
    n, d = x.shape
    x1 = x.unsqueeze(0).expand(n, n, d)
    x2 = x.unsqueeze(1).expand(n, n, d)
    stack = torch.stack([x1, x2], dim=0)
    pairwise_d = torch.norm(stack[0] - stack[1], dim=-1)

    return pairwise_d


def denormalize_img(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    x: the normalized data (n_samples, n_channels, height, width), ndarray
    mean: the mean used for normalization
    std: the std used for normalization
    """

    std = np.array(std)
    mean = np.array(mean)
    std = std.reshape((1, -1, 1, 1))
    mean = mean.reshape((1, -1, 1, 1))

    if len(x.shape) == 3:
        x = x.reshape((1,) + x.shape)
        x = x * std + mean
        x = x.reshape(x.shape[1:])
    else:
        x = x * std + mean

    return x


def compute_eer(embeddings, labels, metric="l2"):
    """
    embeddings: the embeddings of the samples
    labels: the labels of the samples
    first compute pairwise distances between embeddings
    then compute the EER
    """
    if metric == "l2":
        d_fn = lambda x, y: np.linalg.norm(x - y)
    else:
        raise NotImplementedError

    scores, answers = [], []
    for i in tqdm(range(len(embeddings))):
        for j in range(i + 1, len(embeddings)):
            scores.append(d_fn(embeddings[i], embeddings[j]))
            answers.append(labels[i] == labels[j])

    scores = np.array(scores)
    scores = 1 - (scores - scores.min()) / (
        scores.max() - scores.min()
    )  # normalize to [0, 1] and invert
    answers = np.array(answers)

    fpr, tpr, thresholds = roc_curve(answers, scores)
    eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]

    return eer


def mean_and_std(*x):
    x = [np.array(i) for i in x]
    mean = np.mean(x, axis=0) * 100
    std = np.std(x, axis=0) * 100

    # round to 4 decimal places
    mean = np.round(mean, 4)
    std = np.round(std, 4)

    return mean, std


def plot_spectrogram(array, label_1, label_2, is_log_scale=True):
    """
    Plot a spectrogram
    array: the spectrogram to plot (n_freq_bins, n_time_bins)
    log_scale: whether to apply log scale
    """
    if is_log_scale:
        # array = np.exp(array)
        pass
    array = array.transpose()
    plt.figure(figsize=(12, 4))
    plt.imshow(array, aspect="auto", origin="lower", cmap="inferno")
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.yscale("log") if is_log_scale else plt.yscale("linear")
    # plt.ylim(0, array.shape[1])
    plt.title("Spectrogram" + " c" + str(label_1) + " s" + str(label_2))
    gcf = plt.gcf()
    plt.close()
    return gcf


def plot_waveform_and_boundaries(
    audio, boundaries_label, boundaries_predicted, sr=16000
):
    """
    audio: the audio signal
    boundaries: the predicted boundaries. should be a binary array, with the same length as the audio
    sr: the sampling rate of the audio

    return the current figure, ready to be logged to wandb
    """
    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    for i in range(len(boundaries_label)):
        if boundaries_label[i] == 1:
            plt.axvline(x=i, color="green", linestyle="--")
    for i in range(len(boundaries_predicted)):
        if boundaries_predicted[i] == 1:
            plt.axvline(x=i, color="red", linestyle="--")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Waveform and Boundaries")
    gcf = plt.gcf()
    plt.close()

    return gcf


def plot_image(img):
    """
    img: the image to plot, (3, height, width)
    img should be between 0 and 255, in uint8 format
    """
    plt.imshow(img.transpose(1, 2, 0))
    plt.axis("off")
    gcf = plt.gcf()
    plt.close()

    return gcf


def plot_slot_attention_masks(masks):
    """
    mask: a 3D tensor of shape (n_slots, width, height) or a 4D tensor of shape (n_slots, 1, width, height)
    plot them in a row
    """
    if len(masks.shape) == 3:
        masks = masks.unsqueeze(1)
    n_slots = masks.shape[0]
    fig, axs = plt.subplots(1, n_slots, figsize=(n_slots * 2, 2))
    for i in range(n_slots):
        axs[i].imshow(masks[i][0], cmap="gray")
        axs[i].axis("off")
    plt.tight_layout()
    gcf = plt.gcf()
    plt.close()

    return gcf
