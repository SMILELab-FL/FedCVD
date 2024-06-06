from copy import deepcopy

import pandas as pd
import torch
import sklearn.metrics as metrics
import numpy as np
import json
from scipy.ndimage import _ni_support, generate_binary_structure, binary_erosion, distance_transform_edt


def get_pred_label(pred_score: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        one = torch.ones_like(pred_score, dtype=torch.float32)
        zero = torch.zeros_like(pred_score, dtype=torch.float32)
        pred_label = torch.where(pred_score >= 0.5, one, zero)
    return pred_label


def transfer_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def calculate_accuracy_every_label(pred_label, true_label, normalize=False):
    pred_label = pred_label.flatten()
    true_label = true_label.flatten()
    return metrics.accuracy_score(true_label, pred_label, normalize=normalize)


"""
将标签作为整体计算准确率, 当所有标签分类正确时记为正确
"""


def calculate_accuracy(pred_label, true_label, normalize=False):
    return metrics.accuracy_score(true_label, pred_label, normalize=normalize)


"""
计算每一个标签的准确率
"""


def calculate_accuracy_per_label(pred_label, true_label, normalize=False):
    pred_label = pred_label.T
    true_label = true_label.T
    n_classes = pred_label.shape[0]
    per_accuracy = np.zeros((n_classes,), dtype=float)
    for i in range(n_classes):
        per_accuracy[i] = metrics.accuracy_score(true_label[i], pred_label[i], normalize=normalize)
    return per_accuracy


def calculate_multilabel_confusion_matrix_info(pred_label, true_label):
    mcm = metrics.multilabel_confusion_matrix(true_label, pred_label)
    tn, tp, fn, fp = mcm[:, 0, 0], mcm[:, 1, 1], mcm[:, 1, 0], mcm[:, 0, 1]
    return tn, tp, fn, fp


def calculate_miss_rate(tp, fn):
    miss_rate = fn / (tp + fn + 1e-7)
    return miss_rate


def calculate_specificity(fp, tn):
    specificity = tn / (fp + tn + 1e-7)
    return specificity


def calculate_fall_out(fp, tn):
    fall_out = fp / (fp + tn + 1e-7)
    return fall_out


def calculate_multilabel_metric(tn, tp, fn, fp):
    pfp = np.where((tp + fp) == 0, 1, fp)
    rfn = np.where((tp + fn) == 0, 1, fn)
    mtp = np.where((tp + fn) == 0, 1, tp)
    sfp = np.where((fp + tn) == 0, 1, fp)
    ftn = np.where((fp + tn) == 0, 1, tn)

    precision = tp / (tp + pfp)
    recall = tp / (tp + rfn)
    miss_rate = fn / (mtp + fn)
    specificity = tn / (sfp + tn)
    fall_out = fp / (fp + ftn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        'precision': precision,
        'recall': recall,
        'miss_rate': miss_rate,
        'specificity': specificity,
        'fall_out': fall_out,
        'accuracy': accuracy
    }


"""
计算每一个标签的 F1 分数
"""


def calculate_f1_score(pred_label, true_label, average=None, zero_division=0):
    return metrics.f1_score(true_label, pred_label, average=average, zero_division=zero_division)


"""
计算每一个标签的精确度
"""


def calculate_precision(pred_label, true_label, average=None, zero_division=0):
    return metrics.precision_score(true_label, pred_label, average=average, zero_division=zero_division)


"""
计算每一个标签的召回率
"""


def calculate_recall(pred_label, true_label, average=None, zero_division=0):
    return metrics.recall_score(true_label, pred_label, average=average, zero_division=zero_division)


"""
计算整体的汉明损失
"""


def calculate_hamming_loss(pred_label, true_label):
    return metrics.hamming_loss(true_label, pred_label)


def calculate_average_precision_score(pred_score, true_label, average=None):
    return metrics.average_precision_score(true_label, pred_score, average=average)


def calculate_precision_recall_fscore_support(pred_label, true_label, average=None):
    return metrics.precision_recall_fscore_support(true_label, pred_label, average=average)


def calculate_roc_auc_score(pred_score, true_label, average=None):
    class_sum = np.sum(true_label, axis=0)
    cal_col = []
    uncal_col = []
    for col in range(len(class_sum)):
        if class_sum[col] == 0:
            uncal_col.append(col)
        else:
            cal_col.append(col)
    if not uncal_col:
        return metrics.roc_auc_score(true_label, pred_score, average=average)
    true_label_copy = true_label.copy()
    true_label_copy[0][uncal_col] = 1
    roc_auc_score = metrics.roc_auc_score(true_label_copy, pred_score, average=average)
    roc_auc_score[uncal_col] = np.nan
    return roc_auc_score


def calculate_multilabel_metrics(pred_score, pred_label, true_label, average=None, normalize=True, zero_division=0):
    accuracy = calculate_accuracy(pred_label, true_label, normalize=normalize)
    per_accuracy = calculate_accuracy_per_label(pred_label, true_label, normalize=normalize)
    precision = calculate_precision(pred_label, true_label, average=average, zero_division=zero_division)
    recall = calculate_recall(pred_label, true_label, average=average, zero_division=zero_division)
    f1_score = calculate_f1_score(pred_label, true_label, average=average, zero_division=zero_division)
    micro_f1 = calculate_f1_score(pred_label, true_label, average="micro", zero_division=zero_division)
    average_precision_score = calculate_average_precision_score(pred_score, true_label, average=average)
    roc_auc_score = calculate_roc_auc_score(pred_score, true_label, average=average)
    hamming_loss = calculate_hamming_loss(pred_label, true_label)

    tn, tp, fn, fp = calculate_multilabel_confusion_matrix_info(pred_label, true_label)
    missing_rate = 1 - recall
    specificity = calculate_specificity(fp, tn)
    fall_out = 1 - specificity
    false_alarm = 1 - precision

    metric_dict = {'accuracy': accuracy,
                   'per_accuracy': per_accuracy.tolist(),
                   'precision': precision.tolist(),
                   'recall': recall.tolist(),
                   'f1_score': f1_score.tolist(),
                   'micro_f1': micro_f1,
                   'average_precision_score': average_precision_score.tolist(),
                   'roc_auc_score': roc_auc_score.tolist(),
                   'hamming_loss': hamming_loss,
                   'missing_rate': missing_rate.tolist(),
                   'specificity': specificity.tolist(),
                   'fall_out': fall_out.tolist(),
                   'false_alarm': false_alarm.tolist()
                   }
    return metric_dict


def transfer_metrics_to_dataframe(metric_dict: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(metric_dict)


class Accumulator:
    """
    For accumulating sums over `n` variables
    """

    def __init__(self, n: int = 1):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [x + float(y) for x, y in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Evaluator:
    def __init__(self):
        self.data = {
            "train": {},
            "local_test": {},
            "global_test": {}
        }

    def evaluate(self, mode, epoch, pred_score, pred_label, true_label):
        raise NotImplementedError()

    def save(self, path):
        with open(path, "w") as file:
            file.write(json.dumps(self.data))


class MultiLabelEvaluator(Evaluator):
    def add_dict(self, mode, epoch, metric_dict):
        self.data[mode][epoch] = metric_dict

    def evaluate(self, mode, epoch, pred_score, pred_label, true_label):
        metric_dict = calculate_multilabel_metrics(pred_score, pred_label, true_label)
        self.data[mode][epoch] = metric_dict


class FedClientMultiLabelEvaluator(Evaluator):
    def add_dict(self, mode, cround, epoch, metric_dict):
        if cround not in self.data[mode].keys():
            self.data[mode][cround] = {}
        self.data[mode][cround][epoch] = metric_dict


class FedServerMultiLabelEvaluator(Evaluator):
    def add_dict(self, mode, cround, metric_dict):
        self.data[mode][cround] = metric_dict


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError(
            "The first supplied array does not contain any binary object."
        )
    if 0 == np.count_nonzero(reference):
        raise RuntimeError(
            "The second supplied array does not contain any binary object."
        )

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def hausdorff_distance(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`assd`
    :func:`asd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    if np.sum(result) == 0 or np.sum(reference) == 0:
        return 100
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def cal_hd(pred, target, mask, num_class=4, reduction='mean'):
    hds = np.zeros(pred.shape[0])
    for idx in range(pred.shape[0]):
        if mask[idx] != 0:
            hds[idx] = hausdorff_distance(pred[idx], target[idx])
        else:
            preds = np.array([np.where(pred[idx] == c, 1, 0) for c in range(1, num_class)])
            targets = np.array([np.where(target[idx] == c, 1, 0) for c in range(1, num_class)])
            hds[idx] = np.mean([hausdorff_distance(preds[i], targets[i]) for i in range(len(preds))])
    if reduction == 'mean':
        return np.mean(hds)
    elif reduction == 'sum':
        return np.sum(hds)
    else:
        return hds


def shield(pred, mask):
    # (batch, n_classes, h, w)
    shield_pred = torch.argmax(pred.detach(), dim=1) if pred.dim() == 4 else deepcopy(pred.detach())
    one_mask = torch.eq(mask, 1).flatten()
    two_mask = torch.eq(mask, 2).flatten()
    shield_pred[one_mask] = torch.where(shield_pred[one_mask] != 1, 0, shield_pred[one_mask])
    shield_pred[two_mask] = torch.where(shield_pred[two_mask] != 2, 0, shield_pred[two_mask])
    return shield_pred

def generate_pseudo_label(pred_label, true_label, mask):
    # (batch, n_classes, h, w)
    pseudo_label = deepcopy(pred_label.detach())
    # (batch, h, w)
    zero_mask = torch.eq(mask, 0).flatten()
    one_mask = torch.eq(mask, 1).flatten()
    two_mask = torch.eq(mask, 2).flatten()
    pseudo_label[one_mask] = torch.where(pseudo_label[one_mask] == 1, 0, pseudo_label[one_mask])
    pseudo_label[one_mask] = torch.where(true_label[one_mask] == 1, 1, pseudo_label[one_mask])
    pseudo_label[two_mask] = torch.where(pseudo_label[two_mask] == 2, 0, pseudo_label[two_mask])
    pseudo_label[two_mask] = torch.where(true_label[two_mask] == 2, 2, pseudo_label[two_mask])
    pseudo_label[zero_mask] = true_label[zero_mask]
    return pseudo_label
