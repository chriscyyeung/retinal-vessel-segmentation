import numpy as np
from medpy import metric as mt


def accuracy(pred, true):
    """Calculates the accuracy of the prediction by dividing the number of
    correctly classified images by the total number of test samples.

    :param pred: an array of shape (n_samples, n_classes) of the predicted
                 class labels
    :param true: the ground truth class labels
    :return: a float representing the accuracy of the prediction
    """
    num_correct = np.sum((pred == true).numpy().all(axis=1))
    return num_correct / pred.shape[0]


def calculate_eval_metric(predicted, target_gt):
    dc = mt.binary.dc(predicted, target_gt)
    precision = mt.binary.precision(predicted, target_gt)
    recall = mt.binary.recall(predicted, target_gt)
    sensitivity = mt.binary.sensitivity(predicted, target_gt)
    specificity = mt.binary.specificity(predicted, target_gt)

    return dc, precision, recall, sensitivity, specificity
