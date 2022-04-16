import numpy as np
import sklearn.metrics as metrics
from medpy import metric as mt


def calculate_eval_metric(predicted, target_gt):
    dc = mt.binary.dc(predicted, target_gt)
    precision = mt.binary.precision(predicted, target_gt)
    recall = mt.binary.recall(predicted, target_gt)
    sensitivity = mt.binary.sensitivity(predicted, target_gt)
    specificity = mt.binary.specificity(predicted, target_gt)

    return dc, precision, recall, sensitivity, specificity


def calculate_auc_test(prediction, label):
    # read images
    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()


    label_1D = label_1D / 255

    auc = metrics.roc_auc_score(label_1D, result_1D)

    # print("AUC={0:.4f}".format(auc))

    return auc


def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    return acc, sen
