# coding:utf-8
import numpy as np
import torch


def intersect_and_union(num_class, logits, labels):
    logits = logits.argmax(0)
    intersect = logits[logits == labels]
    area_intersect = torch.histc(intersect.float(), bins=(num_class), min=0, max=num_class - 1)
    area_pred_label = torch.histc(logits.float(), bins=(num_class), min=0, max=num_class - 1)
    area_label = torch.histc(labels.float(), bins=(num_class), min=0, max=num_class - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(num_class, logits, labels):
    total_area_intersect = torch.zeros((num_class,), dtype=torch.float64)
    total_area_union = torch.zeros((num_class,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_class,), dtype=torch.float64)
    total_area_label = torch.zeros((num_class,), dtype=torch.float64)

    total_area_intersect = total_area_intersect.cuda()
    total_area_union = total_area_union.cuda()
    total_area_pred_label = total_area_pred_label.cuda()
    total_area_label = total_area_label.cuda()

    for i in range(logits.shape[0]):
        it_logit = logits[i]
        it_label = labels[i]
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(num_class, it_logit, it_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    iou = total_area_intersect / total_area_union
    miou = iou.mean()
    acc = total_area_intersect / total_area_label
    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label
    return all_acc, iou, miou, acc, precision, recall


def calculate_accuracy(logits, labels):
    predictions = logits.argmax(1)
    no_count = (labels == -1).sum()
    count = ((predictions == labels) * (labels != -1)).sum()
    acc = count.float() / (labels.numel() - no_count).float()
    return acc


def calculate_meaniou(num_class, logits, labels):
    unique_labels = np.unique(labels)

    I = np.zeros(num_class)
    U = np.zeros(num_class)

    for index, val in enumerate(unique_labels):
        pred_i = logits == val
        label_i = labels == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou
