from evaluation import calculate_meaniou, calculate_accuracy


def most_sure(pred1, pred2, num_class):
    #####得到每个像素，两种网络预测的类别
    sure_class1 = pred1.argmax(1)
    sure_class2 = pred2.argmax(1)

    #####得到每个像素，所选类别的预测值,即pred[x,c,w,h]

    #####比较两者，若不同，比较预测值
    for x in pred1.shape[0]:
        for w in pred1.shape[2]:
            for h in pred1.shape[3]:
                if sure_class1[w, h] != sure_class2[w, h]:
                    #####选择更高预测值的类别
                    if pred1[x, sure_class1[w, h], w, h] <= pred2[x, sure_class1[w, h], w, h]:
                        pred1[x, :, w, h] = pred2[x, :, w, h]
    ####这种方法速度很慢，我想想有没有矩阵运算方法
    return pred1


def get_result(num_class, pred, logit):
    result = calculate_meaniou(num_class, pred, logit)
    return result


def merge_prediction(pred1, pred2, logit, num_class, mode):
    score_1 = get_result(num_class, pred1, logit)
    score_2 = get_result(num_class, pred2, logit)

    if mode == "sample":
        pred = most_sure(pred1, pred2, num_class)
    return pred
