import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def accuraci(y_pred, y_true) :
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp + tn) / (tp + tn + fp + fn)

def sensitivity(y_pred, y_true) :
    # y_pred = y_pred.to_numpy().flatten()
    # y_true = y_true.to_numpy().flatten()
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn)

def specificity(y_pred, y_true) :
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tn / (tn + fp)

def precision(y_pred, y_true) :
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp)

def f1_score(y_pred, y_true) :
    prec = precision(y_pred, y_true)
    rec = sensitivity(y_pred, y_true)
    return 2 * (prec * rec) / (prec + rec)

def printScoring(y_pred,y_pred_precision,y_true, retrn = False) :
    if retrn == True :
        return {
            'Accuracy': accuraci(y_pred, y_true),
            'Sensitivity': sensitivity(y_pred, y_true),
            'Specificity': specificity(y_pred, y_true),
            'Precision': precision(y_pred, y_true),
            'F1 Score': f1_score(y_pred, y_true),
            'AUROC': roc_auc_score(y_true, y_pred_precision),
            'AUPR': average_precision_score(y_true, y_pred_precision)
        }
    # up to 3 decimal point
    print()
    print('Accuracy : {:.7f}'.format(accuraci(y_pred, y_true)))
    print('Sensitivity : {:.7f}'.format(sensitivity(y_pred, y_true)))
    print('Specificity : {:.7f}'.format(specificity(y_pred, y_true)))
    print('Precision : {:.7f}'.format(precision(y_pred, y_true)))
    print('F1 Score : {:.7f}'.format(f1_score(y_pred, y_true)))
    print('AUROC : {:.7f}'.format(roc_auc_score(y_true, y_pred_precision)))
    print('AUPR : {:.7f}'.format(average_precision_score(y_true, y_pred_precision)))
    # print new line
    print()
