import numpy as np

#remove once implemented
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score

def calcAccuracy(y1: list, y2: list) -> float:
    """returns the accuracy calculated from y1 and y2"""
    same = sum(i == j for i, j in zip(y1, y2))
    accuracy = same / len(y1)

    return accuracy

def calcConfusionMatrix(actual: list, predicted: list, lables: list) -> np.ndarray:
    """generates a confusion matrix based of actual and predicted"""
    matrix = np.zeros((len(lables), len(lables)))

    for a, p in zip(actual, predicted):
        if p <= 0 or p > len(lables):
            continue
        if a <= 0 or a > len(lables):
            continue

        matrix[a-1][p-1] += 1

    return matrix

def confusionMatrixDisplay(confusionMatrix: np.ndarray, lables: list) -> ConfusionMatrixDisplay:
    """returns a confusion matrix component for mathplotlib"""
    return ConfusionMatrixDisplay(confusionMatrix, display_labels=lables)

def calcPrecision(y1: list, y2: list) -> float:
    """returns the macro precision calculated from y1 and y2"""
    return precision_score(y1, y2, average='macro', zero_division = 0.0)

def calcRecall(y1: list, y2: list) -> float:
    """returns the macro recall calculated from y1 and y2"""
    return recall_score(y1, y2, average='macro', zero_division = 0.0)

def calcF1Score(y1: list, y2: list) -> float:
    """returns the macro F1 score calculated from y1 and y2"""
    return f1_score(y1, y2, average='macro', zero_division = 0.0)