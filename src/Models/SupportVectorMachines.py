import pandas as pd
import numpy as np

class BinarySVM:
    LAMBDA: float
    LEARNING_RATE: float
    ITERATIONS: int

    w: int
    b: int

    def __init__(self, l: float = 0.001, lr: float = 0.01, i: float = 1000):
        self.LAMBDA = l
        self.LEARNING_RATE = lr
        self.ITERATIONS = i

    def fit(self, xTrain: pd.DataFrame, yTrain: pd.DataFrame):
        """uses the given data frames to calculate approximations for w and b assuming the yTrain is -1 or 1"""
        x = xTrain.to_numpy()
        y = yTrain.to_numpy()

        self.b = float(y.mean())
        self.w = np.zeros(len(xTrain.columns))

        for _ in range(self.ITERATIONS):
            for i, xi in enumerate(x):
                yi = y[i]
                fx = np.dot(self.w, xi) - self.b

                wDot = 0 if yi*fx >= 1 else np.dot(xi, yi)
                self.w -= self.LEARNING_RATE * (2 * self.LAMBDA * self.w - wDot)

                self.b -= 0 if yi*fx >= 1 else self.LEARNING_RATE * yi

    def predict(self, xTest: pd.DataFrame) -> list[str]:
        """predicts the class for each row in xTest using the formula"""
        return [self.predictSingle(row.values) for _, row in xTest.iterrows()]

    def predictSingle(self, x: list) -> list[int]:
        """predicsts the class of x using the formaula"""
        distance = np.dot(x, self.w) - self.b
        return (1, distance) if distance >= 0 else (-1, distance)


class MultiFactorSVM:
    binarySVMs: dict[str: BinarySVM]
    classNames: list[str]

    LAMBDA: float
    LEARNING_RATE: float
    ITERATIONS: int

    def __init__(self, l: float = 0.001, lr: float = 0.01, i: float = 1000):
        self.LAMBDA = l
        self.LEARNING_RATE = lr
        self.ITERATIONS = i

    def fit(self, xTrain: pd.DataFrame, yTrain: pd.DataFrame):
        """generates each binary SVM and fits the data to them"""
        self.classNames = sorted(set([str(i) for i in yTrain.values.tolist()]))
        self.binarySVMs = {key: BinarySVM(self.LAMBDA, self.LEARNING_RATE, self.ITERATIONS) for key in self.classNames}
        
        for key, svm in self.binarySVMs.items():
            yTrainBinary = yTrain.copy()

            yTrainBinary[yTrainBinary != int(key)] = -1
            yTrainBinary[yTrainBinary == int(key)] = 1

            svm.fit(xTrain, yTrainBinary)

    def predict(self, xTest: pd.DataFrame) -> list[str]:
        """predicts the class for each row in xTest using the formula"""
        predictions = None
    
        for key, svm in self.binarySVMs.items():
            newPredictions = [(int(key), i[1]) for i in svm.predict(xTest)]

            if predictions == None:
                predictions = newPredictions
                continue
            
            for i, val in enumerate(newPredictions):
                if val[1] > predictions[i][1]:
                    predictions[i] = newPredictions[i]

        return [i[0] for i in predictions]
