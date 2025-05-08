import pandas as pd
import math

def calcMean(list: list):
    return sum(list) / len(list)

def calcVariance(list: list):
    mean = calcMean(list)
    return sum([(i-mean)**2 for i in list]) / (len(list)-1)

class NaiveBayes:
    x: pd.DataFrame
    y: pd.DataFrame

    featureNames: list[str]
    classNames: list[str]
    classProbabilities: dict[str: float]

    def __init__(self):
        pass

    def fit(self, xTrain: pd.DataFrame, yTrain: pd.DataFrame):
        """fits the x and y dataframes for making predictions"""
        self.x = xTrain
        self.y = yTrain

        self.featureNames = xTrain.columns.tolist()
        self.classNames = sorted(set([str(i) for i in yTrain.values.tolist()]))
        self.classProbabilities = {key: float(0) for key in self.classNames}

        for i in yTrain:
            self.classProbabilities[str(i)] += 1
        
        for key, value in self.classProbabilities.items():
            self.classProbabilities[key] = value/len(yTrain)

    def predict(self, xTest: pd.DataFrame) -> list[str]:
        """takes in a dataframe of data points, and returns a list of classes that each data point is most likely to belong to"""
        return [self.predictSingle(row) for _, row in xTest.iterrows()]

    def predictSingle(self, x: list):
        """takes in a single data point and returns the class that it most likely belongs to"""
        classProbabilities = {key: float(0) for key in self.classNames}
        
        for name in self.classNames:
            classProbabilities[name] = self.calcClassProbability(x, name)
        
        return max(classProbabilities, key=classProbabilities.get)


    def calcClassProbability(self, x: list, classStr: str) -> float:
        """calculates the probability P(C|A1, A2, ..., An)"""
        #select all from xTrain wher yTrain = class
        classRows = self.x[self.y.astype(str) == classStr]

        p = self.classProbabilities[classStr]

        for feature, value in zip(self.featureNames, x):
            featuresValues = classRows[feature].tolist()
            mean = calcMean(featuresValues)
            variance = calcVariance(featuresValues)
            
            p *= self.calcFeatureProbability(value, mean, variance)

        return p

    def calcFeatureProbability(self, x: float, featureMean: float, featureVariance: float) -> float:
        """calculates the probability for the given datapoint, mean and variance
            i.e.: P(A|C) - the C comes from the fact that featureMean and featureVariance are calculated from values where the class is C"""
        if featureVariance == 0:
            featureVariance = 0.00001

        p = 1/(math.sqrt(2 * math.pi * featureVariance))
        exponent = ((x-featureMean)**2) / (2 * featureVariance)
        exponent = -exponent
        p *= math.e**exponent

        return p
