import math
import pandas as pd

class KNN:
    k: int
    classValues: int

    x: pd.DataFrame
    y: pd.DataFrame

    def __init__(self, k: int = 3):
        self.k = k
    
    def fit(self, xTrain: pd.DataFrame, yTrain: pd.DataFrame):
        """fits the x and y dataframes for making predictions"""
        self.x = xTrain
        self.y = yTrain
        self.classValues = sorted(set([str(i) for i in yTrain.values.tolist()]))

    
    def predict(self, xTest: pd.DataFrame) -> list[str]:
        """takes in a dataframe of data points, and returns a list of classes that each data point is most likely to belong to"""
        return [self.predictSingle(row) for _, row in xTest.iterrows()]


    def predictSingle(self, x: list) -> str:
        """takes in a single data point and returns the class that it most likely belongs to"""
        closestK = self.getKClosest(x)
        tally = {key: 0 for key in self.classValues}

        for distance in closestK:
            index = distance[0]
            dataPoint = self.y.loc[index]
            tally[str(dataPoint)] += 1

        return max(tally, key=tally.get)

    def getKClosest(self, x: list) -> list:
        """x is one data point, returns a list of the the k closest data point containing their 
        index and distance"""
        maxDistance = float('inf')
        points = []
        length = 0

        for index, row in self.x.iterrows():
            dist = DistanceMetrics.manhattanDistance(x, row.tolist())
            if dist < maxDistance or length != self.k:
                length += 1
                maxDistance = dist
                points.append([index, dist])

        sortedPoints = sorted(points, key=lambda x: x[1])

        return sortedPoints[0:self.k]

class DistanceMetrics:
    def euclideanDistance(dp1: list, dp2: list) -> int:
        """calculates the squared euclidean distance between data point 1 and 2 each data point is a list of attributes"""
        tDistance = 0
        for i, j in zip(dp1, dp2):
            tDistance += (i-j)**2
        return math.sqrt(tDistance)

    def squaredEuclideanDistance(dp1: list, dp2: list) -> int:
        """calculates the euclidean distance between data point 1 and 2 each data point is a list of attributes"""
        tDistance = 0
        for i, j in zip(dp1, dp2):
            tDistance += (i-j)**2
        return tDistance

    def manhattanDistance(dp1: list, dp2: list) -> int:
        """calculates the manhattan distance between data point 1 and 2 each data point is a list of attributes"""
        tDistance = 0
        for i, j in zip(dp1, dp2):
            tDistance += abs(i-j)
        return tDistance

    def cosineDistance(dp1: list, dp2: list) -> int:
        """calculates the cosine distance between data point 1 and 2 each data point is a list of attributes"""
        dotProduct = 0
        sumOfiSquares = 0
        sumOfjSquares = 0
        for i, j in zip(dp1, dp2):
            dotProduct += i*j
            sumOfiSquares += i**2
            sumOfjSquares += j**2
        return dotProduct / (sumOfiSquares * sumOfjSquares)

    def hammingDistance(dp1: list, dp2: list) -> int:
        """calculates the hamming distance between data point 1 and 2 each data point is a list of attributes"""
        distance = 0
        for i, j in zip(dp1, dp2):
            if i != j:
                distance += 1
        return distance