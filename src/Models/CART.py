import pandas as pd

class Node:
    feature: str
    threshold: float
    class_: str
    left: "Node"
    right: "Node"

    def __init__(self, feature: str = "", threshold: float = 0, class_: str = "", left: "Node" = None, right: "Node" = None):
        self.feature = feature
        self.threshold = threshold
        self.class_ = class_
        self.left = left
        self.right = right

    def traverse(self, x: pd.Series) -> str:
        """returns the class for the given data point"""
        if self.class_ != "":
            return self.class_

        if (x[self.feature] < self.threshold):
            return self.left.traverse(x)
        else:
            return self.right.traverse(x)

class CART:
    maxDepth: int
    tree: Node

    x: pd.DataFrame
    y: pd.DataFrame

    def __init__(self, maxDepth: int):
        self.maxDepth = maxDepth

    def fit(self, xTrain: pd.DataFrame, yTrain: pd.DataFrame):
        """fits the x and y dataframes for making predictions"""
        self.x = xTrain
        self.y = yTrain

        self.tree = self.train(xTrain, 0)

    def train(self, xTrain: pd.DataFrame, depth: int) -> Node:
        """recursive function that generates and returns the decision tree, xTrain contains all data points for the current node
            is the dictionary operator is either < or >="""
        
        list = [self.y.loc[i] for i, _ in xTrain.iterrows()]
        classes = set(list)

        if len(classes) <= 1 or depth >= self.maxDepth:
            val = max(classes, key=list.count)
            return Node(class_=val) #returns the leaf node
        
        feature, threshold, leftDf, rightDf = self.getBestSplit(xTrain)
        left, right = None, None

        if leftDf is not None:
            left = self.train(leftDf, depth+1)
        if rightDf is not None:
            right = self.train(rightDf, depth+1)
        
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def getBestSplit(self, xTrain: pd.DataFrame) -> list[str, float, pd.DataFrame, pd.DataFrame]:
        """returns the best splitt of xTrain based on the data point dataPoint"""
        splitFeatureIndex = ""
        splitThreshold = 0
        splitGiniImpurity = float('inf')
        left = None
        right = None

        for _, dataPoint in xTrain.iterrows():
            for index, value in dataPoint.items():
                s = xTrain.shape[0]

                xSubTrain1 = xTrain.copy()
                xSubTrain1 = xSubTrain1[xSubTrain1[index] < value]
                s1 = xSubTrain1.shape[0]
                if s1 == 0:
                    continue
                giniImpurity1 = self.calcGiniImpurity(xSubTrain1)

                xSubTrain2 = xTrain.copy()
                xSubTrain2 = xSubTrain2[xSubTrain2[index] >= value]
                s2 = xSubTrain2.shape[0]
                if s2 == 0:
                    continue
                giniImpurity2 = self.calcGiniImpurity(xSubTrain2)
                
                finalGiniImpurity = (s1 * giniImpurity1 + s2 * giniImpurity2) / s
                
                if finalGiniImpurity < splitGiniImpurity:
                    splitFeatureIndex = index
                    splitThreshold = value
                    splitGiniImpurity = finalGiniImpurity
                    left = xSubTrain1
                    right = xSubTrain2

        return splitFeatureIndex, splitThreshold, left, right

    def calcGiniImpurity(self, xTrain: pd.DataFrame) -> float:
        """calculates the gini impurity of the given data frame"""
        classes = sorted(set([str(i) for i in self.y.values.tolist()]))
        dict = {key: float(0) for key in classes}
        total = xTrain.shape[0]

        if total == 0:
            return 100

        for index, row in xTrain.iterrows():
            dict[str(self.y.loc[index])] += 1

        giniImpurity = 1
        for _, value in dict.items():
            giniImpurity -= (value/total)**2

        return giniImpurity

    def predict(self, xTest: pd.DataFrame) -> list[str]:
        """takes in a dataframe of data points, and returns a list of classes that each data point is most likely to belong to"""
        return [self.predictSingle(row) for _, row in xTest.iterrows()]

    def predictSingle(self, x: pd.Series):
        """takes in a single data point and returns the class that it most likely belongs to"""
        return self.tree.traverse(x)
