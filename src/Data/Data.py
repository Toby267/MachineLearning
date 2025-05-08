import pandas as pd

from Data.StandardScaler import StandardScaler

def getDataFrame() -> pd.DataFrame:
    """returns the glass data frame"""
    df = pd.read_csv("./glass.data", header=None)
    df.columns = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Class"]
    df = df.drop("ID", axis=1)
    return df

def trainTestSplit(independent: pd.DataFrame, dependant: pd.DataFrame, testSize: float, seed: int) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """splits the data as in sklearn.model_selection.train_test_split"""
    independentShuffled = independent.sample(frac=float(1), random_state=seed)
    dependentShuffled = dependant.sample(frac=float(1), random_state=seed)
    
    testCount = independentShuffled.shape[0] * testSize
    testCount = round(testCount)
    
    return independentShuffled.iloc[testCount:], independentShuffled.iloc[:testCount], dependentShuffled.iloc[testCount:], dependentShuffled.iloc[:testCount]

#for decision trees
def getSplitData() -> list:
    """returns the glass data frame split into xTrain, xTest, yTrain, and yTest"""
    df = getDataFrame()

    dependant = df["Class"]
    independent = df.drop("Class", axis=1)

    #xTrain, xTest, yTrain, yTest
    return trainTestSplit(independent, dependant, float(0.2), 39018563)

#for KNN, Naive bayes, and SVM
def getStandardisedSplitData() -> list:
    """returns the glass data frame split and standardised into xTrain, xTest, yTrain, and yTest"""
    df = getDataFrame()

    dependant = df["Class"]
    independent = df.drop("Class", axis=1)

    scaler = StandardScaler()
    independentScaled = scaler.fitTransform(independent)

    #xTrain, xTest, yTrain, yTest
    return trainTestSplit(independentScaled, dependant, float(0.2), 39018563)