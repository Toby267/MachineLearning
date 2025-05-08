import time
import matplotlib.pyplot as plt

from Data.Data import getSplitData, getStandardisedSplitData
from Data.Evaluation import calcAccuracy, calcConfusionMatrix, confusionMatrixDisplay, calcPrecision, calcRecall, calcF1Score

from Models.KNearestNeighbour import KNN
from Models.CART import CART
from Models.NaiveBayes import NaiveBayes
from Models.SupportVectorMachines import MultiFactorSVM

#To run, please put glass.data next to Main.py in the src directory
def main():
    #get data
    xTrain, xTest, yTrain, yTest = getSplitData()
    xTrainScaled, xTestScaled, _, _ = getStandardisedSplitData()

    #KNN
    startTime = time.time()

    knn = KNN(3)
    knn.fit(xTrainScaled, yTrain)
    trainTime = time.time() - startTime

    predictions = knn.predict(xTestScaled)
    predictionTime = time.time() - startTime - trainTime

    confusionMatrix = calcConfusionMatrix([i for i in yTest], [int(i) for i in predictions], [1, 2, 3, 4, 5, 6, 7])
    display = confusionMatrixDisplay(confusionMatrix, [1, 2, 3, 4, 5, 6, 7])
    display.plot()

    print(f"KNN Accuracy: {calcAccuracy([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Precision: {calcPrecision([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Recall: {calcRecall([i for i in yTest], [int(i) for i in predictions])}")
    print(f"F1: {calcF1Score([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Train time: {trainTime}, Testing time: {predictionTime}")




    #CART
    startTime = time.time()

    cart = CART(9)
    cart.fit(xTrain, yTrain)
    trainTime = time.time() - startTime
    
    predictions = cart.predict(xTest)
    predictionTime = time.time() - startTime - trainTime

    confusionMatrix = calcConfusionMatrix([i for i in yTest], [int(i) for i in predictions], [1, 2, 3, 4, 5, 6, 7])
    display = confusionMatrixDisplay(confusionMatrix, [1, 2, 3, 4, 5, 6, 7])
    display.plot()

    print(f"\nCART Accuracy: {calcAccuracy([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Precision: {calcPrecision([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Recall: {calcRecall([i for i in yTest], [int(i) for i in predictions])}")
    print(f"F1: {calcF1Score([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Training time: {trainTime}, Testing time: {predictionTime}")




    #Naive bayes
    startTime = time.time()

    naiveBayes = NaiveBayes()
    naiveBayes.fit(xTrainScaled, yTrain)
    trainTime = time.time() - startTime
    
    predictions = naiveBayes.predict(xTestScaled)
    predictionTime = time.time() - startTime - trainTime

    confusionMatrix = calcConfusionMatrix([i for i in yTest], [int(i) for i in predictions], [1, 2, 3, 4, 5, 6, 7])
    display = confusionMatrixDisplay(confusionMatrix, [1, 2, 3, 4, 5, 6, 7])
    display.plot()

    print(f"\nNaive Bayes Accuracy: {calcAccuracy([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Precision: {calcPrecision([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Recall: {calcRecall([i for i in yTest], [int(i) for i in predictions])}")
    print(f"F1: {calcF1Score([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Training time: {trainTime}, Testing time: {predictionTime}")




    #SVM
    startTime = time.time()

    svm = MultiFactorSVM(0.00003, 0.05, 1000)
    svm.fit(xTrainScaled, yTrain)
    trainTime = time.time() - startTime
    
    predictions = svm.predict(xTestScaled)
    predictionTime = time.time() - startTime - trainTime

    confusionMatrix = calcConfusionMatrix([i for i in yTest], [int(i) for i in predictions], [1, 2, 3, 4, 5, 6, 7])
    display = confusionMatrixDisplay(confusionMatrix, [1, 2, 3, 4, 5, 6, 7])
    display.plot()

    print(f"\nSVM Accuracy: {calcAccuracy([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Precision: {calcPrecision([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Recall: {calcRecall([i for i in yTest], [int(i) for i in predictions])}")
    print(f"F1: {calcF1Score([i for i in yTest], [int(i) for i in predictions])}")
    print(f"Training time: {trainTime}, Testing time: {predictionTime}")




    #display data
    plt.show()

if __name__ == "__main__":
    main()
