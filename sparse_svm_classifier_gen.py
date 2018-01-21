import numpy as np
import glob
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def svm_classify(pathToTrainingFeatures,dictionaryPath, pathToTestingFeatures):
    lines=[]
    for name in glob.glob(pathToTrainingFeatures+'*.txt'):
        with open(name) as fp:
            lines=fp.readlines()

    familylables=np.load(pathToTrainingFeatures+lines[1].strip())
    featureVectors=[]
    print('Loading features from train.npy files')
    for line in lines[2:]:
        featureVectors.append(np.load(pathToTrainingFeatures+line.strip()))

    aggregatedVectors=[]
    # print('Aggregating features')
    # for vector in featureVectors:
    #     aggregatedVectors.append(np.array(vector).mean(2))
    #     aggregatedVectors.append(np.array(vector).std(2))
    # aggregatedVectors = np.load('aggregatedFeatures.npy')

    ##missing delta and delta delta

    x_train = np.concatenate(tuple(featureVectors), axis=1)
    sparsedictionary=np.load(dictionaryPath)

    y_train = familylables

    num_of_train_set = len(y_train)

    # x_train = x_train[0:num_of_train_set]
    # y_train = y_train[0:num_of_train_set]

    print('Reshaping and multiplying with sparse dictionary')

    x_train = x_train.reshape((num_of_train_set*126, -1))
    x_train = x_train.dot(sparsedictionary)
    x_train = x_train.reshape((num_of_train_set, -1))

    print('X_train shape:')
    print(x_train.shape)
    print('Y_train shape:')
    print(y_train.shape)

    print('Performing cross validation')
    # # model = SVC(C=10)
    model = SVC(C=0.01, gamma=0.00001)
    kfold = model_selection.KFold(n_splits=5, random_state=3, shuffle=True)
    crossval_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    print('Cross-validation results')
    print(crossval_results)
    print(crossval_results.mean())

    # lines = []
    # for name in glob.glob(pathToTestingFeatures + '*.txt'):
    #     with open(name) as fp:
    #         lines = fp.readlines()
    #
    # familylables = np.load(pathToTestingFeatures + lines[1].strip())
    # featureVectors = []
    # print('Loading features from test.npy files')
    # for line in lines[2:]:
    #     a=pathToTestingFeatures + line.strip()
    #     featureVectors.append(np.load(a))

    # aggregatedVectors = []
    # print('Aggregating features')
    # for vector in featureVectors:
    #     print('Mean')
    #     aggregatedVectors.append(np.array(vector).mean(2))
    #     print('Std')
    #     aggregatedVectors.append(np.array(vector).std(2))
    # np.save('aggregatedFeaturesTest.npy', np.asarray(aggregatedVectors))

    # x_test = np.concatenate(tuple(featureVectors), axis=1)
    #
    # y_test = familylables
    #
    # num_of_test_set = len(y_test)
    #
    # # x_test = x_test[0:num_of_test_set]
    # # y_test = y_test[0:num_of_test_set]
    #
    # x_test = x_test.reshape((num_of_test_set * 126, -1))
    # x_test = x_test.dot(sparsedictionary)
    # x_test = x_test.reshape((num_of_test_set, -1))
    #
    # print('Testing on unknown data')
    # print('Fitting model')
    # model.fit(x_train, y_train)
    # print('Generating model predictions')
    # outputs = model.predict(x_test)
    # print('Test results')
    # print('Accuracy score: ')
    # print(accuracy_score(y_test, outputs))
    # print('Confusion matrix: ')
    # print(confusion_matrix(y_test, outputs))
    # print('Classification report: ')
    # print(classification_report(y_test, outputs))

if __name__ == '__main__':
    svm_classify("data20171209T230230994029svm_trainingAE/folder/","sparsedictK512lambda10.85batchsize128iter1000.npy","data20171209T235437739623svm_testing/")