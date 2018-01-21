import numpy as np
import glob
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def svm_classify(pathToTrainingFeatures,pathToTestingFeatures):
    lines=[]
    for name in glob.glob(pathToTrainingFeatures+'*.txt'):
        with open(name) as fp:
            lines=fp.readlines()

    familylables=np.load(pathToTrainingFeatures+lines[1].strip())
    featureVectors=[]
    print('Loading features from valid.npy files')
    for line in lines[2:]:
        featureVectors.append(np.load(pathToTrainingFeatures+line.strip()))

    featureVectors[0]=np.reshape(featureVectors[0],(featureVectors[0].shape[0],60,126))

    aggregatedVectors=[]
    print('Aggregating features')
    for vector in featureVectors:
        aggregatedVectors.append(np.array(vector).mean(2))
        aggregatedVectors.append(np.array(vector).std(2))

    x_train = np.concatenate(tuple(aggregatedVectors), axis=1)

    y_train =  np.vectorize(str)(familylables)

    print('Performing cross validation')
    model = SVC(C=100, gamma=0.00001)
    kfold = model_selection.KFold(n_splits=5, random_state=3, shuffle=True)
    crossval_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    print('Cross-validation results')
    print(crossval_results)
    print(crossval_results.mean())

    lines = []
    for name in glob.glob(pathToTestingFeatures + '*.txt'):
        with open(name) as fp:
            lines = fp.readlines()

    familylables = np.load(pathToTestingFeatures + lines[1].strip())
    featureVectors = []
    print('Loading features from test.npy files')
    for line in lines[2:]:
        a = pathToTestingFeatures + line.strip()
        featureVectors.append(np.load(a))

    featureVectors[0] = np.reshape(featureVectors[0], (featureVectors[0].shape[0], 60, 126))

    aggregatedVectors = []
    print('Aggregating features')
    for vector in featureVectors:
        aggregatedVectors.append(np.array(vector).mean(2))
        aggregatedVectors.append(np.array(vector).std(2))



    x_test = np.concatenate(tuple(aggregatedVectors), axis=1)

    y_test = np.vectorize(str)(familylables)

    print('Testing on unknown data')
    print('Fitting model')
    model.fit(x_train, y_train)
    print('Generating model predictions')
    outputs = model.predict(x_test)
    print('Test results')
    print('Accuracy score: ')
    print(accuracy_score(y_test, outputs))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, outputs))
    print('Classification report: ')
    print(classification_report(y_test, outputs))



if __name__ == '__main__':
    svm_classify("data20171210T005158194276base_trainingAE/","data20171210T005120109248base_testing/")