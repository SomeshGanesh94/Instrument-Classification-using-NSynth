import spams
import numpy as np
import glob
import datetime
from datetime import time

def sparse_feature_training(pathToTrainingFeatures,pathToTestingFeatures):
    lines=[]
    for name in glob.glob(pathToTrainingFeatures+'*.txt'):
        with open(name) as fp:
            lines=fp.readlines()

    # dataorder = np.load(pathToTrainingFeatures + lines[0].strip())
    # familylables=np.load(pathToTrainingFeatures+lines[1].strip())
    featureVectors=[]
    print('Loading features from valid.npy files')
    for line in lines[2:]:
        featureVectors.append(np.load(pathToTrainingFeatures+line.strip()))

    #featureVectors[0]=np.reshape(featureVectors[0],(featureVectors[0].shape[0],60,126))

    featureVector=featureVectors[0]

    #x_train = np.concatenate(tuple(aggregatedVectors), axis=1)
    x_train = featureVectors[0]
    np.random.shuffle(x_train)
    # x_train = x_train[0:1000]
    x_train = x_train.reshape((126*x_train.shape[0], -1))

    print(x_train.shape)

    X = x_train.transpose()

    print('Training sparse dictionary')

    param = {'K': 512,
             'lambda1': 0.85, 'numThreads': 1, 'batchsize': 128,
             'iter': 1000}

    sparse_dictionary = spams.trainDL(X=X, **param)

    np.save('sparsedict'+'K'+str(param['K'])+'lambda1'+str(param['lambda1'])+'batchsize'+str(param['batchsize'])+'iter'+str(param['iter'])+'.npy', sparse_dictionary)

    print(sparse_dictionary)

if __name__ == '__main__':
    sparse_feature_training("data20171209T211347019822dict_training/(0, 1, 2)/","")