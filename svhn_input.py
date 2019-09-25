import keras
from scipy import io
import numpy as np

def get_svhn_data(data_dir='../datasets'):
    train_data = io.loadmat(data_dir + '/svhn/train_32x32.mat')
    y_train = keras.utils.to_categorical(train_data['y'][:, 0])[:, 1:]
    test_data = io.loadmat(data_dir +'/svhn/test_32x32.mat')
    y_test = keras.utils.to_categorical(test_data['y'][:,0])[:,1:]
    extra_data = io.loadmat(data_dir+'/svhn/extra_32x32.mat')
    y_extra = keras.utils.to_categorical(extra_data['y'][:, 0])[:, 1:][0:10000]

    X_train = np.zeros((73257, 32, 32, 3))
    for i in xrange(len(X_train)):
        X_train[i] = train_data['X'].T[i].T.astype('float32')
    X_test = np.zeros((26032, 32, 32, 3))
    for i in range(len(X_test)):
        X_test[i] = test_data['X'].T[i].T.astype('float32')
    X_extra = np.zeros((10000, 32, 32, 3))
    for i in range(len(X_extra)):
        X_extra[i] = extra_data['X'].T[i].T.astype('float32')
    X_both = np.concatenate([X_train, X_extra], axis=0)
    y_both = np.concatenate([y_train, y_extra], axis=0)
    return X_both.astype(np.float32), y_both.astype(np.float32), X_test.astype(np.float32), y_test.astype(np.float32)


if __name__ == "__main__":
    pass