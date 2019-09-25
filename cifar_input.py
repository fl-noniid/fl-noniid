import cPickle as pickle
import numpy as np
import os
import urllib
import tarfile
import zipfile
import sys

def get_cifar_data(dir, option='train'):
    x = None
    y_fine = None

    maybe_download_and_extract(dir)

    train_name = ['data_batch_' + str(i+1) for i in range(5)]
    eval_name = ['test_batch']
    num_fine_classes = 10
    fine_label_key = 'labels'
    coarse_label_key = ''

    folder_name = dir + '/cifar10'
    if option == "train":
        for f_name in train_name:
            trainfile = os.path.join(folder_name, f_name)
            with open(trainfile, 'rb') as f:
                datadict = pickle.load(f)
                _x = datadict.get("data")
                _x = np.array(_x)
                _x = _x.reshape([-1, 3, 32, 32])
                _x = _x.transpose([0, 2, 3, 1])
                _x = _x.reshape(-1, 32, 32, 3)

                _y_fine = np.array(datadict.get(fine_label_key))
                _y_coarse = np.array(datadict.get(coarse_label_key))

            if x is None:
                x = _x
                y_fine = _y_fine
            else:
                x = np.concatenate((x,_x), axis=0)
                y_fine = np.concatenate((y_fine,_y_fine), axis=0)

    elif option == "test":
        for f_name in eval_name:
            evalfile = os.path.join(folder_name, f_name)
            with open(evalfile, 'rb') as f:
                datadict = pickle.load(f)
                _x = datadict.get("data")
                _x = np.array(_x)
                _x = _x.reshape([-1, 3, 32, 32])
                _x = _x.transpose([0, 2, 3, 1])
                x = _x.reshape(-1, 32, 32, 3)

                y_fine = np.array(datadict.get(fine_label_key))

    def dense_to_one_hot(labels_dense, num_classes):
        if num_classes is 0:
            labels_one_hot = None
        else:
            num_labels = labels_dense.shape[0]
            index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    return x, dense_to_one_hot(y_fine, num_classes=num_fine_classes)

def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_download_and_extract(dir):
    main_directory = dir + '/'

    if not os.path.exists(main_directory):
        os.makedirs(main_directory)
        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar = file_path
        file_path, _ = urllib.urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")
        os.rename(main_directory + "./cifar-10-batches-py", main_directory + './cifar10')
        os.remove(zip_cifar)

if __name__ =="__main__":
    pass
