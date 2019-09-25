import numpy as np

def generate_nonIID(images, labels):
    num_class = 10

    sorted_images = [[] for _ in range(num_class)]
    sorted_labels = [[] for _ in range(num_class)]

    for idx, val in enumerate(labels):
        sorted_labels[np.where(val==1)[0][0]].append(val)
        sorted_images[np.where(val==1)[0][0]].append(images[idx])

    for i in range(num_class):
        sorted_images[i] = np.asarray(sorted_images[i])
        sorted_labels[i] = np.asarray(sorted_labels[i])

    clients_images = np.asarray(sorted_images)
    clients_labels = np.asarray(sorted_labels)
    return clients_images, clients_labels

if __name__ =="__main__":
    pass