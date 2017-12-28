import numpy as np
import cv2

class ImageDataGenerator:
    def __init__(self, class_list, horizontal_flip=False, shuffle=False,
                 mean=np.array([79., 89., 117.]), scale_size=(128, 59),
                 nb_classes=1780):
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0

        self.read_class_list(class_list)
        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, class_list):
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                self.labels.append(int(items[1]))
            self.data_size = len(self.labels)

            self.data = np.ndarray([self.data_size, self.scale_size[0], self.scale_size[1], 3])
            for i in range(len(self.images)):
                img = cv2.imread(self.images[i])
                if self.horizontal_flip and np.random.random() < 0.5:
                    img = cv2.flip(img, 1)
                # img = cv2.resize(img, (self.scale_size[1], self.scale_size[0]))
                # img = img.astype(np.float32)
                # img -= self.mean
                self.data[i] = img


    def shuffle_data(self):
        data = self.data.copy()
        labels = self.labels.copy()
        self.data = []
        self.labels = []
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.data.append(data[i])
            self.labels.append(labels[i])


    def reset_pointer(self):
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        images = self.data[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]

        # update pointer
        self.pointer += batch_size

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        # return array of images and labels
        return images, one_hot_labels,labels













