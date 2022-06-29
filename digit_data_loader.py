# encoding=utf-8
"""
    Created on 10:35 2018/12/29
    @author: Jindong Wang
"""

import gzip
import pickle
import struct

from scipy.io import loadmat, savemat

import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch


## For loading datasets of MNIST, USPS, and SVHN.


class GetDataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, data, label,
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """

        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(img.shape)
        if img.shape[0] != 1:
            # print(img)
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))
        #
        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # print(np.vstack([im,im,im]).shape)
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
            #  return img, target
        return img, target

    def __len__(self):
        return len(self.data)


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot


def load_mnist(mnist_dir=str):
    train_data_path = mnist_dir + 'train-images-idx3-ubyte'
    train_label_path = mnist_dir + 'train-labels-idx1-ubyte'
    test_data_path = mnist_dir + 't10k-images-idx3-ubyte'
    test_label_path = mnist_dir + 't10k-labels-idx1-ubyte'

    # train_img
    with open(train_data_path, 'rb') as f:
        data = f.read(16)
        des, img_nums, row, col = struct.unpack_from(
            '>IIII', data, 0)
        train_x = np.zeros((img_nums, row * col))
        for index in range(img_nums):
            data = f.read(784)
            if len(data) == 784:
                train_x[index, :] = np.array(
                    struct.unpack_from(
                        '>' + 'B' * (row * col), data,
                        0)).reshape(1, 784)
        f.close()
    # train label
    with open(train_label_path, 'rb') as f:
        data = f.read(8)
        des, label_nums = struct.unpack_from('>II', data, 0)
        train_y = np.zeros((label_nums, 1))
        for index in range(label_nums):
            data = f.read(1)
            train_y[index, :] = np.array(
                struct.unpack_from('>B', data, 0)).reshape(
                1, 1)
        f.close()

    # test_img
    with open(test_data_path, 'rb') as f:
        data = f.read(16)
        des, img_nums, row, col = struct.unpack_from(
            '>IIII', data, 0)
        test_x = np.zeros((img_nums, row * col))
        for index in range(img_nums):
            data = f.read(784)
            if len(data) == 784:
                test_x[index, :] = np.array(
                    struct.unpack_from(
                        '>' + 'B' * (row * col), data,
                        0)).reshape(1, 784)
        f.close()
    # test label
    with open(test_label_path, 'rb') as f:
        data = f.read(8)
        des, label_nums = struct.unpack_from('>II',
                                             data, 0)
        test_y = np.zeros((label_nums, 1))
        for index in range(label_nums):
            data = f.read(1)
            test_y[index, :] = np.array(
                struct.unpack_from('>B', data,
                                   0)).reshape(1, 1)
        f.close()

    img_train = train_x.reshape(
        (train_x.shape[0], 1, 28, 28))
    img_test = test_x.reshape(
        (test_x.shape[0], 1, 28, 28))
    train_y=train_y.astype('int64')
    test_y=test_y.astype('int64')

    label_train = train_y.reshape(-1)
    label_test = test_y.reshape(-1)
    return img_train, label_train, img_test, label_test

def load_svhn(path_train, path_test):
    svhn_train =loadmat(path_train)
    svhn_test = loadmat(path_test)
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label=svhn_train['y'].reshape(-1)
    svhn_label[np.where(svhn_label == 10)] = 0
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = svhn_test['y'].reshape(-1)
    svhn_label_test[np.where(svhn_label_test == 10)] = 0
    svhn_label=svhn_label.astype('int64')
    svhn_label_test=svhn_label_test.astype('int64')
    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test

def load_usps(path, all_use=True):
    f = gzip.open(path, 'rb')
    data_set = pickle.load(f, encoding='bytes')
    f.close()
    img_train = data_set[0][0]      #7438,1,28,28)
    label_train = data_set[0][1]    #7438
    img_test = data_set[1][0]       #1860,1,28,28
    label_test = data_set[1][1]

    label_train=label_train.astype('int64')
    label_test = label_test.astype('int64')
    inds = np.random.permutation(img_train.shape[0])
    if all_use :
        img_train = img_train[inds][:]
        label_train = label_train[inds][:]
    else:
        img_train = img_train[inds][:]
        label_train = label_train[inds][:]
    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))
    return img_train, label_train, img_test, label_test

def load_dataset( root_dir):
    data = loadmat(root_dir+'.mat')
    train_img = data['train_img']
    train_label = data['train_label'].reshape(-1)
    test_img = data['test_img']
    test_label = data['test_label'].reshape(-1)
    return train_img, train_label, test_img, test_label


def load_digit_data( src,tar, batch_size):
    src_train_img, src_train_label, src_test_img, src_test_label = load_dataset(src)
    tar_train_img, tar_train_label, tar_test_img, tar_test_label = load_dataset(tar)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_src_train, data_src_test = GetDataset(src_train_img, src_train_label,
                                               transform), GetDataset(src_test_img,
                                                                      src_test_label,
                                                                      transform)
    data_tar_train, data_tar_test = GetDataset(tar_train_img, tar_train_label,
                                               transform), GetDataset(tar_test_img,
                                                                      tar_test_label,
                                                                      transform)

    source_loader = torch.utils.data.DataLoader(data_src_train, batch_size=batch_size, shuffle=True,
                                                     drop_last=True,
                                                     num_workers=4)

    target_train_loader = torch.utils.data.DataLoader(data_tar_train, batch_size=batch_size, shuffle=True,
                                                     drop_last=True,
                                                     num_workers=4)
    target_test_loader = torch.utils.data.DataLoader(data_tar_test, batch_size=batch_size, shuffle=True,
                                                     drop_last=False,
                                                     num_workers=4)
    return source_loader, target_train_loader,target_test_loader


def save_data2mat(root_dir='./data/digits/'):
    data=load_mnist(root_dir)
    savemat(root_dir+'mnist.mat',{'train_img':data[0],
                         'train_label':data[1],
                         'test_img':data[2],
                         'test_label':data[3]})
    data =load_usps(root_dir + 'usps_28x28.pkl')
    savemat(root_dir+'usps.mat',{'train_img':data[0],
                         'train_label':data[1],
                         'test_img':data[2],
                         'test_label':data[3]})
    data = load_svhn(root_dir + 'svhn_train_32x32.mat',
            root_dir + 'svhn_test_32x32.mat')
    savemat(root_dir+'svhn.mat',{'train_img':data[0],
                         'train_label':data[1],
                         'test_img':data[2],
                         'test_label':data[3]})

#save_data2mat()
