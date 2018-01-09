# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:40:28 2018

@author: kim
"""

import numpy as np
import cupy as cp
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import matplotlib.pyplot as plt
from chainer.cuda import to_cpu
from chainer.dataset import concat_examples
from chainer.datasets import cifar


class ResNet34(chainer.Chain):
    def __init__(self, n_block_num = 5):
        w = chainer.initializers.HeNormal()
        super(ResNet34, self).__init__(
                conv1 = L.Convolution2D(None, 16, 3, 1, 2, initialW=w, nobias = True),
                bn1 = L.BatchNormalization(16),
                res2 = ResBlock(n_block_num, 16, 16, 1),
                res3 = ResBlock(n_block_num, 32, 16),
                res4 = ResBlock(n_block_num, 64, 32),
                fc5 = L.Linear(64, 10))
                
    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = F.average_pooling_2d(h, h.shape[2:], stride = 1)
        h = self.fc5(h)
        if chainer.config.train:
            return h
        return F.softmax(h)
    
class ResBlock(chainer.ChainList):
    def __init__(self, n_layers, n_number, n_pre, stride=2):
        w = chainer.initializers.HeNormal()
        super(ResBlock, self).__init__()
        self.add_link(ResRes(n_number, n_pre, stride, True))
        for _ in range(n_layers - 1):
            self.add_link(ResRes(n_number, n_pre))
            
    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x

class ResRes(chainer.Chain):
    def __init__(self, n_number, n_pre, stride=1, proj=False):
        w = chainer.initializers.HeNormal()
        super(ResRes, self).__init__()
        with self.init_scope():
            self.conva = L.Convolution2D(None, n_number, 3, stride, 1, initialW=w, nobias=True)
            self.convb = L.Convolution2D(n_number, n_number, 3, 1, 1, initialW=w, nobias=True)
            self.bn_a = L.BatchNormalization(n_number)
            self.bn_b = L.BatchNormalization(n_number)
            if proj:
                self.convr = L.Convolution2D(n_pre, n_number, 1, stride, 0, initialW=w, nobias=True)
                self.bn_r = L.BatchNormalization(n_number)
        self.proj = proj
        
    def __call__(self, x):
        h = F.relu(self.bn_a(self.conva(x)))
        h = self.bn_b(self.convb(h))
        if self.proj:
            x = self.bn_r(self.convr(x))
        return F.relu(h+x)
    
    
train,test = cifar.get_cifar10()

train_iter = iterators.SerialIterator(train, 128)
test_iter = iterators.SerialIterator(test, 2000, repeat=False, shuffle=False)

gpu_id = 0

Itrain, Ttrain = concat_examples(train,gpu_id)
Train_mean = cp.mean(Itrain)
Train_std = cp.std(Itrain)

model = ResNet34().to_gpu(gpu_id)

optimizer = optimizers.MomentumSGD(lr=0.1, momentum = 0.9)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

max_epoch = 50000

plt.figure(figsize=(7,5))

test_acc_array = []
train_acc_array = []

while train_iter.epoch < max_epoch:
    train_accuracies = []
    
    # ----------------- training loop 한 주기 설정----------------------
    train_batch = train_iter.next()
    image_train, target_train = concat_examples(train_batch, gpu_id)
    
#    image_train = (image_train - Train_mean)
    image_train = (image_train - Train_mean) / Train_std
    
    Iarray_pad = cp.pad(image_train,((0,0),(0,0),(4,4),(4,4)),mode='constant')
    Iarray_flip = cp.flip(Iarray_pad,1)
    a = [0,1]
    filp_flag = cp.random.choice(a,128)
    b = [0,1,2,3,4,5,6,7]
    c = [0,1,2,3,4,5,6,7]
    height_rand = cp.random.choice(b,128)
    width_rand = cp.random.choice(c,128)
    image_train_dataAug = cp.zeros([Iarray.shape[0],3,32,32],'float32')
    
    for i in range(0,image_train.shape[0]-1):
        height = height_rand[i]
        width = width_rand[i]
        if filp_flag[i] == 0 :
            image_train_dataAug[i,:,:,:] = Iarray_pad[i,:,height:height+32,width:width+32]
        else:
            image_train_dataAug[i,:,:,:] = Iarray_flip[i,:,height:height+32,width:width+32]
    # 초반 Network 결과를 예측
#    prediction_train = model(image_train)
    prediction_train = model(image_train_dataAug)
    
    # loss 계산
    train_accuracy = F.accuracy(prediction_train, target_train)
    train_accuracies.append(train_accuracy.data)
    loss = F.softmax_cross_entropy(prediction_train, target_train)
    
    #Gradient 계산
    model.cleargrads()
    loss.backward()
    
    # update
    optimizer.update()
    #---------------------------여기까지------------------------------
    if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

        test_losses = []
        test_accuracies = []
        
        while True:
            test_batch = test_iter.next()
            image_test, target_test = concat_examples(test_batch, gpu_id)
            image_test = (image_test - Train_mean) / Train_std

            # Forward the test data
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                prediction_test = model(image_test)

            # Calculate the loss
            loss_test = F.softmax_cross_entropy(prediction_test, target_test)
            test_losses.append(loss_test.data)

            # Calculate the accuracy
            
            test_accuracy = F.accuracy(prediction_test, target_test)
            test_accuracies.append(test_accuracy.data)

            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break
            
        train_accuracy = cuda.to_cpu(cp.stack(train_accuracies).mean())
        train_acc_array.append(train_accuracy)
        test_loss = cuda.to_cpu(cp.stack(test_losses).mean())
        test_accuracy = cuda.to_cpu(cp.stack(test_accuracies).mean())
        test_acc_array.append(test_accuracy)
        
        print('epoch:{:02d} train_loss:{:.04f} train_acc:{:.04f}'.format(train_iter.epoch, float(to_cpu(loss.data)), float(train_accuracy)), end=' ')
        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(float(test_loss), float(test_accuracy)))
#        
#        if (train_iter.epoch % 10) == 0 :
        plt.plot(train_acc_array)
        plt.plot(test_acc_array)
        plt.legend(["train_acc", "test_acc"], loc=4)
        plt.title("Accuracy of digit recognition.")
        plt.grid()
        plt.show()  
        
        