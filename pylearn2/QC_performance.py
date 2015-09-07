#!/bin/python
#-----------------------------------------------------------------------------
# File Name : QC_performance.py
# Purpose: Script for testing performance of QC using a discretized ReLU that would mimic a digital spiking neuron
#
# Author: Emre Neftci
#
# Creation Date : 14-07-2015
# Last Modified : Mon 07 Sep 2015 11:33:26 AM UTC
#
# Copyright : Emre Neftci (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import sys
import os
import argparse
import numpy as np
from QC_dataset import Questions

from pylearn2.utils import serial
from theano import tensor as T
from theano import function


import pickle
train_data = pickle.load(file('questions_train.pkl','r'))
test_data = pickle.load(file('questions_test.pkl','r'))


model = serial.load(sys.argv[1])
X = model.get_input_space().make_theano_batch()
Y = model.fprop(X)

f = function(X, T.argmax(Y[0],axis = 2)[-1][0], allow_input_downcast=True)

a=[]
b=[]
train_it = train_data.iterator(batch_size=1, mode='sequential')
while True:
    try:
        cu,ci = train_it.next()
        a.append(int(f(cu,np.ones([100,1]))))
        b.append(int(np.argmax(ci[-1],axis=1)))
    except StopIteration:
        break

print "Training data performance"
print np.mean(np.array(a)==np.array(b))

a=[]
b=[]
test_it = test_data.iterator(batch_size=1, mode='sequential')
while True:
    try:
        cu,ci = test_it.next()
        a.append(int(f(cu,np.ones([100,1]))))
        b.append(int(np.argmax(ci[-1],axis=1)))
    except StopIteration:
        break

print "Testing data performance"
print np.mean(np.array(a)==np.array(b))

