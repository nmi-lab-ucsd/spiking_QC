#!/bin/python
#-----------------------------------------------------------------------------
# File Name : questions_relu.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 14-07-2015
# Last Modified : Fri 17 Jul 2015 06:58:29 AM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 

import sys, cPickle
import numpy as np
from scipy.io import loadmat, savemat
from pylearn2.utils import serial
from theano import tensor as T
from theano import function

round_it = True

#{'ABBR': 0, 'DESC': 1, 'ENTY': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5}
#Gain computed for the scale = .35, grid = 16 case
bd = 1.0
B = 16/bd #Scaling of weights
A = 18 #Scaling of firing rates


'''
questions_nobias.pkl: no biases, post rounding, 16 levels, A=20, stepify = 15, scale = .5, result: 0.65
'''

def softmax(w):
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1)
    return dist

def round_model_inplace(weight_dict, bd = None, nlvls=16):
    if bd == None:
        bd = .5
    lgrid = nlvls/(bd*2)
    for k,o in weight_dict.iteritems():
        if k[0] != '_':
            o[:] = np.round(o*lgrid)/lgrid
            o[:] = np.clip(o, -bd, bd-2*bd/nlvls)


out = loadmat(sys.argv[1])

if round_it:
    round_model_inplace(out, bd=bd)

#Check that there are no biases
assert np.prod(out['recurrent_layer_b'] == 0)==1
assert np.prod(out['softmax_b'] == 0)==1
assert np.prod(out['projection_layer_b'] == 0)==1




for k,o in out.iteritems():
    if k[0] != '_':
        o[:] *= B 

def stepify(x, bd=15):        
        return np.maximum(-bd,np.minimum(bd, np.floor(x)))

def relu(x):
    return stepify(np.maximum(0,  x)*1/B)
    #return np.maximum(0,  x)*1/B

def fprop(u, rout, parms):
    pout = relu(np.dot(u*A,   parms['projection_layer_W'])+parms['projection_layer_b'])
    return relu(np.dot(pout,parms['recurrent_layer_W']) + np.dot(rout,parms['recurrent_layer_U']) + parms['recurrent_layer_b']), pout

#For testing/validating
#model = serial.load('questions_73_78_.35_16_48.pkl')
#X = model.get_input_space().make_theano_batch()
#Y = model.fprop(X)
#f = function(X, Y[0], allow_input_downcast=True)

test_data = cPickle.load(file('questions_test.pkl','r'))
test_it = test_data.iterator(batch_size=1, mode='sequential')
    
a = []
b = []
ca = []
test_data, test_labels = [], []
k = 0
while True:
    try:
        cu,ci = test_it.next()
        test_labels.append(int(np.argmax(ci[-1],axis=1)))
        rout = np.zeros(out['recurrent_layer_b'].shape[1])
        sentence = []
        for i, c in enumerate(cu):
            rout,pout = fprop(c, rout, out)             
            sentence.append(pout[0])
            if k == 0:
                ca.append(rout[0])
        o = np.dot(rout,out['softmax_W']) + out['softmax_b'] 
        test_data.append(np.array(sentence))
        res = np.argmax(o)
        a.append(res)
        b.append(int(np.argmax(ci[-1],axis=1)))
        k += 1
    except StopIteration:
        break

print "Testing data performance"
print np.mean(np.array(a)==np.array(b))

matlab_out = out.copy()

matlab_out['test_data']=test_data
matlab_out['test_labels']=test_labels

savemat('weights_4bit.mat', matlab_out)
