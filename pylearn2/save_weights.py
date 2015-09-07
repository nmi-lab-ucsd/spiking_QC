# File Name : questions_relu.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 14-07-2015
# Last Modified : Tue 14 Jul 2015 02:34:49 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 

import numpy as np
from pylearn2.utils import serial
from scipy.io import savemat
import sys

def save_mat(model_path=None,
                       model=None,
                       rescale='none',
                       border=False,
                       norm_sort=False,
                       dataset=None):
    """
    Saves all weights in a .mat file
    
    Parameters
    ----------
    model_path : str
        Filepath of the model to make the report on.
    """

    if model is None:
        print 'loading model'
        model = serial.load(model_path)
        print 'loading done'

    weight_dict = {}

    for layer in model.layers:
        print layer.layer_name
        for i, p in enumerate(layer.get_params()):
            weight_dict[str(p)] = layer.get_param_values()[i]
        
    savemat('weights.mat', weight_dict)
    return weight_dict
    
    

if __name__ == '__main__':
    print sys.argv[1]
    out = save_mat(sys.argv[1])
    print out
    
    
    
    
    
    
    
    
    
    
    
