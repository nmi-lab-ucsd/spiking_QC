# File Name : questions_relu.py
# Purpose: Writes the parameters into matlab .mat format for True_North
#
# Author: Emre Neftci, Peter Diehl, Guido Zarrella
#
# Creation Date : 14-07-2015
# Last Modified : Mon 07 Sep 2015 11:32:02 AM UTC
#
# Copyright : Emre Neftci (c) 
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
    print 'Loading ' + sys.argv[1]
    out = save_mat(sys.argv[1])
    
    
    
    
    
    
    
    
    
    
    
