#!/usr/bin/env python2
#-----------------------------------------------------------------------------
# File Name : train.py
# Purpose: pylearn2 train script based on pylearn2-train.py
#
# Author: Emre Neftci
#
# Creation Date : 14-07-2015
# Last Modified : Mon 07 Sep 2015 10:37:08 AM UTC
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
# -*- encoding: utf-8 -*-
from theano.compat.six import exec_

#This is the only difference: we must import Questions because the data is pickeled as Questions objects.
from QC_dataset import Questions

from pylearn2.scripts import train
filename = train.__file__
f = open(filename[:-1 if filename.endswith('.pyc') else None])
exec_(f.read())

