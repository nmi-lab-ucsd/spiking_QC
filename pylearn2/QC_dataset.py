#-----------------------------------------------------------------------------
# File Name : QC_dataset.yaml
# Purpose: pylearn2 yaml script for question classification, Dataset wrapper for the Questions dataset Modified from penn tree bank script
#
# Author: Emre Neftci, Peter Diehl, Guido Zarrella
#
# Creation Date : 14-07-2015
# Last Modified : Mon 07 Sep 2015 11:33:16 AM UTC
#
# Copyright : Emre Neftci (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from functools import wraps
import warnings, os, collections

import numpy as np
from numpy.lib.stride_tricks import as_strided

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.sandbox.nlp.datasets.text import TextDatasetMixin
from pylearn2.utils import serial
from pylearn2.datasets import control
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils.rng import make_np_rng
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.space import IndexSpace, VectorSpace, CompositeSpace
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.sandbox.rnn.utils.iteration import SequenceDatasetIterator

def word2vec(input_filename, output_filename = 'tmp_word2vec.txt'):
    os.system("word2vec -train {0} -output {1} -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15".format(input_filename, output_filename))
    return read_wordvec_to_dict(output_filename)

def read_wordvec_to_dict(input_filename = "tmp_word2vec.txt", trunc_w2v = None, convert_fn=None):
    '''
    Read the output of word2vec program (ascii output) into dictionary
    *trunc_w2v*: truncate word2vec dictionary to first 1000 entries, and use average of remaining word vectors for the remaining (using defaultdict)
    *convert_fn*: arbitrary function that is applied to val
    '''
    d = {}
    keys = []
    vals = []
    with open(input_filename) as f:
        for i,line in enumerate(f):
           (key, val) = line.split(' ',1)
           val = np.fromstring(val,'float',-1,sep=' ')
           if convert_fn is not None:
               val = convert_fn(val)
           vals.append(val)
           keys.append(key)
    avg_vect = np.mean(vals, axis=0)
    if trunc_w2v is None:
        trunc_w2v = len(vals)
    return collections.defaultdict(lambda : avg_vect, zip(keys[:trunc_w2v],vals[:trunc_w2v]))

def normalize_words(input_filename, output_filename):
    os.system("awk '{{print tolower($0);}}' < {0}      | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/\"/ \" /g' -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' -e 's/\;/ \; /g' -e 's/\:/ \: /g' > {1}".format(input_filename, output_filename))

def read_normalized_sentences(input_filename, output_filename = None, which_targets = 'fine'):
    '''
    Normalized with awk
    '''
    fine_tags = []
    coarse_tags = []
    if which_targets == 'fine':
        tags = fine_tags
    else:
        tags = coarse_tags

    allwords = []
    with open(input_filename) as alldata:
        for line_no, line in enumerate(alldata):
            tokens = line.strip().split(' ')
            words = tokens[1:]
            coarse_label, fine_label = tokens[0].split(':')
            fine_tags.append( fine_label) 
            coarse_tags.append( coarse_label) 
            allwords.append(' '.join(words[:-1])+'\n')

    with open('/tmp/sentences','w') as fh:
        fh.writelines(allwords)

    if output_filename == None:
        normalize_words('/tmp/sentences', '/tmp/nse')
        with open('/tmp/nse', 'r') as fh:
            return fh.readlines(), tags
    else:
        normalize_words('/tmp/sentences', output_filename)

def sentence_to_vector_sequence(sentence, wordvec_dict):
    '''
    Use word2vec to transform a sentence into sequences of vectors
    '''
    words = sentence.split(' ') 
    return np.array([wordvec_dict[w] for w in words])

def pad_square(data_list, length=None):
    n_samples = len(data_list)
    if length is None:
        length = np.max([len(d) for d in data_list])
    ret = np.zeros([n_samples,length,data_list[0][0].shape[0]])
    for i,d in enumerate(data_list):
        if len(d)>0:
            a = min(len(d),length)
            ret[i,:a,:] = d[:a]
    return ret


def permute_matrix(vector, shape=None):
    import scipy.sparse as sparse
    indptr = range(vector.shape[0]+1)
    ones = np.ones(vector.shape[0])
    if shape is None:
        permut = sparse.csr_matrix((ones, vector, indptr))
    else:
        permut = sparse.csr_matrix((ones, vector, indptr), shape = (vector.shape[0], shape))
    return permut.toarray()

class Questions(VectorSpacesDataset):
    """
    Parameters
    ----------
    which_set : {'train', 'test'}
        Choose the set to use
    context_len : int
        The size of the context i.e. the number of words or chars used
        to predict the subsequent word.
    """
    def __init__(self, which_set, which_targets, word2vec_dict={}, eos = True, one_hot_label=True, padding = False, debug=False):
        self.eos = eos
        self.dim_features = len(word2vec_dict.values()[0])
        self.eos_vector = np.zeros(self.dim_features)
        self.debug = debug
        self.padding = padding
        self.one_hot_label = one_hot_label
        # Load data from disk
        self.sequences, self.labels = self._load_data(which_set, which_targets, word2vec_dict)
        # Standardize data

        self.num_words = np.max([len(sequence) for sequence
                                        in self.sequences]) + 1
        self.num_examples = len(self.sequences)

        self.data = (self.sequences, self.labels)

        source = ('features', 'targets')
        space = CompositeSpace([
            SequenceDataSpace(VectorSpace(dim=self.dim_features)),
            SequenceDataSpace(VectorSpace(dim=self.num_categories)),
        ])


        super(Questions, self).__init__(
            data=(self.data),
            data_specs=(space, source)
        )

    def get_data(self):
        return self.data

    def _load_path(self, which_set, which_targets, word2vec_dict={}):
        if which_targets not in ['fine', 'coarse']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["fine","coarse"].')

        if which_set not in ['train', 'test']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        if control.get_load_data():
            path = "${PYLEARN2_DATA_PATH}/TREC_question_type_data/"
            if which_set == 'train':
                data_path = path + 'trecqc.train_5500.label.txt'
            else:
                assert which_set == 'test'
                data_path = path + 'trecqc.test_500.label.txt'
            data_path = serial.preprocess(data_path)
        self.path = path
        return data_path

    def _load_data(self, which_set, which_targets, word2vec_dict={}):
        #Load path
        data_path = self._load_path(which_set, which_targets, word2vec_dict)
        data, labels = read_normalized_sentences(data_path, which_targets = which_targets)
        #Save sentences for validation/debug
        if self.debug:
            self.sentences = data
        all_labels = np.unique(labels)

        #Use integers instead of strings for labels, and map labels
        self.num_categories = len(np.unique(labels))+0
        self.labels_dict = labels_dict = dict(zip(all_labels,0+np.arange(len(all_labels)))) #0+ so that 0 is the not care label
        self.labels_int = labels_int = np.array(map(labels_dict.get, labels)).reshape(-1,1)

        #Create data and label vector with not care = 0 labels for all but the last word in the sentence.
        tmp_data = []
        tmp_label = []
        for i,s in enumerate(data):
            data_vector = sentence_to_vector_sequence(s, word2vec_dict)
            if self.eos:
                data_vector = np.row_stack([data_vector, self.eos_vector])
            tmp_data.append(data_vector)
            #Add in a slot for the label
            #Create row of label neurons
            label_vector = np.zeros([len(data_vector), self.num_categories])
            label_vector[-1,labels_int[i]] = 1
            tmp_label.append(label_vector)
            #Concatenate all and yield
            #samples.append(data_labels)

        #Padding with zeros to obtain a square input matrix, not necessary for SequenceDataSpace class
        if self.padding:
            raise NotImplementedError
            #data_w2v = pad_square(data_w2v)

        return np.array(tmp_data), np.array(tmp_label)

    def _create_subset_iterator(self, mode, batch_size=None, num_batches=None,
                                rng=None):
        subset_iterator = resolve_iterator_class(mode)
        if rng is None and subset_iterator.stochastic:
            rng = make_np_rng()
        return subset_iterator(self.get_num_examples(), batch_size,
                               num_batches, rng)

    @wraps(VectorSpacesDataset.iterator)
    def iterator(self, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False, mode=None):
        subset_iterator = self._create_subset_iterator(
            mode=mode, batch_size=batch_size, num_batches=num_batches, rng=rng
        )
        # This should be fixed to allow iteration with default data_specs
        # i.e. add a mask automatically maybe?
        if data_specs is None:
            data_specs = self.data_specs
        return SequenceDatasetIterator(self, data_specs, subset_iterator,
                                       return_tuple = return_tuple)

class QuestionsSpikeTimes(Questions):
    def __init__(self, which_set, which_targets, word2vec_dict, debug =False):
        super(QuestionsSpikeTimes, self).__init__( which_set, which_targets, word2vec_dict, padding = False, one_hot_label=False, debug = debug, eos = True)

    def _load_data(self, which_set, which_targets, word2vec_dict):
        data, labels = super(QuestionsSpikeTimes, self)._load_data(which_set, which_targets, word2vec_dict)
        data = np.array([ np.round(dd) for dd in data])
        labels = np.array([ np.round(dd) for dd in labels])
        return data, labels

    def iterator_spiketrains(self, n_samples = None, target_duration=16, t_per_word=17, **kwargs):
        '''
        kwargs passed to iterator
        '''
        if not kwargs.has_key('mode'):
            kwargs['mode'] = 'sequential'
        if n_samples is None:
            n_samples = len(self.labels)
        it = self.iterator(batch_size = 1, num_batches = n_samples, **kwargs)
        samples = []
        dim_total = self.dim_features + self.num_categories
        #For loop instead of while to avoid the need of catching stopiteration exception
        for _ in range(n_samples):
            sentence_data, label_data = it.next()
            data_words = []
            label_words = []
            #Create temporal code. While data component is temporal code, label is rate code
            #This gives the following wierd transformations
            for i,data in enumerate(sentence_data):
                data = permute_matrix(data.reshape(self.dim_features), t_per_word)
                data_words.append(data)            
            #No need to have a end of sentence vector
            data_words[-1] *= 0

            for i,label in enumerate(label_data):
                label_words.append(np.zeros([label.shape[1], t_per_word]))

            #Extract the label
            label_id = np.nonzero(label[0])[0]
            label_words[-1][label_id, :target_duration] = 1            
            #Concatenate all the words in the sentence
            sentence = np.column_stack([a for a in data_words])
            target = np.column_stack([a for a in label_words])
            #Create row of label neurons
            #Concatenate all and yield
            yield np.row_stack([sentence, target])
            #samples.append(data_labels)

def read_word2vec_truncated(input_filename, steps = 16, trunc_w2v = 1000, trunc_min=-1, trunc_max=1):
    '''
    Read the output of word2vec program (ascii output) into dictionary and truncate vector elements to [trunc_min, trunc_mas] and discretize to *steps* levels.
    *steps*: number of discretization steps
    *trunc_w2v*: truncate word2vec dictionary to first 1000 entries, and use average of remaining word vectors for the remaining (using defaultdict)
    *trunc_w2v*: truncate word2vec dictionary to first 1000 entries, and use average of remaining word vectors for the remaining (using defaultdict)
    *trunc_min, truncmax*: min max to truncate at
    '''
    w2v_range = trunc_max - trunc_min
    scale = steps/w2v_range                     
    def fn(val):
        if trunc_min is not None:
            val = np.maximum(trunc_min, val)
        if trunc_max is not None:
            val = np.minimum(trunc_max, val)
        return (val-trunc_min)*scale
    return read_wordvec_to_dict(input_filename, trunc_w2v, convert_fn = fn)

if __name__ == '__main__':
    print 'Reading word vectors ...'
    data_path = "${PYLEARN2_DATA_PATH}/word2vec/basic.wikiw2v.64.ascii"
    prep_data_path = serial.preprocess(data_path)
    d = read_wordvec_to_dict(input_filename = prep_data_path)  

    print 'Building questions train dataset ...'
    questions_train = Questions('train','coarse',d)

    print 'Pickling ...'
    serial.save('questions_train.pkl', questions_train)

    print 'Building questions test dataset ...'
    questions_test = Questions('test','coarse',d)

    print 'Pickle ...'
    serial.save('questions_test.pkl', questions_test)   
