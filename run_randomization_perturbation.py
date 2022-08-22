from __future__ import print_function

import tensorflow as tf



import tensorflow as tf
import basenji
import sys
sys.path.append('basenji/')


from optparse import OptionParser
import json
import os
import pdb
import sys
import time

import h5py
from intervaltree import IntervalTree
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import tensorflow as tf

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

from basenji import bed
from basenji import dataset
from basenji import plots
from basenji import seqnn
from basenji import trainer
def make_groundtruth_gen(seqs_dna):
    for seq in seqs_dna:
        seq_1hot = dna_io.dna_1hot(seq)
        yield seq_1hot
        
# Randomize the one-hot encoding of 'vec' from start -> end. Randomizes #(replicates) times. 
# We can also consider maintaining the same distribution of nucleotides, or only shuffling the sequence. 
def np_randomize(vec, start, end, replicates, p_dist=[1/4]*4):
    """ randomize the vector from ind=start to ind=end to generate a new vector. 'replicates' represents the 
	number of times this is done.
   """
    # p_dist is a multinomial over new bases
    # ps is of shape (4, replicates)
    delsize = end-start
    ps = arr([p_dist]*4)
    print(vec.shape)
    assert (np.sum(ps, axis = 1) == 1).all()
    assert ps.shape[1] == 4
    assert vec.shape[1] == 4
    assert vec.shape[0] >= end
    assert start >= 0
    output = np.tile(vec, (replicates, 1, 1))
    ps = arr([1/4]*4)
    deletions = np.random.multinomial(1, ps, (delsize, replicates))
    deletions = deletions.swapaxes(1, 0)
    output[:, start:end, :] = 0
    output[:, start:end, :] = deletions
    return output
def get_tile_inds(seqlength, windowsize, total_length=6400*4):
    """ For a given windowsize, get startinds and endinds for non-overlapping windows which together cover
        a size of total_length.
    """
    assert seqlength%2 == 0 
    assert seqlength%(2*windowsize) == 0 
    assert total_length%(2*windowsize) == 0 
    
    side_inds = (total_length//windowsize)//2
    
    all_startinds = np.arange(0, seqlength, windowsize)
    
    n = len(all_startinds)
    midpt = n//2
    L = all_startinds[midpt : midpt+side_inds]
    R = all_startinds[midpt-side_inds: midpt]
    assert len(L) == len(R)
    startinds = all_startinds[midpt-side_inds : midpt+side_inds]
    endinds = startinds + windowsize
    n_tiles = len(startinds)
    return np.stack([startinds, endinds]), n_tiles
def rand_mut_gen(dnaseq, tile_inds, n_tiles, replicates=100):
    """Construct generator for randomization of vector at 
        all tiles. replicates variable represents number of randomized new sequences."""
    seq_1hot = dna_io.dna_1hot(dnaseq)
    #yield seq_1hot
    # for mutation positions
    for tile in range(n_tiles):
        start_ind = tile_inds[0, tile]
        end_ind = tile_inds[1, tile]
        randvecs = np_randomize(seq_1hot, start_ind, end_ind, replicates)
        for randvec in randvecs:
            yield randvec

windowsizes = [6400]

out_dir = '/Genomics/pritykinlab/zzhao/HiC/drosophila_data_output'
params_file = f'{out_dir}/params_tutorial.json'
with open(params_file) as params_open:
    params = json.load(params_open)
model_params = params['model']
seqnn_model = seqnn.SeqNN(model_params)
model_file = f'{out_dir}/model_best.h5'
seqnn_model.restore(model_file)

tf.config.run_functions_eagerly(True)

replicates = 1000
# output the scores (MSEs) to a file 
scores_h5_file = f'./gabe_scores.h5'
if os.path.isfile(scores_h5_file):
    os.remove(scores_h5_file)

scores_h5 = h5py.File(scores_h5_file, 'w')
for window in windowsizes:
    _, n_tiles = get_tile_inds(seqlength, window)
    n = len(dnaseq)
    scores_h5.create_dataset(f'{window}', dtype='float16',
          shape=(n, n_tiles, replicates))
    # The scores file has a unique dataset for each windowsize. This is of size (n_seqs, n_tiles, replicates) 
    # since each seq has n_tiles to cover the sequence with the windowsize; each tile is repeated replicates number of times


batch_size = 64
region_dict = {}
bedpath = 'suHw' # This is the bedfile with regions of interest
gpath = './nochr.dm6.fa'
seqs_dna, seqs_coords = bed.make_bed_seqs(
    bedpath, gpath, model_params['seq_length'], stranded=False)

num_seqs = len(seqs_dna)
groundtruth_gen = make_groundtruth_gen(seqs_dna)
groundtruth_stream = stream.PredStreamGen(seqnn_model, groundtruth_gen, batch_size)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from time import time
from sklearn.metrics import mean_squared_error
from itertools import chain
# make sequence generator
t1 = time()
for c, dnaseq in enumerate(seqs_dna):
    groundtruth = groundtruth_stream[c]    
    for window in windowsizes[::-1]:
        all_windowsizes = []
        all_tile_inds = []
        
        tile_inds, n_tiles = get_tile_inds(seqlength, window)
        seqs_gen = rand_mut_gen(dnaseq, tile_inds, n_tiles, replicates=replicates)
        all_windowsizes.append(window)
        all_tile_inds.append(tile_inds[0, :])
        preds_stream = stream.PredStreamGen(seqnn_model, seqs_gen, batch_size)
        deltas = []
        for ind in range(0, n_tiles*replicates):
            deltas.append(mean_squared_error(preds_stream[ind], groundtruth))
        deltas = arr(deltas).reshape(n_tiles, replicates)
        scores_h5[f'{window}'][c, :, :] = deltas
        print("Finished one loop!")
        break
    break
t1f = time()
print('DONE', t1f-t1)




