
# coding: utf-8

#latest update: enabled support for cross entropy loss (CE), weighted CE, and focal loss, suuport for multiple optimizers
#previous update: implement support to parallel processing of input data

#Batch mode example
#python3.6 ../10-fold_CV_imputation_autoencoder_from_grid_search_v3_online_data_augmentation_parallel.py ../HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf.clean3.subset1000 best_hp_set.txt False 0 0
#/bin/python3.6 script_name.py imputed_test_subset.vcf 3_hyper_par_set.txt False 0 0
#CUDA_VISIBLE_DEVICES=0 /bin/python3.6 ../script_name.py imputed_test_subset.vcf 3_hyper_par_set.txt True 0 0

#sequential mode
#python3.6 ../10-fold_CV_imputation_autoencoder_from_grid_search_v3_online_data_augmentation_parallel.py ../HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf.clean4 0.039497493748 0.001096668917 0.001 0.021708661247 sigmoid 5.6489904e-05 3 Adam FL False 0 0

import math #sqrt

import tensorflow as tf

#from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import itertools

import pandas as pd

import random #masking

# sorting results
from collections import defaultdict
from operator import itemgetter

import timeit #measure runtime

from tqdm import tqdm # progress bar

from scipy.stats import pearsonr #remove this, nan bugs are not corrected

from tensorflow.python.client import device_lib #get_available_devices()

from joblib import Parallel, delayed #parallel tasks
import multiprocessing #parallel tasks

from sklearn.model_selection import KFold

from scipy.stats import linregress
#linregress(a, b)[2] gives the correlation, correcting for nans due to monomorphic predictions
#linregress(a, b)[3] gives the p-value correl 
#a are all the predicted genotypes for one given SNP, in ALT dosage form, b is the same but for experimental SNPs

import sys #arguments
import operator #remove entire columns from 2d arrays

from functools import partial # pool.map with multiple args
import subprocess as sp #run bash commands that are much faster than in python (i.e cut, grep, awk, etc)

from minepy import pstats, cstats

###################################OPTIONS#############################################

#Performance options
do_parallel = True #load data and preprocess it in parallel
do_parallel_masking = True #also do the masking in parallel
do_parallel_MAF = True #also do MAF calculation in parallel
use_cuDF = False #TODO enable data loading directly in the GPU


#backup options and reporting options
save_model = False #[True,False] save final model generated after all training epochs, generating a checkpoint that can be loaded and continued later
save_pred = False #[True,False] save predicted results for each training epoch and for k-fold CV
resuming_step = 1001 #number of first step if recovery mode True
save_summaries = False #save summaries for plotting in tensorboard
detailed_metrics = False #time consuming, calculate additional accuracy metrics for every variant and MAF threshold
report_perf_by_rarity = True
full_train_report = False #True: use full training set to report loss, accuracy, etc (a lot more VRAM needed) in the final result list, False: calculate average statistics per batch (less VRAM needed)
#lower and upper thresholds for rare VS common variants analyses
rare_threshold1 = 0
rare_threshold2 = 0.01
common_threshold1 = 0.01
common_threshold2 = 1


#Learning options
categorical = "False" #False: treat variables as numeric allele count vectors [0,2], True: treat variables as categorical values (0,1,2)(Ref, Het., Alt)
split_size = 100 #number of batches
my_keep_rate = 1 #keep rate for dropout funtion, 1=disable dropout
kn = 1 #number of k for k-fold CV (kn>=2); if k=1: just run training
training_epochs = 50 #learning epochs (if fixed masking = True) or learning permutations (if fixed_masking = False), default 500, number of epochs or data augmentation permutations (in data augmentation mode when fixed_masking = False)
#761 permutations will start masking 1 marker at a time, and will finish masking 90% of markers
last_batch_validation = False #when k==1, you may use the last batch for valitation if you want
#optimizer_type = "Adam" #Optimizers available now: Adam, RMSProp, GradientDescent
#loss_type = "WMSE" #MSE, CE, WMSE, WCE, FL, Pearson, WMSE #mean(((1.5-MAF)^5)*MSE)
#gamma = 5.0 #gamma hyper parameter value, >0 = use WMSE with selected gamma, 0 = do simple MSE as loss function, -1, use Pearson r2 as loss function
alt_signal_only = False #TODO Wether to treat variables as alternative allele signal only, like Minimac4, estimating the alt dosage
hsize = 0.5 #[0.1-1] size ratio for hidden layer, multiplied by number of input nodes

#Masking options
replicate_ARIC = False #True: use the same masking pattern as ARIC genotype array data, False: Random masking
fixed_masking = False #True: mask variants only at the beggining of the training cycle, False: mask again with a different pattern after each iteration (data augmentation mode)
mask_per_sample = True #True: randomly mask genotypes per sample instead of mask the entire variant for all samples, False: mask the entire variant for all samples 
random_masking = True #set random masks instead of preset ones
#disable_masking = True #disable masking completly, just for learning the data structure in the grid search
#PARAMETER NOW initial_masking_rate = 1/846 #when fixed_masking = False begin masking one variant at a time and gradually increase one by one, not random, 9p21.3 case is 1/846 for incrementing one variant at a time
mask_preset = False #True: mask from genotype array
#fixed_masking_rate = True #True: do not increment the ratio of variants masked at each permutation, False: do increment
#maximum_masking_rate=0
shuffle = True #Whether shuffle data or not at the begining of training cycle. Not necessary for online data augmentation.
repeat_cycles = 1 #how many times to repeat the masking rate

#Recovery options, online machine learning mode
#recovery_mode = "False" #False: start model from scratch, True: Recover model from last checkpoint
#path to restore model, in case recovery_mode = True 
model_path = './recovery/inference_model-1.ckpt'

#Testing options
test_after_train_step = True #True: Test on independent dataset after each epoch or permutation; False: run default training without testing
#in case test_after_step == True; provide an independent data set for testing
test_input_path = "../ARIC_PLINK_flagged_chromosomal_abnormalities_zeroed_out_bed.lifted_NCBI36_to_GRCh37.GH.ancestry-1.chr9_intersect1.vcf.gz.9p21.3.recode.vcf"
test_ground_truth_path = "../c1_ARIC_WGS_Freeze3.lifted_already_GRCh37_intersect1.vcf.gz.9p21.3.recode.vcf"
#in case test_after_step == True; provide a 2 column, tab delimited file specifying the genetic positions of the SNPs used in training, so these positions can be mapped to the positions in the input file for testing
test_position_path = "../HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf.pos.clean4"

#debugging options
verbose=0

###################################OPTIONS#############################################
 
#global variables
MAF_all_var = [] #MAF calculated only once for all variants,remove redundancies
rare_indexes = [] #indexes of rare variants
common_indexes = [] #indexes of common variants
MAF_all_var_vector = [] #vector of weights equal to number of output nodes (featuresX2 by default)
ncores = multiprocessing.cpu_count() #for parallel processing
config = tf.ConfigProto(log_device_placement=False)
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4
#config.gpu_options.per_process_gpu_memory_fraction = 0.20
config.gpu_options.allow_growth=True

# parsing vcf files, 
#If categorical false: convert 1|1 -> [0,2], 1|0 or 0|1 -> [1,1], 0|0 -> [2,0], missing -> [0,0]
#If categorical true: convert 1|1 -> 2, 1|0 or 0|1 -> 1, 0|0 -> 0, missing -> -1
#categorical=false is the only mode fully suported now
#todo: finish implenting support for categorical (onehot encoding), just for comparison purposes

def convert_genotypes_to_int(indexes, file, categorical="False"):
    print("process:", multiprocessing.current_process().name, "arguments:", indexes, ":", file, ":", categorical)
    #
    j=0
    command = "cut -f"
    
    for i in range(len(indexes)):
        #print(str(indexes[i]))
        command = command + str(indexes[i]+1)
        if(i<len(indexes)-1):
            command = command + ","
               
    command = command + " " + file
    #print(command)
    result = sp.check_output(command, encoding='UTF-8', shell=True)
    #result = result.stdout
    df = []
    
    first=True
    i=0
    for ln in result.split('\n'):
        i+=1
        if(not ln.startswith("#")):
            if(first==False and ln):
                tmp = ln.split('\t')
                #print(i, ": ", tmp, ": ", ln)
                df.append(tmp)
            else:
                first=False

    df = list(map(list, zip(*df)))   
    
    #print("BATCH SHAPE: ", len(df), len(df[0]))
    #print(df[0])
    new_df = 0
    if(categorical=="False"):
        new_df = np.zeros((len(df),len(df[0]),2))
        #new_df = np.zeros((df.shape[1]-9,len(df)*2))
    else:
        new_df = np.zeros((len(df),len(df[0])))
    #print(type(df))
    i = 0 #RR column index
    j = 0 #RR row index
    idx = 0
    my_hom = 2
    
    if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
        my_hom = 1
    
    #print(len(df), df[0][0])
    #print(len(df[0]))
    while i < len(df): #sample index, total-9 (9 for first comment columns)
        
        if(j==(len(df)-1)):
            print(j)
        
        j = 0 #variant index, while variant index less than total variant count
        while j < len(df[0]): #"|" is present when phased data is proved, "/" is usually unphased
            #print j

            df[i][j] = str(df[i][j])
            if(df[i][j].startswith('1|1') or df[i][j].startswith('1/1')):
                if(categorical=="True" or alt_signal_only==True):
                    new_df[idx][j] = 2
                else:
                    #new_df[idx][j] = np.array([0,2])
                    new_df[idx][j][0] = 0
                    #new_df[idx][j] = 0
                    #idx+=1
                    #new_df[idx][j] = 2
                    new_df[idx][j][1] = my_hom
            elif(df[i][j].startswith('1|0') or df[i][j].startswith('0|1') or df[i][j].startswith('1/0') or df[i][j].startswith('0/1')):
                if(categorical=="True" or alt_signal_only==True):
                    new_df[idx][j] = 1
                else:
                    #new_df[idx][j] = np.array([1,1])
                    new_df[idx][j][0] = 1
                    new_df[idx][j][1] = 1
                    #new_df[idx][j] = 1
                    #idx+=1
                    #new_df[idx][j] = 1
            elif(df[i][j].startswith('0|0') or df[i][j].startswith('0/0')):
                if(categorical=="True"  or alt_signal_only==True): 
                    new_df[idx][j] = 0
                else:
                    #new_df[idx][j] = np.array([2,0])
                    new_df[idx][j][0] = my_hom
                    new_df[idx][j][1] = 0
                    #new_df[idx][j] = 2
                    #idx+=1
                    #new_df[idx][j] = 0
            else:
                if(categorical=="True"):
                    new_df[idx][j] = -1
                elif(alt_signal_only==True):
                    new_df[idx][j] = 0
                else:
                    #new_df[idx][j] = np.array([0,0])                     
                    new_df[idx][j][0] = 0 
                    new_df[idx][j][1] = 0
                    #new_df[idx][j] = 0
                    #idx+=1
                    #new_df[idx][j] = 0
                #RR I forgot to mention that we have to take into account possible missing data
                #RR in case there is missing data (NA, .|., -|-, or anything different from 0|0, 1|1, 0|1, 1|0) = 3
            j += 1
        i += 1
        #pbar.update(1)
        idx += 1

    #print("processed_data")
    #for i in range(10):
    #    print(new_df[i][0])

    #the data needs to be flattened because the matrix multiplication step (x*W) 
    #doesn't support features with subfeatures (matrix of vectors)
    #new_df = np.reshape(new_df, (new_df.shape[0],new_df.shape[1]*2))

    return new_df.tolist()

#split inut data into chunks so we can prepare batches in parallel
def chunk(L,nchunks):
    L2 = list()
    j = round(len(L)/nchunks)
    chunk_size = j
    i = 0
    while i < len(L):
        chunk = L[i:j]
        L2.append(chunk)
        i = j
        j += chunk_size
        if(j>len(L)):
            j = len(L)
    return L2
#new_df = pd.DataFrame() # B) during

#parse initial_masking_rate if fraction is provided
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)        
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))

def process_data(file, categorical="False"):
     
    #Header and column names start with hashtag, skip those
    ncols = pd.read_csv(file, sep='\t', comment='#',header=None, nrows=1)    
    ncols = ncols.shape[1]
    
    print("Processing input data.")
    print("categorical: ", categorical)
    n_samples = ncols-9
    #RR subtract 9 from the number of columns to get the number of SNPs, 
    #RR because the first 9 columns are variant information, not genotypes
    print("number of samples: ", n_samples)
    
    indexes = list(range(9,ncols))
            
    start = timeit.default_timer()
    
    if(do_parallel==False):
        results = convert_genotypes_to_int(indexes, file, categorical)
        print( len(results), len(results[0]), len(results[0][0]))

    else:
        chunks = chunk(indexes, ncores )        

        pool = multiprocessing.Pool(ncores)

        results = pool.map(partial(convert_genotypes_to_int, file=file, categorical=categorical),chunks)
      
        pool.close()
        pool.join()
                        
        print(len(results), len(results[0]), len(results[0][0]) , len(results[0][0][0]))
    
        #for i in range(len(results)):
        #    print(len(results[i]))
    
        #merge outputs from all processes, reshaping nested list
        results = [item for sublist in results for item in sublist]

    print(len(results), len(results[0]), len(results[0][0]) )

    print("This file contains {} features (SNPs) and {} samples (subjects)".format(len(results[0]), n_samples))
    
    indexes = list(range(len(results[0])))

    results = np.asarray(results)
    #tf.reset_default_graph()
    #with tf.Session(config=config) as sess:
    #    results = sess.run(tf.constant(results))
    
    #reset tensorflow session
    #tf.reset_default_graph()
    #sess.close() 
    
    #new_df = [[0] * len(df) for i in range(len(df_T)-8)]
    #new_df = convert_genotypes_to_int(df, categorical)
    
    stop = timeit.default_timer()
    print('Time to load the data (sec): ', stop - start)
    
    start = timeit.default_timer()

    global MAF_all_var
    
    if(do_parallel_MAF == False):        
        
        MAF_all_var = calculate_MAF_global_GPU(indexes, results, categorical)
    
    else:
        chunks = chunk(indexes,ncores)
        
        pool = multiprocessing.Pool(ncores)

        MAF_all_var = pool.map(partial(calculate_MAF_global, inx=results, categorical=categorical),chunks)

        pool.close()
        pool.join()
        
        #merge outputs from all processes, reshaping nested list
        MAF_all_var = [item for sublist in MAF_all_var for item in sublist]
        
        
        global MAF_all_var_vector
        MAF_all_var_vector = []
        
        for i in range(len(MAF_all_var)):
            MAF_all_var_vector.append(MAF_all_var[i])
            MAF_all_var_vector.append(MAF_all_var[i])
            if(categorical==True):
                MAF_all_var_vector.append(MAF_all_var[i])
    
    global rare_indexes
    global common_indexes
    
    rare_indexes = filter_by_MAF_global(results, MAF_all_var, threshold1=rare_threshold1, threshold2=rare_threshold2, categorical=categorical)    
    common_indexes = filter_by_MAF_global(results, MAF_all_var, threshold1=common_threshold1, threshold2=common_threshold2, categorical=categorical)
    
    print("ALLELE FREQUENCIES", MAF_all_var)
    print("LENGTH1", len(MAF_all_var)) 
    stop = timeit.default_timer()
    print('Time to calculate MAF (sec): ', stop - start)
    
    return results

def filter_by_MAF_global(x, MAFs, threshold1=0, threshold2=1, categorical=False):
    
    #don't do any filtering if the thresholds are 0 and 1
    if(threshold1==0 and threshold2==1):
        return x

    indexes_to_keep = []
    i = 0
    j = 0
    k = 0   
    
    while i < len(MAFs):
        if(MAFs[i]>threshold1 and MAFs[i]<=threshold2):
            if(categorical==True or categorical=="True"):
                indexes_to_keep.append(j)
                indexes_to_keep.append(j+1)
                indexes_to_keep.append(j+2)
            elif(categorical==False or categorical=="False"):
                indexes_to_keep.append(k)
                indexes_to_keep.append(k+1)            
        i += 1
        j += 3
        k += 2
        
    return indexes_to_keep

def mask_data(indexes, mydata, mask_rate=0.9, categorical="False"):
    #start = timeit.default_timer()
    
    #def duplicate_samples(mydata, n):
    #    i=1
    #    while i < n:
    #        mydata.append(mydata)
    #        i+=1
    #    return mydata
    # random matrix the same shape of your data
    #print(len(mydata))
    if(disable_masking == True):
        print("No masking will be done for this run... Just learning data structure")
        return mydata, mydata
    
    #pick samples assigned to worker
    if(do_parallel_masking==True):
        mydata = mydata[indexes]
        
    original_data = np.copy(mydata)
    
    def do_masking(mydata,maskindex):     
        #print("Masking markers...")
        #print(maskindex)
        
        #print(mydata.shape)
        
        for i in maskindex:
            #print(len(mydata[i]))
            j = 0
            while j < len(mydata):
                if(categorical=="True"):
                    mydata[j][i]=-1
                elif(alt_signal_only==True):
                    mydata[j][i]=0
                else:
                    mydata[j][i]=[0,0]
                j=j+1
    
  
        return mydata    
    
    nmask = int(round(len(mydata[0])*mask_rate))
    # random mask for which values will be changed
    if(random_masking == True):
        maskindex = random.sample(range(0, len(mydata[0])), nmask)
        masked_data = do_masking(np.copy(original_data),maskindex)
        #stop = timeit.default_timer()
        #print('Time to mask the data (sec): ', stop - start)
        return masked_data, original_data
    elif(mask_preset==True):    
        maskindex = pd.read_csv('masking_pattern.txt', sep=',', comment='#',header=None)
        # export pandas obj as a list of tuples
        maskindex = [tuple(x) for x in maskindex.values]
        npatterns = len(maskindex)
        for i in range(npatterns):
            maskindex_i = list(maskindex[i])
            masked_data_tmp = do_masking(np.copy(original_data),maskindex_i)
            if(i==0):
                masked_data = np.copy(masked_data_tmp)
                unmasked_data = np.copy(original_data)
            else:
                unmasked_data.append(original_data)
                masked_data.append(masked_data_tmp)
            if(i==(npatterns-1)):
                #stop = timeit.default_timer()
                #print('Time to mask the data (sec): ', stop - start)
                return masked_data, unmasked_data
    else: #gradual masking in ascending order
        #if(categorical=="False"):
        #    #MAFs = calculate_MAF(original_data, False)
        #else:
        #    MAFs = calculate_MAF(original_data, True)
        #original_indexes = range(len(MAFs))
        original_indexes = range(len(MAF_all_var))
        myd = dict(zip(original_indexes,MAF_all_var))
        myd_sorted = sorted(myd.items(), key=itemgetter(1)) #0 keys, 1, values
        sorted_indexes = list(myd_sorted.keys())
        mask_index = sorted_indexes[0:nmask+1]
        masked_data = do_masking(np.copy(original_data),maskindex)
        return masked_data, original_data
    

# In[ ]:
def mask_data_per_sample(indexes, mydata, mask_rate=0.9, categorical="False"):
    #start = timeit.default_timer()
    # random matrix the same shape of your data
    #print(len(mydata))
    nmask = int(round(len(mydata[0])*mask_rate))
    # random boolean mask for which values will be changed
        
    #for i in range(10):
    #    print(mydata[i][0:11])
    if(do_parallel_masking==True):
        mydata = mydata[indexes]
    
    j = 0
    while j < len(mydata):
        #redefines which variants will be masked for every new sample
        maskindex = random.sample(range(0, len(mydata[0]-1)), nmask) 

        for i in maskindex:
            if(categorical=="True"):
                mydata[j][i]=-1
            elif(alt_signal_only==True):
                mydata[j][i]=0                   
            else:
                mydata[j][i]=[0,0]
        j=j+1

    #stop = timeit.default_timer()
    #print('Time to mask the data (sec): ', stop - start)  
    return mydata

def map_genotypes(indexes, refpos, infile, categorical):
    print("process:", multiprocessing.current_process().name, "arguments:", indexes, ":", infile, ":", categorical)
    #
    j=0
    command = "cut -f"
    
    for i in range(len(indexes)):
        #print(str(indexes[i]))
        command = command + str(indexes[i]+1)
        if(i<len(indexes)-1):
            command = command + ","
               
    command = command + " " + infile
    #print(command)
    result = sp.check_output(command, encoding='UTF-8', shell=True)
    
     
    #result = result.stdout
    df = []
    
    first=True
    i=0
    for ln in result.split('\n'):
        i+=1
        if(not ln.startswith("#")):
            if(first==False and ln):
                tmp = ln.split('\t')
                #print(i, ": ", tmp, ": ", ln)
                df.append(tmp)
            else:
                first=False

    df = list(map(list, zip(*df)))   

    command2 = "cut -f1,2 " + infile
    
    result2 = sp.check_output(command2, encoding='UTF-8', shell=True)
    
    #result = result.stdout
    dfp = []
    
    first=True
    i=0
    for ln in result2.split('\n'):
        i+=1
        if(not ln.startswith("#")):
            if(first==False and ln):
                tmp = ln.split('\t')
                #print(i, ": ", tmp, ": ", ln)
                tmp[0] = int(tmp[0])
                tmp[1] = int(tmp[1])
                dfp.append(tmp)
            else:
                first=False

    dfp = list(map(list, zip(*dfp)))   
    #print("BATCH SHAPE: ", len(df), len(df[0]))
    #print(df[0])
    new_df = 0
    if(categorical=="False"):
        new_df = np.zeros((len(df),len(refpos),2))
        #new_df = np.zeros((df.shape[1]-9,len(df)*2))
    else:
        new_df = np.zeros((len(df),len(refpos)))
    #print(type(df))
    
    
    inpos = pd.Series(range(len(dfp[1])), index=dfp[1])
    
    #print(inpos[2])
    #print(inpos)
    #print(refpos)
    
    #genetic variants are rows and samples are columns
    
    #print(new_df.shape)
    i = 0 #RR column index
    j = 0 #RR row index
    idx = 0
    #print("Processing input data.")
    #print(categorical)
    myidx = 0
    
    #print("SSSSSSSSSSS", refpos[0], inpos.keys()[0])
    my_hom = 2
    if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
        my_hom = 1
    
    while i < len(df):
        j = 0
        while j < len(refpos): #"|" is present when phased data is proved, "/" is usually unphased
            if(refpos[j] in inpos.keys()):
                #print("RRRRRRRRRRRRRRRRRRR")
                myidx = inpos[refpos[j]]
                #print(j)
                #print(inpos[refpos[j]])
                #print(refpos[j])
                #print(df[i][myidx])
                #print(df[i+1][myidx])
                if(df[i][myidx].startswith('1|1') or df[i][myidx].startswith('1/1')):
                    if(categorical=="True" or alt_signal_only==True):
                        new_df[idx][j] = 2
                    else:
                        #new_df[idx][j] = np.array([0,2])
                        new_df[idx][j][0] = 0
                        new_df[idx][j][1] = my_hom
                elif(df[i][myidx].startswith('1|0') or df[i][myidx].startswith('0|1') or df[i][myidx].startswith('1/0') or df[i][myidx].startswith('0/1')):
                    if(categorical=="True" or alt_signal_only==True):
                        new_df[idx][j] = 1
                    else:
                        #new_df[idx][j] = np.array([1,1])
                        new_df[idx][j][0] = 1
                        new_df[idx][j][1] = 1
                elif(df[i][myidx].startswith('0|0') or df[i][myidx].startswith('0/0')):
                    if(categorical=="True" or alt_signal_only==True): 
                        new_df[idx][j] = 0
                    else:
                        #new_df[idx][j] = np.array([2,0])
                        new_df[idx][j][0] = my_hom
                        new_df[idx][j][1] = 0
                else:
                    if(categorical=="True"):
                        new_df[idx][j] = -1
                    elif(alt_signal_only==True):
                        new_df[idx][j] = 0
                    else:
                        #new_df[idx][j] = np.array([0,0]) 
                        new_df[idx][j][0] = 0 
                        new_df[idx][j][1] = 0 
            else:
                if(categorical=="True"):
                    new_df[idx][j] = -1
                elif(alt_signal_only==True):
                    new_df[idx][j] = 0                   
                else:
                    #new_df[idx][j] = np.array([0,0]) 
                    new_df[idx][j][0] = 0 
                    new_df[idx][j][1] = 0 
                #if(idx==0):
                    #print(j)
                #RR in case there is missing data (NA, .|., -|-, or anything different from 0|0, 1|1, 0|1, 1|0) = 3
            j += 1
        i += 1
        #pbar.update(1)
        idx += 1

    #print("processed_data")
    #for i in range(10):
    #    print(new_df[i][0])

    #the data needs to be flattened because the matrix multiplication step (x*W) 
    #doesn't support features with subfeatures (matrix of vectors)
    #new_df = np.reshape(new_df, (new_df.shape[0],new_df.shape[1]*2))
    #print(new_df.shape)
    #pbar.close()
    #stop = timeit.default_timer()
    #print('Time to load and process testing the data (sec): ', stop - start)
    
    return new_df    
    
    

def process_testing_data(posfile, infile, ground_truth=False, categorical="False"):
    
    start = timeit.default_timer()
    
    #Header and column names start with hastag, skip those
    #posfile should contain 2 columns separated by tab: 1st = chromosome ID, 2nd = position
    #vcf can be imported as posfile as well, but will take much longer to read and process
    refpos = pd.read_csv(posfile, sep='\t', comment='#',header=None)
    
    #0      22065657
    #1      22065697
    #2      22065904
    #3      22065908
    #4      22065974
    #5      22065977
    
    refpos = pd.Series(refpos[1], index=range(len(refpos[1])))

    #print(refpos[1])
    
    #Header and column names start with hashtag, skip those
    ncols = pd.read_csv(infile, sep='\t', comment='#',header=None, nrows=1)    
    ncols = ncols.shape[1]
    
    print("Processing input data.")
    print("categorical: ", categorical)
    n_samples = ncols-9
    #RR subtract 9 from the number of columns to get the number of SNPs, 
    #RR because the first 9 columns are variant information, not genotypes
    print("number of samples: ", n_samples)
    
    indexes = list(range(9,ncols))
            
    start = timeit.default_timer()
    
    if(do_parallel==False):
        results = map_genotypes(indexes, refpos, file, categorical)
        print( len(results), len(results[0]), len(results[0][0]))

    else:
        chunks = chunk(indexes, ncores )        

        pool = multiprocessing.Pool(ncores)

        results = pool.map(partial(map_genotypes, refpos=refpos, infile=infile, categorical=categorical), chunks)
      
        pool.close()
        pool.join()
                            
    #infile is the input file: genotype data set to be imputed, or ground truth to mapped
    #df = pd.read_csv(infile, sep='\t', comment='#',header=None)
    
    #0      22065657
    #1      22066211
    #2      22066363
    #3      22066572
    #4      22067004
    #5      22067276
    
    
    print(len(results), len(results[0]), len(results[0][0]) , len(results[0][0][0]))
    
    #for i in range(len(results)):
    #    print(len(results[i]))
    
    #merge outputs from all processes, reshaping nested list
    results = [item for sublist in results for item in sublist]

    print(len(results), len(results[0]), len(results[0][0]) )

    print("This file contains {} features (SNPs) and {} samples (subjects)".format(len(results[0]), n_samples))
    
    indexes = list(range(len(results[0])))

    results = np.asarray(results)
    
    print("calculating MAFs for testing data...")


    if(ground_truth==True):

        print("calculating MAFs for testing data...")

        global test_MAF_all_var

        if(do_parallel_MAF == False):

            test_MAF_all_var = calculate_MAF_global_GPU(indexes, results, categorical)

        else:
            chunks = chunk(indexes, ncores )

            pool = multiprocessing.Pool(ncores)

            test_MAF_all_var = pool.map(partial(calculate_MAF_global_test, inx=results, categorical=categorical),chunks)

            pool.close()
            pool.join()

            #merge outputs from all processes, reshaping nested list
            test_MAF_all_var = [item for sublist in test_MAF_all_var for item in sublist]

        print("ALLELE FREQUENCIES", test_MAF_all_var)

    stop = timeit.default_timer()
    print('Time to load and process testing the data (sec): ', stop - start)

    return results


def variable_summaries(var):
    #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def logfunc(x, x2):
    x = tf.cast(x, tf.float64)
    x2 = tf.cast(x2, tf.float64)
    
    eps=tf.cast(1e-14, tf.float64)
    one=tf.cast(1.0, tf.float64)
    eps2 = tf.subtract(one,eps)
    
    cx = tf.clip_by_value(x, eps, eps2)
    cx2 = tf.clip_by_value(x2, eps, eps2)
    return tf.multiply( x, tf.log(tf.div(cx,cx2)))


#Kullback-Leibler divergence equation (KL divergence)
#The result of this equation is added to the loss function result as an additional penalty to the loss based on sparsity
def KL_Div(rho, rho_hat):

    rho = tf.cast(rho, tf.float64)
    rho_hat = tf.cast(rho_hat, tf.float64)

    KL_loss = rho * logfunc(rho, rho_hat) + (1 - rho) * logfunc((1 - rho), (1 - rho_hat))
    
    #rescaling KL result to 0-1 range
    return tf.div(KL_loss-tf.reduce_min(KL_loss)+1e-10,tf.reduce_max(KL_loss)-tf.reduce_min(KL_loss))

    #RR I just simplified the KL divergence equation according to the book:
    #RR "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurélien Géron
    #RR Example source code here https://github.com/zhiweiuu/sparse-autoencoder-tensorflow/blob/master/SparseAutoEncoder.py
    #RR KL2 is the classic sparsity implementation, source reference: https://github.com/elykcoldster/sparse_autoencoder/blob/master/mnist_sae.py
    #https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
def pearson_r_loss(y_true, y_pred):
    
    #y_true = tf.cast(y_true, tf.float32)
    #y_pred = tf.cast(y_pred, tf.float32)
    
    pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true, name='pearson_r')
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'pearson_r'  in i.name.split('/')]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        pearson_r = tf.identity(pearson_r)
        pearson_r = tf.square(pearson_r)
        rloss = tf.subtract(1.0, pearson_r, name='reconstruction_loss')
        #return 1-pearson_r**2
        return rloss

def weighted_MSE(y_pred, y_true):
    MSE_per_var = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)), axis=0)
    
    #The condition tensor acts as a mask that chooses, based on the value at each element, whether the corresponding element / row in the output should be taken from x (if true) or y (if false).
    #cond_shape = y_true.get_shape()
    #x = tf.ones(cond_shape)
    #y = tf.zeros(cond_shape)
    
    #tf.where(tf.equal(0,tf.round(y_true)), x, y)       
    #mean(((1.5-MAF)^5)*MSE)
    weights = tf.subtract(1.5, MAF_all_var_vector)
    weights = tf.pow(weights, gamma)
    weighted_MSE_per_var = tf.multiply(MSE_per_var, weights)
     
    weighted_MSE_loss = tf.reduce_mean(weighted_MSE_per_var, name='reconstruction_loss')
    return weighted_MSE_loss


def calculate_pt(y_pred, y_true):

    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true,tf.float64)

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

    #ref:https://github.com/unsky/focal-loss/blob/master/focal_loss.py
    pt_1 = tf.clip_by_value(pt_1, 1e-8, 1.0) #avoid log(0) that returns inf
    #pt_1 = tf.add(pt_1, 1e-8) #avoid log(0) that returns inf

    #if value is zero in y_true, than take value from y_pred, otherwise, write zeros
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_0 = tf.clip_by_value(pt_0, 0, 1.0-1e-8)
        
    return pt_0, pt_1

def calculate_CE(y_pred, y_true):

    pt_0, pt_1 = calculate_pt(y_pred, y_true)
    one = tf.cast(1.0, tf.float64)
    eps=tf.cast(1.0+1e-8, tf.float64)
    n1 =  tf.cast(-1.0, tf.float64)

    CE_1 = tf.multiply(n1,tf.log(pt_1))
    CE_0 = tf.multiply(n1,tf.log(tf.subtract(eps,pt_0)))

    #CE_1 = tf.abs(CE_1)
    #CE_0 = tf.abs(CE_0)

    return CE_0, CE_1

def cross_entropy(y_pred, y_true):

    pt_0, pt_1 = calculate_pt(y_pred, y_true)

    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    CE_1 = tf.reduce_sum(CE_1)
    CE_0 = tf.reduce_sum(CE_0)

    CE = tf.add(CE_1, CE_0, name='reconstruction_loss')

    return CE

def calculate_alpha():

    one=tf.cast(1.0, tf.float64)
    eps=tf.cast(1.0-1e-4, tf.float64)

    alpha = tf.multiply(tf.cast(MAF_all_var_vector,tf.float64),2.0)
    alpha = tf.clip_by_value(alpha, 1e-4, eps)

    alpha_1 = tf.divide(one, alpha)
    alpha_0 = tf.divide(one, tf.subtract(one,alpha))
    #alpha_1 = alpha_0
    
    return alpha_0, alpha_1

def weighted_cross_entropy(y_pred, y_true):

    one=tf.cast(1.0, tf.float64)
    eps=tf.cast(1.0+1e-8, tf.float64)
    n1 =  tf.cast(-1.0, tf.float64)

    #pt_0, pt_1 = calculate_pt(y_pred, y_true)

    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    alpha_0, alpha_1 = calculate_alpha()

    WCE_per_var_1 = tf.multiply(CE_1, alpha_1)
    WCE_per_var_0 = tf.multiply(CE_0, alpha_0)

    WCE_1 = tf.reduce_sum(WCE_per_var_1)
    WCE_0 = tf.reduce_sum(WCE_per_var_0)

    WCE = tf.add(WCE_1, WCE_0, name='reconstruction_loss')

    return WCE


def calculate_gamma(y_pred, y_true):
    
    one=tf.cast(1.0, tf.float64)
    eps=tf.cast(1.0+1e-8, tf.float64)

    my_gamma=tf.cast(gamma, tf.float64)

    pt_0, pt_1 = calculate_pt(y_pred, y_true)

    #if statement to avoid useless calculaions
    if(gamma == 0):
        gamma_0 = one
        gamma_1 = one
    elif(gamma == 1):
        gamma_0 = pt_0
        gamma_1 = tf.subtract(eps, pt_1)
    else:
        gamma_0 = tf.pow(pt_0, my_gamma)
        gamma_1 = tf.pow(tf.subtract(eps, pt_1), my_gamma)
    
    return gamma_0, gamma_1

#ref: https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
def focal_loss(y_pred, y_true):
    
    #avoid making useless calculations if gamma==0
    #if(gamma==0):
    #    WCE = weighted_cross_entropy(y_pred, y_true)
    #    return WCE

    one=tf.cast(1.0, tf.float64)
    
    #my_gamma=tf.cast(gamma, tf.float64)

    #pt_0, pt_1 = calculate_pt(y_pred, y_true)

    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    alpha_0, alpha_1 = calculate_alpha()
    
    gamma_0, gamma_1 = calculate_gamma(y_pred, y_true)

    FL_per_var_1 = tf.multiply(CE_1, alpha_1)
    FL_per_var_0 = tf.multiply(CE_0, alpha_0)

    #avoid useless calculations
    if(gamma>0):
        FL_per_var_1 = tf.multiply(gamma_1, FL_per_var_1)
        FL_per_var_0 = tf.multiply(gamma_0, FL_per_var_0)

    FL_1 = tf.reduce_sum(FL_per_var_1)
    FL_0 = tf.reduce_sum(FL_per_var_0)

    FL = tf.add(FL_1, FL_0, name='reconstruction_loss')

    return FL

def fl01(y_pred, y_true):
    
    #avoid making useless calculations if gamma==0
    #if(gamma==0):
    #    WCE = weighted_cross_entropy(y_pred, y_true)
    #    return WCE

    one=tf.cast(1.0, tf.float64)
    #eps=tf.cast(1.0+1e-8, tf.float64)
    
    CE_0, CE_1 =  calculate_CE(y_pred, y_true)

    alpha_0, alpha_1 = calculate_alpha()

    FL_per_var_1 = tf.multiply(CE_1, alpha_1)
    FL_per_var_0 = tf.multiply(CE_0, alpha_0)
     
    FL_per_var_1 = tf.multiply(FL_per_var_1, gamma_1)
    FL_per_var_0 = tf.multiply(FL_per_var_0, gamma_0)

    FL_1 = tf.reduce_sum(FL_per_var_1)
    FL_0 = tf.reduce_sum(FL_per_var_0)

    return FL_0, FL_1

def f1_score(y_pred, y_true, sess):
    
    f1s = [0, 0, 0]
    two = tf.cast(2.0, tf.float64)
    eps = tf.cast(1e-8, tf.float64)

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    
    y_true = tf.clip_by_value(y_true, 0.0, 1.0) #in case the input encoding is [0,1,2]
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(tf.multiply(y_pred, y_true), axis=axis)
        FP = tf.count_nonzero(tf.multiply(y_pred, tf.subtract(y_true,1.0)), axis=axis)
        FN = tf.count_nonzero(tf.multiply(tf.subtract(y_pred,1.0),y_true), axis=axis)
        
        TP = tf.cast(TP, tf.float64)
        FP = tf.cast(FP, tf.float64)
        FN = tf.cast(FN, tf.float64)
        
        TP = tf.add(TP, eps)
        
        precision = tf.divide(TP, tf.add(TP, FP))
        recall = tf.divide(TP, tf.add(TP, FN))
        #f1 = tf.multiply(two, tf.multiply(precision, tf.divide(recall, tf.add(precision, recall))))
        top = tf.multiply(precision, recall)
        bottom = tf.add(precision, recall)
        f1 = tf.multiply(two, tf.divide(top,bottom))

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights = tf.divide(weights,tf.reduce_sum(weights))

    f1s[2] = tf.reduce_sum(tf.multiply(f1, weights))

    micro, macro, weighted = sess.run(f1s)
    
    return micro, macro, weighted

    
    
def pearson_r_per_var(x, y):
    
    mx, my = tf.reduce_mean(x, axis=0), tf.reduce_mean(y, axis=0)
    xm, ym = tf.subtract(x, mx), tf.subtract(y, my)
    r_num = tf.reduce_sum(tf.multiply(xm, ym), axis=0)
    
    r_num = tf.add(r_num, 1e-12)
    
    r_den = tf.sqrt( tf.multiply( tf.reduce_sum(tf.square(xm), axis=0), tf.reduce_sum(tf.square(ym), axis=0) ) )
    
    r_den = tf.add(r_den, 1e-12)

    r = tf.divide(r_num, r_den)
    r = tf.maximum(tf.minimum(r, 1.0), -1.0)
    #r = tf.where(tf.is_nan(r), tf.zeros_like(r), r)   
                   
    mr = tf.reduce_mean(r)
    
                   
    return mr

    

#drop out function will exclude samples from our input data
#keep_rate will determine how many samples will remain after randomly droping out samples from the input
def dropout(input, name, keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, keep_rate)
    return out
    # call function like this, p1 is input, name is layer name, and keep rate doesnt need explanation,  
    # do1 = dropout(p1, name='do1', keep_rate=0.75)

    #A value of 1.0 means that dropout will not be used.
    #TensorFlow documentation https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#dropout


# In[ ]:


#Example adapted and modified from 
#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

#Encodes a Hidden layer
def encoder(x, func, l1_val, l2_val, weights, biases, units_num, keep_rate=1): #RR added keep_rate
    x=tf.cast(x, tf.float64)
    
    print("Setting up encoder/decoder.")
    if(l2_val==0):
        regularizer = tf.contrib.layers.l1_regularizer(l1_val)
    else:
        regularizer = tf.contrib.layers.l1_l2_regularizer(l1_val,l2_val)

    #dropout   
    if keep_rate != 1: ##RR added dropout
            x = dropout(x, 'x', keep_rate) ##RR added dropout
        
    if func == 'sigmoid':
        print('Encoder Activation function: sigmoid')
        
        #tf.nn.sigmoid computes sigmoid of x element-wise.
        #Specifically, y = 1 / (1 + exp(-x))
        #tf.matmul multiply output of input_layer with a weight matrix and add biases
        #tf.matmul Multiplies matrix a by matrix b, producing a * b
        #If one or both of the matrices contain a lot of zeros, a more efficient multiplication algorithm can be used by setting the corresponding a_is_sparse or b_is_sparse flag to True. These are False by default.
        #tf.add will sum the result from tf.matmul (input*weights) to the biases
        #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        #with tf.device("/gpu:1"):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

        #This layer implements the operation: 
        #outputs = activation(inputs * weights + bias) 
        #where activation is the activation function passed as the activation argument, defined inprevious line
        #The function is applied after the linear transformation of inputs, not W*activation(inputs)
        #Otherwise the function would not allow nonlinearities
        layer_1 = tf.layers.dense(layer_1, units=units_num, kernel_regularizer= regularizer)
                       
    elif func == 'tanh':
        print('Encoder Activation function: tanh')
        #with tf.device("/gpu:1"):
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_1 = tf.layers.dense(layer_1, units=units_num, kernel_regularizer= regularizer)
        #layer_1 = tf.layers.dense(layer_1, units=221, kernel_regularizer= regularizer)
    elif func == 'relu':
        print('Encoder Activation function: relu')
        #with tf.device("/gpu:1"):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
        layer_1 = tf.layers.dense(layer_1, units=units_num, kernel_regularizer= regularizer)
        #layer_1 = tf.layers.dense(layer_1, units=221, kernel_regularizer= regularizer)

    return layer_1
        
def decoder(x, func, weights, biases):
    
    x = tf.cast(x, tf.float64)
    
    if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
        entropy_loss = True
    else:
        entropy_loss = False
        
    if func == 'sigmoid':
        print('Decoder Activation function: sigmoid')
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        #rescaling, if dealing with categorical variables or factors, tf.reduce_max(x) will result in 1
        if(entropy_loss==False):
            layer_1 = tf.multiply(layer_1, tf.reduce_max(x), name="y_pred")
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    elif func == 'tanh':
        print('Decoder Activation function: tanh')
        layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        #rescaling, if dealing with categorical variables or factors, tf.reduce_max(x) will result in 1
        if(entropy_loss==False):
            layer_1 = tf.div(tf.multiply(tf.add(layer_1, 1), tf.reduce_max(x)), 2, name="y_pred")
        else:
            layer_1 = tf.div(tf.add(layer_1, 1), 2, name="y_pred")
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    elif func == 'relu':
        print('Decoder Activation function: relu')
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']), name="y_pred")
        #no rescaling needed
        #layer_1 = tf.round(layer_1)
        #layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1'])

    return layer_1


#TODO, experimental, debug this function
def mean_empirical_r2_GPU(x,y, categorical=False):
    #This calculates exactly the same r2 as the empirical r2hat from minimac3 and minimac4
    #The estimated r2hat is different
    j=0
    mean_correl = 0
    correls = []
    
    #tf.reset_default_graph()
    
    sess = tf.Session(config=config) 
    
    while j < len(y[0]):
        if(categorical==False):
            getter = operator.itemgetter([j+1])
            x_genotypes = list(map(list, map(getter, np.copy(x))))
            y_genotypes = list(map(list, map(getter, np.copy(y))))
            x_genotypes = list(np.array(x_genotypes).flat)
            y_genotypes = list(np.array(y_genotypes).flat)
            #print("GGGGGG")
            #print(x_genotypes)
            #print(y_genotypes)
            correl, _ = sess.run(tf.contrib.metrics.streaming_pearson_correlation(x_genotypes, y_genotypes))
            mean_correl = sess.run( tf.add(mean_correl, tf.divide(correl,len(y[0]))) )
            j+=2
        else:
            x_genotypes = []
            y_genotypes = []
            for i in range(len(y)):
                x_genotypes.append(np.argmax(x[i][j:j+3]))                
                y_genotypes.append(np.argmax(y[i][j:j+3]))
            
            correl, _ = sess.run(tf.contrib.metrics.streaming_pearson_correlation(x_genotypes, y_genotypes))
            mean_correl = sess.run( tf.add(mean_correl, tf.divide(correl,len(y[0]))) )
            j+=3
        correls.append(mean_correl)
        #print("mean_correl",mean_correl)
    
    #tf.reset_default_graph()
    sess.close()
    
    return mean_correl, correls



def mean_empirical_r2(x,y, categorical=False):
    #This calculates exactly the same r2 as the empirical r2hat from minimac3 and minimac4
    #The estimated r2hat is different
    j=0
    mean_correl = 0
    correls = []
    while j < len(y[0]):
        if(categorical==False):
            getter = operator.itemgetter([j+1])
            x_genotypes = list(map(list, map(getter, np.copy(x))))
            y_genotypes = list(map(list, map(getter, np.copy(y))))
            x_genotypes = list(np.array(x_genotypes).flat)
            y_genotypes = list(np.array(y_genotypes).flat)
            #print("GGGGGG")
            #print(x_genotypes)
            #print(y_genotypes)
            correl = linregress(x_genotypes, y_genotypes)[2]
            mean_correl += correl/len(y[0])
            j+=2
        else:
            x_genotypes = []
            y_genotypes = []
            for i in range(len(y)):
                x_genotypes.append(np.argmax(x[i][j:j+3]))                
                y_genotypes.append(np.argmax(y[i][j:j+3]))
            
            correl = linregress(x_genotypes, y_genotypes)[2]
            mean_correl += correl/len(y[0])
            j+=3
        correls.append(mean_correl)
        #print("mean_correl",mean_correl)
    return mean_correl, correls

def calculate_MAF(x, categorical=False):
    j=0
    MAF_list = []
    if(categorical==True):
        while j < (len(x[0])-2):
            ref = 0
            alt = 0
            MAF = 0        
            for i in range(len(x)):
                allele_index = np.argmax(x[i][j:j+3])
                if(allele_index == 0):
                    ref+=2
                elif(allele_index == 1):
                    ref+=1
                    alt+=1
                elif(allele_index == 2):
                    alt+=2
            if(alt<=ref):
                MAF=alt/(ref+alt)
                #major=ref/len(y)
            else:
                MAF=ref/(ref+alt)
                #major=alt/len(y)
                #print(MAF)
            MAF_list.append(MAF)    
            j+=3           
    elif(categorical==False):
        if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
            while j < (len(x[0])-1):
                ref = 0
                alt = 0
                MAF = 0       
                for i in range(len(x)):                   
                    ref+=x[i][j]
                    alt+=x[i][j+1]
                    if(x[i][j] != x[i][j+1]):
                        ref+=x[i][j]                     
                        alt+=x[i][j+1]                       
                if(alt<=ref):
                    MAF=alt/(ref+alt)
                    #major=ref/len(y)
                else:
                    MAF=ref/(ref+alt)
                MAF_list.append(MAF)    
                j+=2
        else:
            while j < (len(x[0])-1):
                ref = 0
                alt = 0
                MAF = 0       
                for i in range(len(x)):
                    ref+=x[i][j]
                    alt+=x[i][j+1]   
                if(alt<=ref):
                    MAF=alt/(ref+alt)
                    #major=ref/len(y)
                else:
                    MAF=ref/(ref+alt)
                MAF_list.append(MAF)    
                j+=2
    return MAF_list

def calculate_MAF_global_GPU(indexes, inx, categorical="False"):
    

    j=0
    if(do_parallel_MAF==True):
        getter = operator.itemgetter(indexes)
        x = list(map(list, map(getter, np.copy(inx))))
    else:
        x = inx
    MAF_list = []
        
    #tf.reset_default_graph()
    
    with tf.Session(config=config) as sess:        
       
        #print("LENGTH", len(x[0]))
        if(categorical=="True"):
            while j < (len(x[0])):
                ref = 0
                alt = 0
                MAF = 0        
                for i in range(len(x)):
                    if(x[i][j] == 0):
                        ref = sess.run(tf.add(ref,2))
                    elif(x[i][j] == 1):
                        ref = sess.run(tf.add(ref,1))
                        alt = sess.run(tf.add(alt,1))
                    elif(x[i][j] == 2):
                        alt = sess.run(tf.add(alt,2))
                if(alt<=ref):
                    MAF=sess.run(tf.div(alt,tf.add(ref,alt)))
                    #major=ref/len(y)
                else:
                    MAF=sess.run(tf.div(ref,tf.add(ref,alt)))
                    #major=alt/len(y)
                    #print(MAF)
                MAF_list.append(MAF)
                j+=1          
        elif(categorical=="False"):
            if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
                while j < (len(x[0])):
                    ref = 0
                    alt = 0
                    MAF = 0        
                    for i in range(len(x)):
                        ref = sess.run(tf.add(ref,x[i][j][0]))
                        alt = sess.run(tf.add(alt,x[i][j][1])) 
                        if(x[i][j][0]!=x[i][j][1]):
                            ref = sess.run(tf.add(ref,x[i][j][0]))
                            alt = sess.run(tf.add(alt,x[i][j][1]))
                    if(alt<=ref):
                        MAF=sess.run(tf.div(alt,tf.add(ref,alt)))
                        #major=ref/len(y)
                    else:
                        MAF=sess.run(tf.div(ref,tf.add(ref,alt)))
                    MAF_list.append(MAF)    
                    j+=1            
            else:    
                while j < (len(x[0])):
                    ref = 0
                    alt = 0
                    MAF = 0        
                    for i in range(len(x)):
                        ref = sess.run(tf.add(ref,x[i][j][0]))
                        alt = sess.run(tf.add(alt,x[i][j][1]))  
                    if(alt<=ref):
                        MAF=sess.run(tf.div(alt,tf.add(ref,alt)))
                        #major=ref/len(y)
                    else:
                        MAF=sess.run(tf.div(ref,tf.add(ref,alt)))
                    MAF_list.append(MAF)    
                    j+=1
    
    #reset tensorflow session
    #tf.reset_default_graph()
    sess.close()
    return MAF_list

def calculate_MAF_global(indexes, inx, categorical="False"):
    j=0
    if(do_parallel_MAF==True):
        getter = operator.itemgetter(indexes)
        x = list(map(list, map(getter, np.copy(inx))))
    else:
        x = inx
    MAF_list = []
    #print("LENGTH", len(x[0]))
    if(categorical=="True"):
        while j < (len(x[0])):
            ref = 0
            alt = 0
            MAF = 0        
            for i in range(len(x)):
                if(i == 0):
                    ref+=2
                elif(i == 1):
                    ref+=1
                    alt+=1
                elif(i == 2):
                    alt+=2
            if(alt<=ref):
                MAF=alt/(ref+alt)
                #major=ref/len(y)
            else:
                MAF=ref/(ref+alt)
                #major=alt/len(y)
                #print(MAF)
            MAF_list.append(MAF)    
            j+=1          
    elif(categorical=="False"):
        if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
            while j < (len(x[0])):
                ref = 0
                alt = 0
                MAF = 0
                for i in range(len(x)):
                    ref+=x[i][j][0]
                    alt+=x[i][j][1]
                    if(x[i][j][0]!=x[i][j][1]):
                        ref+=x[i][j][0]
                        alt+=x[i][j][1]
                if(alt<=ref):
                    MAF=alt/(ref+alt)
                    #major=ref/len(y)
                else:
                    MAF=ref/(ref+alt)
                MAF_list.append(MAF)
                j+=1
        else:
            while j < (len(x[0])):
                ref = 0
                alt = 0
                MAF = 0
                for i in range(len(x)):
                    ref+=x[i][j][0]
                    alt+=x[i][j][1]
                if(alt<=ref):
                    MAF=alt/(ref+alt)
                    #major=ref/len(y)
                else:
                    MAF=ref/(ref+alt)
                MAF_list.append(MAF)
                j+=1
    return MAF_list


def calculate_MAF_global_test(indexes, inx, categorical="False"):
    j=0
    if(do_parallel_MAF==True):
        getter = operator.itemgetter(indexes)
        x = list(map(list, map(getter, np.copy(inx))))
    else:
        x = inx
    MAF_list = []
    #print("LENGTH", len(x[0]))
    if(categorical=="True"):
        while j < (len(x[0])):
            ref = 0
            alt = 0
            MAF = 0
            for i in range(len(x)):
                if(i == 0):
                    ref+=2
                elif(i == 1):
                    ref+=1
                    alt+=1
                elif(i == 2):
                    alt+=2
            if(alt==0 and ref==0):
                MAF=-1
            else:
                MAF=MAF_all_var[j]
            MAF_list.append(MAF)
            j+=1
    elif(categorical=="False"):
        if(loss_type=="CE" or loss_type=="WCE" or loss_type=="FL"):
            while j < (len(x[0])):
                ref = 0
                alt = 0
                MAF = 0
                for i in range(len(x)):
                    ref+=x[i][j][0]
                    alt+=x[i][j][1]
                    if(x[i][j][0]!=x[i][j][1]):
                        ref+=x[i][j][0]
                        alt+=x[i][j][1]
                if(alt==0 and ref==0):
                    MAF=-1
                else:
                    MAF=MAF_all_var[j]
                MAF_list.append(MAF)
                j+=1
        else:
            while j < (len(x[0])):
                ref = 0
                alt = 0
                MAF = 0
                for i in range(len(x)):
                    ref+=x[i][j][0]
                    alt+=x[i][j][1]
                if(alt==0 and alt==0):
                    MAF=-1
                else:
                    MAF=MAF_all_var[j]
                MAF_list.append(MAF)
                j+=1
    return MAF_list



#TODO review this and make this work with CE, WCE, and FL
def mean_estimated_r2_GPU(x, categorical=False):

    #tf.reset_default_graph()
    
    sess = tf.Session(config=config)    
    
    def calculate_r2hat(x_genotypes, MAF):
    #This calculates exactly the same r2 as the estimated r2hat from minimac4
    #I copy this from Minimac4 source code, exactly as it is in Minimac4
    #Credits to Minimac4 authors
        r2hat=0
        mycount = tf.cast(len(x_genotypes), tf.float64)
        mysum = tf.cast(0, tf.float64)
        mysum_Sq = tf.cast(0, tf.float64)
        
        if(MAF==0): #dont waste time
            #print("r2hat", r2hat)
            return r2hat
        
        for i in range(len(x_genotypes)):
            #print("X", x_genotypes[i])
            if(x_genotypes[i]==0):
                d = 0
            else:                
                d = np.divide(x_genotypes[i],2)
            
            if (d>0.5):
                d = 1-d
                d = tf.cast(d, tf.float64)
            mysum_Sq = sess.run(tf.add(mysum_Sq, tf.multiply(d,d)))
            mysum = sess.run(tf.add(mysum,d))
        
        if(mycount < 2):#return 0
            #print("r2hat", r2hat)
            return r2hat
        myvar = tf.cast(1e-30, tf.float64)
        myf = sess.run(tf.div(mysum,tf.add(mycount,myvar)))
        #print("myf", myf)
        myevar = sess.run( tf.multiply(myf, tf.subtract(1.0,myf) ) )
        #print("myevar", myf)
        
        #mysum_Sq - mysum * mysum / (mycount + 1e-30)
        myovar = sess.run( tf.divide(tf.subtract(mysum_Sq,tf.multiply(mysum,mysum), tf.add(mycount,myvar))) )
        #print("myovar", myovar)

        if(myovar>0):

            myovar = sess.run(tf.divide(myovar, tf.add(mycount,mayvar)))
            r2hat = sess.run(tf.divide(myovar, tf.add(myevar, myvar)))
        
        #print("r2hat", r2hat)

        return r2hat[0]
     
    j=0
    mean_r2hat = 0
    r2hats = []
    #MAFs = calculate_MAF(x, categorical)
    idx = 0
    print(len(x[0]), len(MAF_all_var))
    while j < len(x[0]):
        if(categorical==False):
            getter = operator.itemgetter([j+1])
            x_genotypes = list(map(list, map(getter, np.copy(x))))
            j+=2
        else:
            x_genotypes = []
            for i in range(len(x)):
                x_genotypes.append(np.argmax(x[i][j:j+3]))                
            j+=3
        
        r2hat = calculate_r2hat(x_genotypes, MAF_all_var[idx])
        r2hats.append(r2hat)
        if(r2hat>0):
            mean_r2hat += r2hat/len(MAF_all_var)
        idx += 1 
        if(idx>=len(MAF_all_var)):
            break

    #tf.reset_default_graph()
    sess.close()
                          
    return mean_r2hat, r2hats

#TODO review this and make this work with CE, WCE, and FL
def mean_estimated_r2(x, categorical=False):

    def calculate_r2hat(x_genotypes, MAF):
    #This calculates exactly the same r2 as the estimated r2hat from minimac4
    #I copy this from Minimac4 source code, exactly as it is in Minimac4
    #Credits to Minimac4 authors
        r2hat=0
        mycount = len(x_genotypes)
        mysum = 0
        mysum_Sq = 0
        
        if(MAF==0): #dont waste time
            #print("r2hat", r2hat)
            return r2hat
        
        for i in range(mycount):
            #print("X", x_genotypes[i])
            if(x_genotypes[i]==0):
                d = 0
            else:
                d = np.divide(x_genotypes[i],2)
            
            if (d>0.5):
                d = 1-d
            mysum_Sq += (d*d)
            mysum += d
        
        if(mycount < 2):#return 0
            #print("r2hat", r2hat)
            return r2hat
        
        myf = mysum / (mycount + 1e-30)
        #print("myf", myf)
        myevar = myf * (1.0 - myf)
        #print("myevar", myf)
        myovar = mysum_Sq - mysum * mysum / (mycount + 1e-30)
        #print("myovar", myovar)

        if(myovar>0):

            myovar = myovar / (mycount + 1e-30)
            r2hat = myovar / (myevar + 1e-30)
        
        #print("r2hat", r2hat)

        return r2hat
     
    j=0
    mean_r2hat = 0
    r2hats = []
    #MAFs = calculate_MAF(x, categorical)
    idx = 0
    print(len(x[0]), len(MAF_all_var))
    while j < len(x[0]):
        if(categorical==False):
            getter = operator.itemgetter([j+1])
            x_genotypes = list(map(list, map(getter, np.copy(x))))
            j+=2
        else:
            x_genotypes = []
            for i in range(len(x)):
                x_genotypes.append(np.argmax(x[i][j:j+3]))                
            j+=3
        
        r2hat = calculate_r2hat(x_genotypes, MAF_all_var[idx])
        r2hats.append(r2hat)
        if(r2hat>0):
            mean_r2hat += r2hat/len(MAF_all_var)
        idx += 1 
        if(idx>=len(MAF_all_var)):
            break

    return mean_r2hat, r2hats


def filter_by_MAF(x,y, MAFs, threshold1=0, threshold2=1, categorical=False):
    
    colsum=np.sum(y, axis=0)
    indexes_to_keep = []
    i = 0
    j = 0
    k = 0   
    
    while i < len(MAFs):
        if(MAFs[i]>threshold1 and MAFs[i]<=threshold2):
            if(categorical==True or categorical=="True"):
                if(colsum[j]!=0 or colsum[j+1]!=0 or colsum[j+2]!=0):
                    indexes_to_keep.append(j)
                    indexes_to_keep.append(j+1)
                    indexes_to_keep.append(j+2)
            elif(categorical==False or categorical=="False"):
                if(colsum[k]!=0 or colsum[k+1]!=0):
                    indexes_to_keep.append(k)
                    indexes_to_keep.append(k+1)            
        i += 1
        j += 3
        k += 2
    
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
    
    return filtered_data_x, filtered_data_y

def accuracy_maf_threshold(sess, x, y, MAFs, threshold1=0, threshold2=1, categorical=False):
    

    filtered_data_x, filtered_data_y = filter_by_MAF(x,y, MAFs, threshold1, threshold2, categorical)
    
    correct_prediction = np.equal( np.round( filtered_data_x ), np.round( filtered_data_y ) )
    accuracy_per_marker = np.mean(correct_prediction.astype(float), 0)
    accuracy = np.mean(accuracy_per_marker)

    #correct_prediction = sess.run(tf.equal( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ))
    #accuracy_per_marker = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0))
    #accuracy = sess.run(tf.reduce_mean(accuracy_per_marker))

    return accuracy, accuracy_per_marker

def MSE_maf_threshold(sess, x, y, MAFs, threshold1=0, threshold2=1, categorical=False):
    
    filtered_data_x, filtered_data_y = filter_by_MAF(x,y, MAFs, threshold1, threshold2, categorical)
    
    MSE_per_marker = np.mean(np.square( np.subtract( np.round( filtered_data_x ), np.round( filtered_data_y ) ) ), 0 ) 
    MSE = np.mean( MSE_per_marker )
    
    #MSE_per_marker = sess.run( tf.reduce_mean(tf.square( tf.subtract( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ) ), 0 ) )
    #MSE = sess.run( tf.reduce_mean( MSE_per_marker ) )

    return MSE, MSE_per_marker

def accuracy_maf_threshold_global(sess, x, y, indexes_to_keep):
    
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
       
    correct_prediction = np.equal( np.round( filtered_data_x ), np.round( filtered_data_y ) )
    accuracy_per_marker = np.mean(correct_prediction.astype(float), 0)
    accuracy = np.mean(accuracy_per_marker)

    #correct_prediction = sess.run(tf.equal( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ))
    #accuracy_per_marker = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 0))
    #accuracy = sess.run(tf.reduce_mean(accuracy_per_marker))

    return accuracy, accuracy_per_marker

def MSE_maf_threshold_global(sess, x, y, indexes_to_keep):
    
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
    
    MSE_per_marker = np.mean(np.square( np.subtract( np.round( filtered_data_x ), np.round( filtered_data_y ) ) ), 0 ) 
    MSE = np.mean( MSE_per_marker )
    
    #MSE_per_marker = sess.run( tf.reduce_mean(tf.square( tf.subtract( tf.round( filtered_data_x ), tf.round( filtered_data_y ) ) ), 0 ) )
    #MSE = sess.run( tf.reduce_mean( MSE_per_marker ) )

    return MSE, MSE_per_marker



def flatten_data(sess, myinput, factors=False):
    x = np.copy(myinput)
    if(factors == False): #if shape=3 we are using allele count instead of factors
        x = np.reshape(x, (x.shape[0],-1))
    else:#do one hot encoding, depth=3 because missing (-1) is encoded to all zeroes
        x = (tf.one_hot(indices=x, depth=3))
        x = (tf.layers.flatten(x))#flattening to simplify calculations later (matmul, add, etc)
        x = x.eval()
        #print("Dimensions after flatting", x.shape)
    return x

#Code modified from example
#https://stackoverflow.com/questions/44367010/python-tensor-flow-relu-not-learning-in-autoencoder-task
def run_autoencoder(learning_rate, training_epochs, l1_val, l2_val, act_val, beta, rho, keep_rate, data_obs):

    prep_start = timeit.default_timer()
    
    beta = tf.cast(beta, tf.float64)
    
    print("Running autoencoder.")
    # parameters
    #learning_rate = 0.01
    #training_epochs = 50
    factors = True
    #subjects, SNP, REF/ALT counts
    if(len(data_obs.shape) == 3):
        factors = False

    print("Input data shape:")
    #print(data_masked.shape)
    print(data_obs.shape)
    
    original_shape = data_obs.shape
    
    batch_size = int(round(len(data_obs)/split_size)) # size of training objects split the dataset in 10 parts
    #print(batch_size)
    
    display_step = 1        

    # define layer size
    if(len(data_obs.shape) == 3):
        n_input = len(data_obs[0])*len(data_obs[0][0])
    else:
        n_input = len(data_obs[0])*3     # input features N_variants*subfeatures
        
    n_hidden_1 = int(round(n_input*hsize))  # hidden layer for encoder, equal to input number of features multiplied by a hidden size ratio
    print("Input data shape after coding variables:")
    print(n_input)
    
    # Input placeholders
    #with tf.name_scope('input'):
        #tf input
    X = tf.placeholder("float", [None, n_input], name="X")
    Y = tf.placeholder("float", [None, n_input], name="Y")
#        else:
#            X = tf.placeholder("float", [None, n_input], name="newX")
#            Y = tf.placeholder("float", [None, n_input], name="newY")
            
    #As parameters of a statistical model, weights and biases are learned or estimated by minimizing a loss function that depends on our data. 
    #We will initialize them here, their values will be set during the learning process
    #def define_weights_biases(n_input, n_hidden_1):
    #with tf.device("/gpu:1"):
    with tf.name_scope('weights'):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], dtype=tf.float64), name="w_encoder_h1"),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input], dtype=tf.float64), name="w_decoder_h1"),
            #'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input], dtype=tf.float64), name="w_decoder_h1"),
        }
        variable_summaries(weights['encoder_h1'])
        
    with tf.name_scope('biases'):
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1], dtype=tf.float64), name="b_encoder_b1"),
            'decoder_b1': tf.Variable(tf.random_normal([n_input], dtype=tf.float64), name="b_decoder_b1"),
        }
        variable_summaries(biases['encoder_b1'])
        #weights['encoder_h1']), biases['encoder_b1'])
        
    #print(X.get_shape())
    keep_prob = tf.placeholder("float", None, name="keep_prob") ## RR adding dropout
    
    with tf.name_scope('Wx_plus_b'):
        # construct model
        encoder_op = encoder(X, act_val, l1_val, l2_val, weights, biases, n_hidden_1, keep_rate)
        #encoder_op = encoder(X, act_val, l1_val, l2_val, weights, biases, n_input, keep_rate)
    print(encoder_op)

    tf.summary.histogram('activations', encoder_op)
    
    encoder_result = encoder_op
    y_pred = decoder(encoder_op, act_val, weights, biases)

    #print(encoder_op)
    # predict result
    #y_pred = decoder_op
    # real input data as labels
    y_true = Y
    #TODO experimental Maximal information criteria calculation needs to be implemented
    #M = tf.zeros([n_input, n_input], name="MIC")

    with tf.name_scope('test'):
        X_gs_reshaped = tf.expand_dims(X, 0) 
        X_gs_reshaped = tf.expand_dims(X_gs_reshaped, -1) 
        Y_gs_reshaped = tf.expand_dims(Y, 0)
        Y_gs_reshaped = tf.expand_dims(Y_gs_reshaped, -1)
        pred_gs_reshaped = tf.expand_dims(y_pred, 0)
        pred_gs_reshaped = tf.expand_dims(pred_gs_reshaped, -1)

        tf.summary.image("input", X_gs_reshaped, max_outputs=training_epochs)
        tf.summary.image("ground_truth", Y_gs_reshaped, max_outputs=training_epochs)
        tf.summary.image("prediction_output",pred_gs_reshaped, max_outputs=training_epochs)
    
    #with tf.name_scope('MIC_comming_soon'): #TODO, accelerate and parallelize the calculation of MIC between activations and genotypes, so we can run MIC metrics as a tensor
    #    M_gs_reshaped = tf.expand_dims(M, 0)
    #    M_gs_reshaped = tf.expand_dims(M_gs_reshaped, -1)
    #    tf.summary.image("MIC", M_gs_reshaped, max_outputs=1)
    
    rho_hat = tf.reduce_mean(encoder_op,0) #RR sometimes returns Inf in KL function, caused by division by zero, fixed wih logfun()
    if (act_val == "tanh"):
        rho_hat = tf.div(tf.add(rho_hat,1.0),2.0) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl
    if (act_val == "relu"):
        rho_hat = tf.div(tf.add(rho_hat,1e-10),tf.reduce_max(rho_hat)) # https://stackoverflow.com/questions/11430870/sparse-autoencoder-with-tanh-activation-from-ufldl

    rho = tf.constant(rho) #not necessary maybe?
        
    with tf.name_scope('sparsity'):
        sparsity_loss = tf.reduce_mean(KL_Div(rho, rho_hat))
        sparsity_loss = tf.cast(sparsity_loss, tf.float64)

        sparsity_loss = tf.clip_by_value(sparsity_loss, 1e-10, 1.0, name="sparsity_loss") #RR KL divergence, clip to avoid Inf or div by zero
    tf.summary.scalar('sparsity_loss', sparsity_loss)

    # define cost function, optimizers
    # loss function: MSE # example cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))
    with tf.name_scope('loss'):

        if(loss_type=="MSE"):
            y_true = tf.cast(y_true, tf.float64)
            reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)), name="reconstruction_loss") #RR simplified the code bellow
        elif(loss_type=="Pearson"):
            reconstruction_loss = pearson_r_loss(y_pred, y_true)
        elif(loss_type=="WMSE"):
            reconstruction_loss = weighted_MSE(y_pred, y_true)
        elif(loss_type=="CE"):
            reconstruction_loss = cross_entropy(y_pred, y_true)
        elif(loss_type=="WCE"):
            reconstruction_loss = weighted_cross_entropy(y_pred, y_true)
        elif(loss_type=="FL"):
            reconstruction_loss = focal_loss(y_pred, y_true)
            if(verbose>0):
                mygamma_0, mygamma_1 = calculate_gamma(y_pred, y_true)
                ce0, ce1 = calculate_CE(y_pred, y_true)
                pt0, pt1 = calculate_pt(y_pred, y_true)
                wce = weighted_cross_entropy(y_pred, y_true)
        else:
            y_true = tf.cast(y_true, tf.float64)            
            reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)), name="reconstruction_loss") #RR 
        # newly added sparsity function (RHO, BETA)
        #cost_sparse = tf.multiply(beta,  tf.reduce_sum(KL_Div(RHO, rho_hat)))
        #print(cost_sparse) 

        cost = tf.reduce_mean(tf.add(reconstruction_loss,tf.multiply(beta, sparsity_loss)), name = "cost") #RR simplified equation
        
    tf.summary.scalar('reconstruction_loss_MSE', reconstruction_loss)
    tf.summary.scalar("final_cost", cost)

    #TODO: add a second scaling factor for MAF loss, beta2*MAF_loss
    #or add MAF as another feature instead of adding it to the loss function
    #p-value for fisher exact test or similar type of test as MAF_loss
    #MAF_loss = 0
    #for i in range(0, len(y_pred[0])):
    #        MAF_loss = MAF_loss + stat.fisher(y_pred[][i],y_true[][i])
    #        
    #cost = tf.reduce_mean(reconstruction_loss + beta * sparsity_loss + beta2 * MAF_loss)
    #cost = tf.reduce_mean(cost + cost_sparse)
    
    correct_prediction = 0
    
    if(factors == False): #TODO: only factors==False is working now, TODO: fix the else: bellow
        y_true = tf.cast(y_true, tf.float64)
        correct_prediction = tf.equal( tf.round( y_pred ), tf.round( y_true ) )
    else:
        y_pred_shape = tf.shape(y_pred)
        y_true_shape = tf.shape(y_true)   
        #new_shape = [tf.cast(y_pred_shape[0], tf.int32), tf.cast(n_input/3,  tf.int32), tf.cast(3, tf.int32)]
        #a = tf.cast(y_pred_shape[0], tf.int32)
        #print("shape for calculating accuracy:", a.eval())
        #print(new_shape.eval())
        #new_shape = [tf.shape(y_true_shape)[0],original_shape[1], 3]
        #reshaping back to original form, so the argmax function can work
        y_pred_tmp = tf.reshape(y_pred, [tf.shape(y_true_shape)[0],original_shape[1], 3]) 
        y_true_tmp = tf.reshape(y_true, [tf.shape(y_true_shape)[0],original_shape[1], 3])
        y_pred_tmp = tf.argmax( y_pred_tmp, 1 ) #back to original categorical form [-1,0,1,2]
        y_true_tmp = tf.argmax( y_true_tmp, 1 ) #back to original categorical form [-1,0,1,2]
        #y_true_tmp = tf.argmax( y_true, 1 ) #back to original categorical form [-1,0,1,2]
        #y_pred_tmp = tf.argmax( y_pred, 1 ) #back to original categorical form [-1,0,1,2]        
     
        correct_prediction = tf.equal( y_pred_tmp, y_true_tmp )
        #correct_prediction = tf.equal( y_pred, y_true )
                
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64), name="accuracy")
    cost_accuracy = tf.add((1-accuracy), tf.multiply(beta, sparsity_loss), name="cost_accuracy")
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cost_accuracy', cost_accuracy)

    
    with tf.name_scope('test_accuracy'):
        test_accuracy = tf.Variable(0,name="test_accuracy")
    
    #The RMSprop optimizer is similar to the gradient descent algorithm with momentum. 
    #The RMSprop optimizer restricts the oscillations in the vertical direction. 
    #Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster. 
    #ref: https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b
    if(recovery_mode=="False"):
        #Converge by accuracy when using factors
        with tf.name_scope('train'):
            if(optimizer_type=="RMSProp"):
                if(factors == False):
                    optimizer = tf.train.RMSPropOptimizer(learning_rate, name="optimizer").minimize(cost)
                else:
                    optimizer = tf.train.RMSPropOptimizer(learning_rate, name="optimizer").minimize(cost_accuracy)
            elif(optimizer_type=="GradientDescent"):
                if(factors == False):
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate, name="optimizer").minimize(cost)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate, name="optimizer").minimize(cost_accuracy)
            elif(optimizer_type=="Adam"):
                if(factors == False):
                    optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(cost)
                else:
                    optimizer = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(cost_accuracy)
                    
        #save this optimizer to restore it later.
        #tf.add_to_collection("optimizer", optimizer)
        #tf.add_to_collection("Y", Y)
        #tf.add_to_collection("X", X)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # initialize variables
    #init = tf.global_variables_initializer();
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    start = timeit.default_timer()
    r_report_time = 0
    mask_time = 0
    time_metrics = 0
    gd_time = 0
    # run autoencoder .........
   
    #with tf.device('/device:GPU:0'):  # Replace with device you are interested in
    #    bytes_in_use = BytesInUse()
    
    if(recovery_mode=="True"):
        tf.reset_default_graph()

    
    with tf.Session(config=config) as sess:        
               
        
        if(save_summaries==True):
            
            train_writer = tf.summary.FileWriter('./train', sess.graph)
            valid_writer = tf.summary.FileWriter('./valid')
            test_writer = tf.summary.FileWriter('./test')
            test_writer_low_MAF = tf.summary.FileWriter('./test_low_MAF')
            test_writer_high_MAF = tf.summary.FileWriter('./test_high_MAF')

            low_MAF_summary=tf.Summary()

            high_MAF_summary=tf.Summary()
        
            run_metadata = tf.RunMetadata()

        merged = tf.summary.merge_all()
        
        if(recovery_mode=="True"):
            print("Restoring model from checkpoint")
            #recover model from checkpoint
            meta_path = model_path + '.meta'    
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, model_path)
            #optimizer = graph.get_tensor_by_name("optimizer:0")
            graph = sess.graph
            #scope_name=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            #print("scope name", scope_name)
            optimizer = graph.get_operation_by_name( "train/optimizer" )
            X = graph.get_tensor_by_name("X:0")
            Y = graph.get_tensor_by_name("Y:0")
            #X = "X:0"
            #Y = "Y:0"
            cost = graph.get_tensor_by_name("loss/cost:0")
            #cost = "cost:0"
            reconstruction_loss = graph.get_tensor_by_name("loss/reconstruction_loss:0")
            #reconstruction_loss = "reconstruction_loss:0"
            #sparsity_loss = "sparsity_loss:0"            
            sparsity_loss = graph.get_tensor_by_name("sparsity/sparsity_loss:0")
            accuracy = graph.get_tensor_by_name("accuracy:0")

            #accuracy = "accuracy:0"
            #cost_accuracy = "cost_accuracy:0"
            cost_accuracy = graph.get_tensor_by_name("cost_accuracy:0")
            y_pred = graph.get_tensor_by_name("y_pred:0")
            
            tf.summary.scalar('reconstruction_loss_MSE', reconstruction_loss)
            tf.summary.scalar("final_cost", cost)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('cost_accuracy', cost_accuracy)
            tf.summary.scalar('sparsity_loss', sparsity_loss)
            
            weights = {
                'encoder_h1': graph.get_tensor_by_name("weights/w_encoder_h1:0"),
                'decoder_h1': graph.get_tensor_by_name("weights/w_decoder_h1:0"),
            }
            variable_summaries(weights['encoder_h1'])
        
            biases = {
                'encoder_b1': graph.get_tensor_by_name("biases/b_encoder_b1:0"),
                'decoder_b1': graph.get_tensor_by_name("biases/b_decoder_b1:0"),
            }
            variable_summaries(biases['encoder_b1'])
            
            decoder_op = y_pred
            #encoder_op = graph.get_tensor_by_name("Wx_plus_b/activation:0")
            
            encoder_op =  graph.get_tensor_by_name("Wx_plus_b/dense/BiasAdd:0")
            tf.summary.histogram('activations', encoder_op)
                       
            
            #M = tf.zeros([n_input, n_input], name="MIC")

            with tf.name_scope('test'):
                X_gs_reshaped = tf.expand_dims(X, 0) 
                X_gs_reshaped = tf.expand_dims(X_gs_reshaped, -1) 
                Y_gs_reshaped = tf.expand_dims(Y, 0)
                Y_gs_reshaped = tf.expand_dims(Y_gs_reshaped, -1)
                pred_gs_reshaped = tf.expand_dims(y_pred, 0)
                pred_gs_reshaped = tf.expand_dims(pred_gs_reshaped, -1)

                tf.summary.image("input", X_gs_reshaped, max_outputs=training_epochs)
                tf.summary.image("ground_truth", Y_gs_reshaped, max_outputs=training_epochs)
                tf.summary.image("prediction_output",pred_gs_reshaped, max_outputs=training_epochs)
    
            with tf.name_scope('MIC_comming_soon'): #TODO, accelerate and parallelize the calculation of MIC between activations and genotypes, so we can run MIC metrics as a tensor
                M_gs_reshaped = tf.expand_dims(M, 0)
                M_gs_reshaped = tf.expand_dims(M_gs_reshaped, -1)
                tf.summary.image("MIC", M_gs_reshaped, max_outputs=1)
            
            merged = tf.summary.merge_all()
            #y_pred = "y_pred:0"
            print("Model restored")
        else:    
            sess.run(init)

        prep_stop = timeit.default_timer()
        prep_time = prep_stop-prep_start
        
        mean_cost = 0
        mean_rloss = 0
        mean_sloss = 0
        mean_acc = 0
        mean_cacc = 0   
        mean_F1 = [0,0,0]
        ki = 0
        mean_acc_t = [0,0,0,0,0,0,0]
        my_acc_t = [0,0,0,0,0,0,0]


        data_idx = 0        

        if(kn>=2):
            kf = KFold(n_splits=kn)
        else:
            kf = KFold(n_splits=10)    
        
        mask_start = timeit.default_timer()

        if(disable_masking==False):
            data_masked = np.zeros(data_obs.shape)
            #print(data_obs.shape)
            data_idx = kf.split(data_masked, data_obs)
        else:
            data_idx = kf.split(data_obs, data_obs)

            
        mask_rate = 0
        indexes = list(range(len(data_obs)))        
       
        if(do_parallel_masking==True and disable_masking==False):
            
            chunks = chunk(indexes, ncores )
            if(ncores>len(indexes)):
                chunks = list(range(len(data_obs)))
               
       
        if(fixed_masking == True and disable_masking==False): #mask all data only once before the whole training procedure
            if(do_parallel_masking==True):
                pool = multiprocessing.Pool(ncores)            
            if(initial_masking_rate > 0):
                mask_rate = initial_masking_rate
            else:
                mask_rate = 0.9 #set default if no value is provided
                
            if(mask_per_sample == False):
                if(do_parallel_masking==False):
                    data_masked, _ = mask_data(indexes, np.copy(data_obs), mask_rate,categorical)
                else:
                    data_masked = pool.map(partial(mask_data, mydata=np.copy(data_obs), mask_rate=mask_rate, categorical=categorical),chunks)
            else:
                if(do_parallel_masking==False):
                    data_masked = mask_data_per_sample(np.copy(data_obs), mask_rate, categorical)
                else:
                    data_masked = pool.map(partial(mask_data_per_sample, mydata=np.copy(data_obs), mask_rate=mask_rate, categorical=categorical),chunks)
            
            if(do_parallel_masking==True):
                pool.close()
                pool.join()
                #print("SHAPE ", len(data_masked), len(data_masked[0]), len(data_masked[0][0]))
                if(mask_per_sample == False):                    
                    data_masked = [result[0] for result in data_masked]
                data_masked = [item for sublist in data_masked for item in sublist]
                #print("SHAPE ", len(data_masked), len(data_masked[0]))

            mask_stop = timeit.default_timer()
            print("Time to run masking: ", mask_stop-mask_start)
            mask_time += mast_stop-mask_start

            if(disable_masking==False):
                data_masked = flatten_data(sess, np.copy(data_masked), factors)
                
            data_obs = flatten_data(sess, np.copy(data_obs), factors)
            
            
            
            print(data_masked.shape)
            

        for train_idx, val_idx in data_idx:
            
            if(kn>=2):
                if(fixed_masking == True and disable_masking==False):
                    train_x = data_masked[train_idx]
                    val_x = data_masked[val_idx]
                else: #if fixed masking is false mask later at the begining of each epoch
                    train_x = data_obs[train_idx]
                    val_x = data_obs[val_idx]
                    
                train_y = data_obs[train_idx]
                val_y = data_obs[val_idx]
            else:
                if(fixed_masking == True and disable_masking==False):
                    train_x = np.copy(data_masked)
                    del data_masked
                else:
                    train_x = np.copy(data_obs)                    
                train_y = np.copy(data_obs)

            total_batch = int(train_x.shape[0] / batch_size)
            print(train_x.shape)
            
            ki += 1
            
            if(fixed_masking_rate==True):
                mask_rate = initial_masking_rate
            
            time_epochs = 0
            epoch=0
            cycle_count = -1

            for iepoch in range(training_epochs+1):
                
                if(recovery_mode=="True"):
                    epoch=iepoch+resuming_step
                else:
                    epoch=iepoch
                    
                start_epochs = timeit.default_timer()
                mask_start = timeit.default_timer()
                                
                if(fixed_masking == False and disable_masking==False):
                    if(do_parallel_masking==True): 
                        pool = multiprocessing.Pool(ncores)
                    
                    if(fixed_masking_rate==False and mask_rate<maximum_masking_rate and cycle_count==repeat_cycles):
                        mask_rate += initial_masking_rate
                        
                    if(cycle_count==repeat_cycles):
                        cycle_count = 0
                    else:
                        cycle_count += 1
                    #make new masking on every new iteration
                    if(epoch>=0):#Change from >= to == if you want to mask only at epoch 0, experimental, remove later
                        if(mask_per_sample == True):
                            #print(train_x.shape)
                            if(do_parallel_masking==False):
                                data_masked = mask_data_per_sample(indexes, np.copy(data_obs),mask_rate,categorical)
                            else:
                                data_masked = pool.map(partial(mask_data_per_sample, mydata=np.copy(data_obs), mask_rate=mask_rate, categorical=categorical),chunks)

                        else:
                            if(do_parallel_masking==False):                                   
                                data_masked, _ = mask_data(indexes, np.copy(data_obs),mask_rate,categorical)
                            else:
                                data_masked = pool.map(partial(mask_data, mydata=np.copy(data_obs), mask_rate=mask_rate, categorical=categorical),chunks)
                        if(do_parallel_masking==True):
                            pool.close()
                            pool.join()
                            #print("SHAPE ", len(data_masked), len(data_masked[0]), len(data_masked[0][0]))
                            if(mask_per_sample == False):                    
                                data_masked = [result[0] for result in data_masked]
                            data_masked = [item for sublist in data_masked for item in sublist]
                            #print("SHAPE ", len(data_masked), len(data_masked[0]))

                        if(kn>=2):
                            train_y = np.copy(data_obs[train_idx])
                            train_x = data_masked[train_idx]  
                            val_y = np.copy(data_obs[val_idx])
                            val_x = data_masked[val_idx]    
                            val_x = flatten_data(sess, np.copy(val_x), factors)
                            val_y = flatten_data(sess, np.copy(val_y), factors)                                
                        else:
                            
                            train_x = data_masked
                            train_y = np.copy(data_obs)
                            

                mask_stop = timeit.default_timer()
                mask_time += mask_stop-mask_start
                
                #after masking, flatten data
                train_x = flatten_data(sess, np.copy(train_x), factors)
                train_y = flatten_data(sess, np.copy(train_y), factors)
                    
                if(kn==1 and shuffle==True):
                    randomize = np.arange(len(train_x))
                    np.random.shuffle(randomize)
                    train_x = train_x[randomize]
                    train_y = train_y[randomize]
                    
                                        
                for i in range(total_batch):
                    batch_x = train_x[i*batch_size:(i+1)*batch_size]
                    batch_y = train_y[i*batch_size:(i+1)*batch_size]
                    
                    #calculate cost and optimizer functions                    
                    if(i!=(total_batch-1) or last_batch_validation==False):
                        gd_start = timeit.default_timer()
                        if(save_summaries==True):
                            if(verbose>0):
                                summary, _, c, a, rl, g0, g1, myce0, myce1,mypt0, mypt1, mywce,myy = sess.run([merged,optimizer, cost, accuracy, reconstruction_loss, mygamma_0, mygamma_1, ce0, ce1,pt0, pt1, wce, y_pred], feed_dict={X: batch_x, Y: batch_y} )     
                            else:
                                summary, _, c, a, rl  = sess.run([merged,optimizer, cost, accuracy, reconstruction_loss], feed_dict={X: batch_x, Y: batch_y} )     
                            train_writer.add_run_metadata(run_metadata, 'k%03d-step%03d-batch%04d' % (ki, epoch, i) )
                            train_writer.add_summary(summary, epoch)
                        else:
                            if(verbose>0):
                                _, c, a, rl, g0, g1, myce0, myce1,mypt0, mypt1, mywce,myy = sess.run([optimizer, cost, accuracy, reconstruction_loss, mygamma_0, mygamma_1, ce0, ce1, pt0, pt1, wce, y_pred], feed_dict={X: batch_x, Y: batch_y} )     
                            else:
                                _, c, a, rl = sess.run([optimizer, cost, accuracy, reconstruction_loss], feed_dict={X: batch_x, Y: batch_y} )     
                        gd_stop = timeit.default_timer()
                        gd_time += gd_stop-gd_start
                            
                    else:
                        if(save_summaries==True):
                            mysummary, a, c, _, _, rl  = sess.run([merged,accuracy, cost, y_pred, encoder_op, reconstruction_loss], feed_dict={X: batch_x, Y: batch_y} )                
                            valid_writer.add_summary(mysummary, epoch )
                        else:
                            a, c, rl = sess.run([accuracy, cost, reconstruction_loss], feed_dict={X: batch_x, Y: batch_y} )
                    
                if(save_pred==True and ki==1): #if using cluster run for all k fold: if(save_pred==True):
                    #This generates 1GB files per epoch per k-fold iteration, at least 5TB of space required, uncoment when using the HPC cluster with big storage
                    #my_pred = sess.run([y_pred], feed_dict={X: train_x, Y: train_y} )
                    #print(my_pred.shape) #3D array 1,features,samples
                    #fname = "k-" + str(ki) + "_epoch-" + str(epoch+1) + "_train-pred.out"
                    #with open(fname, 'w') as f:
                    #    np.savetxt(f, my_pred[0])
                    #f.close()

                    my_pred = sess.run([y_pred], feed_dict={X: val_x, Y: val_y} )
                    
                    fname = "k-" + str(ki) + "_epoch-" + str(epoch+1) + "_val-pred.out"
                    with open(fname, 'w') as f:
                        np.savetxt(f,  my_pred[0])
                    f.close()
                
                
                if(kn>=2):#if running k-fold CV
                    if(ki==1): #save only first batch in k-fold CV to not overload tensorboard
                        if(save_summaries==True):
                            mysummary, _, _, genotypes, activations = sess.run([merged,X,y_pred,y_true, encoder_op], feed_dict={X: train_x, Y: train_y} )
                            valid_writer.add_summary(mysummary, epoch ) 
                        
                            
                                      

                #print("Calculating MIC") 
                #mic_c, tic_c =  cstats(genotypes, activations, est="mic_e")
                #M.assign(mic_e).eval() # assign_sub/assign_add is also available.
                #mysummary, _ = sess.run([merged, M] )
                #valid_writer.add_summary(mysummary, epoch )

                                      
################Epoch end here
                if(verbose>0):
                    print("Epoch ", epoch, " done. Masking rate used:", mask_rate, " Initial one:", initial_masking_rate, " Loss: ", c, " Accuracy:", a, " Reconstruction loss (" , loss_type, "): ", rl, "ce0:", myce0, "ce1:", myce1)
                    print("Shape pt0:", mypt0)
                    print("Shape pt1:", mypt1)
                    print("Shape g0:", g0)
                    print("Shape g1:", g1)
                    print("Shape ce0:", myce0)
                    print("Shape ce1:", myce1)
                    print("wce:", mywce)
                    print("fl:", rl)
                    print("myy:", myy)
                else:
                    print("Epoch ", epoch, " done. Masking rate used:", mask_rate, " Initial one:", initial_masking_rate, " Loss: ", c, " Accuracy:", a, " Reconstruction loss (" , loss_type, "): ", rl)

                
                if(save_model==True):
                    #Create a saver object which will save all the variables
                    saver = tf.train.Saver(max_to_keep=2)
        
                    #Now, save the graph
                    filename='./inference_model-' + str(ki) + ".ckpt"
                    print("Saving model to file:", filename)
                    saver.save(sess, filename)
                
                if(test_after_train_step==True and ( ((epoch+1) % (repeat_cycles+1) == 0) or epoch == 0) ):
                    test_summary, test_y_pred, test_accuracy, test_loss, test_cost, newX, newY = sess.run([merged, y_pred, accuracy, reconstruction_loss, cost, X, Y], feed_dict={X: testing_input, Y: testing_ground_truth})
                    #print(testing_input[0][0:300])
                    #print(testing_ground_truth[0][0:300])
                    if(save_summaries==True):
                        test_writer.add_summary(test_summary, epoch )

                    test_accuracy, acc_per_m = accuracy_maf_threshold(sess, test_y_pred, testing_ground_truth, test_MAF_all_var, 0, 1, categorical)
                        
                    print("Test accuracy:", test_accuracy, "Variants used for testing:", len(acc_per_m))
                    
                    new_accuracy, acc_per_m = accuracy_maf_threshold(sess, test_y_pred, testing_ground_truth, test_MAF_all_var, 0, 0.005, categorical)
                    
                    #new_MSE, _ = MSE_maf_threshold(sess, test_y_pred, testing_ground_truth, test_MAF_all_var, 0, 0.005, categorical)
                    new_F1 = f1_score(test_y_pred, testing_ground_truth, sess)
                    
                    if(save_summaries==True):
                        test_summary=tf.Summary()
                    
                        test_summary.value.add(tag='accuracy_1', simple_value = new_accuracy)
                        test_summary.value.add(tag='reconstruction_loss_MSE', simple_value = new_MSE)
                        test_summary.value.add(tag='F1_score_micro', simple_value = new_F1[0])
                        test_summary.value.add(tag='F1_score_macro', simple_value = new_F1[1])
                        test_summary.value.add(tag='F1_score_weighted', simple_value = new_F1[2])
                        
                        test_writer_low_MAF.add_summary(test_summary, epoch)

                                                            
                    print("Test accuracy [0-0.005]:", new_accuracy, " Variants in MAF threshold:", len(acc_per_m)) 
                                        
                    new_accuracy, acc_per_m = accuracy_maf_threshold(sess, test_y_pred, testing_ground_truth, test_MAF_all_var, 0.005, 1, categorical)
                    #new_MSE, _ = MSE_maf_threshold(sess, test_y_pred, testing_ground_truth, test_MAF_all_var, 0.005, 1, categorical)
                    

                    if(save_summaries==True):
                        test_summary=tf.Summary()
                    
                        test_summary.value.add(tag='accuracy_1', simple_value = new_accuracy)
                        test_summary.value.add(tag='reconstruction_loss_MSE', simple_value = new_MSE)
                        test_summary.value.add(tag='F1_score_micro', simple_value = new_F1[0])
                        test_summary.value.add(tag='F1_score_macro', simple_value = new_F1[1])
                        test_summary.value.add(tag='F1_score_weighted', simple_value = new_F1[2])

                        test_writer_high_MAF.add_summary(test_summary, epoch)

                    
                    print("Test accuracy [0.005-1]:", new_accuracy, " Variants in MAF threshold:", len(acc_per_m)) 

                    print("Test F1 scores (micro, macro, weighted):", new_F1)

                    
                    
                    #test_summary, test_y_pred, test_accuracy, test_MSE, test_cost, newX, newY = sess.run([merged, y_pred, accuracy, reconstruction_loss, cost, X, Y], feed_dict={X: testing_input_high_MAF, Y: testing_ground_truth_high_MAF})
                    #test_writer_high_MAF.add_summary(test_summary, epoch )
           
                    #print("Test loss:", test_cost, " Test accuracy [0.005-1]:", test_accuracy) 
                    
                stop_epochs = timeit.default_timer()
                time_epochs += stop_epochs-start_epochs
                
        
            if(save_pred==True and kn>=2): #if kn<2, validation and training are the same
                fname = "k-" + str(ki) + "_val-obs.out"
                with open(fname, 'w') as f:
                    np.savetxt(f, val_y)
                f.close()

                fname = "k-" + str(ki) + "_val-input.out"
                with open(fname, 'w') as f:
                    np.savetxt(f, val_x)
                f.close()

            if(save_pred==True):

                fname = "k-" + str(ki) + "_train-obs.out"
                with open(fname, 'w') as f:
                    np.savetxt(f, train_y)
                f.close()

                fname = "k-" + str(ki) + "_train-input.out"
                with open(fname, 'w') as f:
                    np.savetxt(f, train_x)
                f.close()
            
            if(detailed_metrics==True):
                print("Calculating summary statistics...")
            else:
                print("Skipping detailed statistics...")
                
            start_metrics = timeit.default_timer()

            if(kn>=2):
                if(save_summaries==True):
                    
                    mysummary, my_cost, my_rloss, my_sloss, my_acc, my_cacc  = sess.run([merged, cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy], feed_dict={X: val_x, Y: val_y})
                
                    valid_writer.add_summary(mysummary, epoch )
                else:
                    my_cost, my_rloss, my_sloss, my_acc, my_cacc  = sess.run([cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy], feed_dict={X: val_x, Y: val_y})
                    
                if(detailed_metrics==True):
                    my_pred = sess.run([y_pred], feed_dict={X: val_x, Y: val_y})
                    my_pred = my_pred[0]
                
                    print("Accuracy per veriant...")
                    my_acc_t[0], acc_per_m = accuracy_maf_threshold(sess, my_pred, val_y, MAF_all_var, 0, 1, factors)
                    #my_acc_t[1], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0, 0.005, factors)
                    #my_acc_t[2], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0.005, 1, factors)
                    #my_acc_t[3], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0, 0.001, factors)
                    #my_acc_t[4], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0.001, 1, factors)
                    #my_acc_t[5], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0, 0.01, factors)
                    #my_acc_t[6], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0.01, 1, factors)
                    print("MSE per veriant...")
                    my_MSE, MSE_list = MSE_maf_threshold(sess, my_pred, val_y, MAF_all_var, 0, 1, factors)

                    #my_MAF_list = calculate_MAF(val_y, factors)
                    print("Estimated r2hat per veriant...")
                    mean_r2hat_est, my_r2hat_est = mean_estimated_r2(val_y, factors)
                    print("Empirical r2hat per veriant...")
                    mean_r2hat_emp, my_r2hat_emp = mean_empirical_r2(val_x, val_y, factors)


            else:
                if(save_summaries==True):
                    mysummary, my_cost, my_rloss, my_sloss, my_acc, my_cacc  = sess.run([merged, cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy], feed_dict={X: train_x, Y: train_y})
                
                    valid_writer.add_summary(mysummary, epoch )
                if(full_train_report==False):
                    my_cost = 0
                    my_rloss = 0
                    my_sloss = 0
                    my_acc = 0
                    my_cacc = 0
                    my_F1 = [0,0,0]
                    my_acc_t = [0,0,0,0]

                    for i in range(total_batch):
                        print("Calculating F1 for batch", i)
                        batch_x = train_x[i*batch_size:(i+1)*batch_size]
                        batch_y = train_y[i*batch_size:(i+1)*batch_size]
                        
                        my_cost_tmp, my_rloss_tmp, my_sloss_tmp, my_acc_tmp, my_cacc_tmp, my_pred  = sess.run([cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy, y_pred], feed_dict={X: batch_x, Y: batch_y})
                        my_cost += my_cost_tmp/total_batch
                        my_rloss += my_rloss_tmp/total_batch
                        my_sloss += my_sloss_tmp/total_batch
                        my_acc += my_acc_tmp/total_batch
                        my_cacc += my_cacc_tmp/total_batch
                        
                        my_F1_tmp = f1_score(my_pred, batch_y, sess)
                        
                        my_F1[0] += my_F1_tmp[0]/total_batch
                        my_F1[1] += my_F1_tmp[1]/total_batch
                        my_F1[2] += my_F1_tmp[2]/total_batch
                        
                        #RARE VS COMMON VARS
                        print("Calculating accuracy per MAF threshold for batch", i)
                        if(report_perf_by_rarity==True):
                            #my_acc_tmp, _ = accuracy_maf_threshold(sess, my_pred, batch_y, MAF_all_var, 0.005, 1, factors)
                            #my_acc_t[0] += my_acc_tmp/total_batch
                            #my_acc_tmp, _ = accuracy_maf_threshold(sess, my_pred, batch_y, 0, 0.005, factors)
                            #my_acc_t[1] += my_acc_tmp/total_batch

                            my_acc_tmp, _ = accuracy_maf_threshold_global(sess, my_pred, batch_y, rare_indexes)
                            my_acc_t[2] += my_acc_tmp/total_batch
                            my_acc_tmp, _ = accuracy_maf_threshold_global(sess, my_pred, batch_y, common_indexes)
                            my_acc_t[3] += my_acc_tmp/total_batch
                        else:
                            my_acc_t[0], my_acc_t[1],my_acc_t[2],my_acc_t[3] = "NA","NA", "NA", "NA"
                          

                else:
                    my_cost, my_rloss, my_sloss, my_acc, my_cacc,my_pred  = sess.run([cost, reconstruction_loss, sparsity_loss, accuracy, cost_accuracy,y_pred], feed_dict={X: train_x, Y: train_y})
                    my_F1 = f1_score(my_pred, train_y, sess)
                    if(report_perf_by_rarity==True):
                        #my_acc_t[0], _ = accuracy_maf_threshold(sess, my_pred, val_y, MAF_all_var, 0.005, 1, factors)
                        #my_acc_t[1], _ = accuracy_maf_threshold(sess, my_pred, val_y, 0, 0.005, factors)
                        my_acc_t[2], _ = accuracy_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0, 0.01, factors)
                        my_acc_t[3], _ = accuracy_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0.01, 1, factors) 
                    else:
                        my_acc_t[0], my_acc_t[1],my_acc_t[2],my_acc_t[3] = "NA","NA", "NA", "NA"
 
                r_report_start = timeit.default_timer()
                
                print("Accuracy [MAF 0-0.01]:", my_acc_t[2])
                print("Accuracy [MAF 0.01-1]:", my_acc_t[3])
                print("F1 score [MAF 0-1]:", my_F1)
              
                r_report_stop = timeit.default_timer()
                print("Time to calculate accuracy (rare versus common variants:", r_report_stop-r_report_start)
                r_report_time += r_report_stop-r_report_start
               
                if(detailed_metrics==True):
                
                    #my_pred = sess.run([y_pred], feed_dict={X: train_x, Y: train_y})
                    my_pred = sess.run([y_pred], feed_dict={X: train_x, Y: train_y})
                    my_pred = my_pred[0]
                    #print("my_pred")
                    #print(len(my_pred))
                    #print(len(my_pred[0]))
                    #print("my_train")
                    #print("LENGTH", len(MAF_all_var))

                    #print(len(train_x))
                    #print(len(train_x[0]))
                    print("Accuracy per veriant...")
                    my_acc_t[0], acc_per_m = accuracy_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0, 1, factors)
                    #my_acc_t[1], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0, 0.005, factors)
                    #my_acc_t[2], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0.005, 1, factors)
                    #my_acc_t[3], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0, 0.001, factors)
                    #my_acc_t[4], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0.001, 1, factors)
                    #my_acc_t[5], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0, 0.01, factors)
                    #my_acc_t[6], _ = accuracy_maf_threshold(sess, my_pred, train_y, 0.01, 1, factors)

                    print("MSE per veriant...")
                    my_MSE, MSE_list = MSE_maf_threshold(sess, my_pred, train_y, MAF_all_var, 0, 1, factors)

                    print("Estimated r2hat per veriant...")
                    mean_r2hat_est, my_r2hat_est = mean_estimated_r2(train_y, factors)
                    print("Emprirical r2hat per veriant...")
                    mean_r2hat_emp, my_r2hat_emp = mean_empirical_r2(train_x, train_y, factors)
                               
            print("Maximum VRAM used: ")
            # maximum across all sessions and .run calls so far
            #print(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
            #print(sess.run(bytes_in_use))

            #print("MAF", len(my_MAF_list), "acc", len(acc_per_m), "r2emp", len(my_r2hat_emp), "r2est", len(my_r2hat_emp), "MSE", len(MSE_list))
            if(detailed_metrics==True):
                idx=1
                for i in range(len(MAF_all_var)):
                    print("METRIC_MAF:", MAF_all_var[i])
                    print("METRIC_acc_per_m:", acc_per_m[idx])
                    print("METRIC_r2_emp:", my_r2hat_est[i])
                    print("METRIC_r2_est:", my_r2hat_emp[i])
                    print("METRIC_MSE_per_m:", MSE_list[idx])
                    idx+=2

                
            mean_cost += my_cost/kn
            mean_rloss += my_rloss/kn
            mean_sloss += my_sloss/kn
            mean_acc += my_acc/kn
            mean_cacc += my_cacc/kn
            mean_F1[0] += my_F1[0]/kn
            mean_F1[1] += my_F1[1]/kn
            mean_F1[2] += my_F1[2]/kn
            
            if(detailed_metrics==True):
                for j in range(len(mean_acc_t)):
                    mean_acc_t[j] += my_acc_t[j]/kn
            
            stop_metrics = timeit.default_timer()
            
            time_metrics += stop_metrics-start_metrics
            
            if(kn<=1):
                print("Training done, not running CV.")
                break
            else:            
                print("K-fold iteration: ", ki, " of ", split_size, ".")
        
        #print("Mean accuracy for MAF 0-1 range:", mean_acc_t[0])
        #print("Mean accuracy for MAF 0-0.005 range:", mean_acc_t[1])
        ##print("Mean accuracy for MAF 0.005-1 range:", mean_acc_t[2])
        #print("Mean accuracy for MAF 0-0.001 range:", mean_acc_t[3])
        #print("Mean accuracy for MAF 0.001-1 range:", mean_acc_t[4])
        #print("Mean accuracy for MAF 0-0.01 range:", mean_acc_t[5])
        #print("Mean accuracy for MAF 0.01-1 range:", mean_acc_t[6])
        #print("MSE MAF 0-1 range:", my_MSE)
        #print("r2hat est. MAF 0-1 range:", mean_r2hat_est)
        #print("r2hat emp. MAF 0-1 range:", mean_r2hat_emp)


        #pbar.close()

        print("Optimization finished!")
        
    stop = timeit.default_timer()
    print('Time to run all training (sec): ', stop - start)
    print('Time to run all epochs (sec): ', time_epochs)
    print('Time to run each epoch, average (sec): ', time_epochs/(training_epochs*kn))
    print('Time to run all accuracy metrics (sec): ', time_metrics)
    print('Time to run each accuracy metric (sec): ', time_metrics/kn)
    print('Time to run each k-fold step (sec): ', (stop - start)/kn)
    print('Time to run all gradient descent iteratons (GPU): ', gd_time)
    print('Time to run all masking iteratons (CPU): ', mask_time)    
    print('Time to run performance per MAF threshold calculations (CPU<->GPU):', r_report_time)
    print('Time to define and start graph/session (CPU->GPU): ', prep_time)    

    
    if(save_summaries==True):
        train_writer.close()
        valid_writer.close()   
        test_writer.close() 
        test_writer_low_MAF.close()   
        test_writer_high_MAF.close()   

    #reset tensorflow session
    tf.reset_default_graph()
    sess.close()
    # return the minimum loss of this combination of L1, L2, activation function, beta, rho on a dataset
    #return minLoss, min_reconstruction_loss, min_sparsity_loss, max_correl
    return mean_cost, my_rloss, mean_sloss, mean_acc, my_acc_t[2], my_acc_t[3], mean_F1[0], mean_F1[1], mean_F1[2]


# In[ ]:
def load_test_data():
    if(test_after_train_step==True):
        print("Loading and mapping testing data input file...")
        
        global testing_input
        global testing_ground_truth
        global testing_input_low_MAF
        global testing_input_high_MAF
        global testing_ground_truth_low_MAF
        global testing_ground_truth_high_MAF

        
        testing_input = process_testing_data(test_position_path, test_input_path) #new_df

        print("Loading and mapping testing data ground truth file...")
        testing_ground_truth = process_testing_data(test_position_path, test_ground_truth_path, ground_truth=True) #new_df_obs
        #print(testing_ground_truth[0][0:300])

        
        
        with tf.Session(config=config) as my_sess:
            testing_input = flatten_data(my_sess, testing_input)
            testing_ground_truth = flatten_data(my_sess, testing_ground_truth)
            #print(testing_input[0][0:300])
            #print(testing_ground_truth[0][0:300])
        
        tf.reset_default_graph()
        my_sess.close()
        print("Filtering test data by MAF thresholds [0-0.005] and [0.005-1]...")
        testing_input_low_MAF,testing_ground_truth_low_MAF = filter_by_MAF(testing_input, testing_ground_truth, test_MAF_all_var, 0,0.005, categorical)
        testing_input_high_MAF,testing_ground_high_low_MAF = filter_by_MAF(testing_input, testing_ground_truth, test_MAF_all_var, 0.005,1, categorical)
        
        

def main():
    keep_rate = 1
    #split_size = 10 #k for 10-fold cross-validation
    
    print("This is the name of the script: ", sys.argv[0])
    print("Number of arguments: ", len(sys.argv))
    print("The arguments are: " , str(sys.argv))
    #mask_rate=0.9
    
    #kf = KFold(n_splits=split_size)      
    global recovery_mode
    global initial_masking_rate
    global maximum_masking_rate
    global disable_masking
    global fixed_masking_rate   
        
    global gamma
    global loss_type
    global optimizer_type      
    
    if(len(sys.argv)==6):
        print("Parsing input file: ")
            #Arguments
            #sys.argv[1] = [str] input file (HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf)
            #sys.argv[2] = [str] hyperparameter list file (mylist.txt)
            #sys.argv[3] = [True,False] Recovery mode, default is False
            #sys.argv[4] = [1/846] Initial masking rate               
            #sys.argv[5] = [0.98] Final masking rate
        
        recovery_mode = sys.argv[3]
        initial_masking_rate = convert_to_float(sys.argv[4])
        maximum_masking_rate = convert_to_float(sys.argv[5])

        if(maximum_masking_rate==0):
            disable_masking = True
        else:
            disable_masking = False

        if(maximum_masking_rate==initial_masking_rate):
            fixed_masking_rate = True
        else:
            fixed_masking_rate = False
        
        print("disable_masking =", disable_masking)

        print("fixed_masking_rate =", fixed_masking_rate)
        
        print("initial_masking_rate =", initial_masking_rate)
        
        print("maximum_masking_rate =", maximum_masking_rate)


        print("reading hyperparameter list from file: ", sys.argv[2])
        hp_array = []
        result_list = []

        with open(sys.argv[2]) as my_file:
            for line in my_file:                
                hp_array.append(line.split())
        i = 0

        
        while(i < len(hp_array)):
            
            l1 = float(hp_array[i][0])
            l2 = float(hp_array[i][1])
            beta = float(hp_array[i][2])
            rho = float(hp_array[i][3])
            act = str(hp_array[i][4])
            lr = float(hp_array[i][5])
            gamma = float(hp_array[i][6])
            optimizer_type = str(hp_array[i][7])
            loss_type = str(hp_array[i][8])
            
            if(i==0):
                data_obs = process_data(sys.argv[1],categorical) #input file, i.e: HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf
                load_test_data()



            tmp_list = hp_array[i]
            print("Starting autoencoder training... Parameters:", tmp_list)
            my_cost, my_rloss, my_sloss, my_acc,  my_racc, my_cacc, my_micro, my_macro, my_weighted  = run_autoencoder(lr, training_epochs, l1, l2, act, beta, rho, keep_rate, data_obs)
            tmpResult = [my_cost, my_rloss, my_sloss, my_acc,  my_racc, my_cacc, my_micro, my_macro, my_weighted]
            tmp_list.extend(tmpResult)
            result_list.append(tmp_list)
            i += 1
            print("TMP_RESULT: ", tmp_list)
        return result_list
            
    else:   
        #Arguments
        #sys.argv[1] = [str] input file (HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf)
        #sys.argv[2] = [float] L1 hyperparameter value
        #sys.argv[3] = [float] L2 hyperparameter value
        #sys.argv[4] = [float] Sparsity beta hyperparameter value
        #sys.argv[5] = [float] Sparseness rho hyperparameter value
        #sys.argv[6] = [str] Activation function ('anh')
        #sys.argv[7] = [float] Learning rate hyperparameter value
        #sys.argv[8] = [float] gamma hyper parameter value
        #sys.argv[9] = [string] optimizer type
        #sys.argv[10] = [string] loss function type
        #sys.argv[11] = [True,False] Recovery mode is False
        #sys.argv[12] = [1/846] Initial masking rate
        #sys.argv[13] = [float] Final masking rate

        l1 = float(sys.argv[2])
        l2 = float(sys.argv[3])
        beta = float(sys.argv[4])
        rho = float(sys.argv[5])
        act = str(sys.argv[6])
        lr = float(sys.argv[7])
        gamma = float(sys.argv[8])
        optimizer_type = str(sys.argv[9])
        loss_type = str(sys.argv[10])

        data_obs = process_data(sys.argv[1],categorical) #input file, i.e: HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf
        load_test_data()
       
        #grid = make_grid(l1_arr, l2_arr, beta_arr, rho_arr, act_arr,learning_rate_arr,N)
        tmp_list = sys.argv[2:]

        recovery_mode = sys.argv[11]
        initial_masking_rate = convert_to_float(sys.argv[12])
        maximum_masking_rate = convert_to_float(sys.argv[13])

        if(maximum_masking_rate==0):
            disable_masking = True
        else:
            disable_masking = False

            
        if(maximum_masking_rate==initial_masking_rate):
            fixed_masking_rate = True
        else:
            fixed_masking_rate = False

        print("disable_masking =", disable_masking)

        print("fixed_masking_rate =", fixed_masking_rate)
        
        print("initial_masking_rate =", initial_masking_rate)
        
        print("maximum_masking_rate =", maximum_masking_rate)

        my_cost, my_rloss, my_sloss, my_acc, my_racc, my_cacc, my_micro, my_macro, my_weighted = run_autoencoder(lr, training_epochs, l1, l2, act, beta, rho, keep_rate, data_obs)

        tmpResult = [my_cost, my_rloss, my_sloss, my_acc, my_racc, my_cacc, my_micro, my_macro, my_weighted]

        tmp_list.extend(tmpResult)
        
        return tmp_list



if __name__ == "__main__":
    result = main()
    print("LABELS [L1, L2, BETA, RHO, ACT, LR, gamma, optimizer, loss_type, rsloss, rloss, sloss, acc, ac_r, ac_c, F1_micro, F1_macro, F1_weighted]") 
    if(len(sys.argv)==6):
        i = 0
        while(i < len(result)):
            print("RESULT ", result[i])
            i += 1
    else:
        print("RESULT ", result)

