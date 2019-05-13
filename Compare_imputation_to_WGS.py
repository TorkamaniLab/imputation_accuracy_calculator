import numpy as np
import tensorflow as tf
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

import itertools
import seaborn as sns
import pandas as pd
from scipy.stats import linregress
from scipy import stats
# sorting results
from collections import defaultdict
from operator import itemgetter
import timeit #measure runtime
import random #masking
import multiprocessing
import operator
from functools import partial
import allel
from scipy.spatial.distance import squareform
import sys

import subprocess as sp #run bash commands that are much faster than in python (i.e cut, grep, awk, etc)
import statistics

from scipy.stats import linregress
#linregress(a, b)[2] gives the correlation, correcting for nans due to monomorphic predictions
#linregress(a, b)[3] gives the p-value correl

# In[14]:

do_parallel=True
do_parallel_MAF=True

MAF_all_var_vector = []
MAF_all_var = []

mapped_indexes = []

known_indexes = []

ncores = multiprocessing.cpu_count() #for parallel processing
#ncores=8

input_MAF = False #calculate input MAF, if False, calculate REF MAF only

def convert_genotypes_to_int(indexes, posfile, file, categorical="False", alt_signal_only="False"):
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


    refpos = pd.read_csv(posfile, sep='\t', comment='#',header=None)
    refpos = pd.Series(refpos[1], index=range(len(refpos[1])))

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


    inpos_result = sp.check_output("cut -f 2 "+file, encoding='UTF-8', shell=True)


    first=True
    i=0
    inpos_df = []
    for ln in inpos_result.split('\n'):
        i+=1
        if(not ln.startswith("#")):
            if(first==False and ln):
                tmp = ln.split('\t')
                #print(i, ": ", tmp, ": ", ln)
                inpos_df.append(tmp)
            else:
                first=False

    inpos_df = list(map(list, zip(*inpos_df)))

    #print("INPOS DF",inpos_df[0])

    inpos_df = map(int, inpos_df[0])

    inpos = pd.Series(range(len(df[1])), index=inpos_df)

    #print(inpos)
    #print(refpos)
    
    #print("BATCH SHAPE: ", len(df), len(df[0]))
    #print(df[0])
    new_df = 0
    if(categorical=="False"):
        new_df = np.zeros((len(df),len(refpos),2))
        #new_df = np.zeros((df.shape[1]-9,len(refpos)*2))
    else:
        new_df = np.zeros((len(df),len(refpos)))
    #print(type(df))
    i = 0 #RR column index
    j = 0 #RR row index
    idx = 0
    my_hom = 2

    #my_hom = 1

    print(len(df))
    print(len(df[0]))
    myidx = 0

    #print("KEYSSSSSSSSSSSS",inpos.keys())

    while i < len(df): #sample index, total-9 (9 for first comment columns)

        #if(j==(len(df)-1)):
        #    print(j)
        
        j = 0 #variant index, while variant index less than total variant count
        while j < len(refpos): #"|" is present when phased data is proved, "/" is usually unphased
            #print(j, refpos[j])
            #print(inpos)
            #if(refpos[j]==22067276):
            #    print("found int")
            #if(refpos[j]=="22067276"):
            #    print("found str")
            
            if(refpos[j] in inpos.keys()):
                myidx = inpos[refpos[j]]
                #print(myidx)
                df[i][myidx] = str(df[i][myidx])
                #print(df[i][myidx])

                if(df[i][myidx].startswith('1|1') or df[i][myidx].startswith('1/1')):
                    if(categorical=="True" or alt_signal_only==True):
                        new_df[idx][j] = 2
                    else:
                        new_df[idx][j][0] = 0
                        new_df[idx][j][1] = my_hom
                elif(df[i][myidx].startswith('1|0') or df[i][myidx].startswith('0|1') or df[i][myidx].startswith('1/0') or df[i][myidx].startswith('0/1')):
                    if(categorical=="True" or alt_signal_only==True):
                        new_df[idx][j] = 1
                    else:
                        new_df[idx][j][0] = 1
                        new_df[idx][j][1] = 1
                elif(df[i][myidx].startswith('0|0') or df[i][myidx].startswith('0/0')):
                    if(categorical=="True"  or alt_signal_only==True):
                        new_df[idx][j] = 0
                    else:
                        new_df[idx][j][0] = my_hom
                        new_df[idx][j][1] = 0
                else:
                    if(categorical=="True"):
                        new_df[idx][j] = -1
                    elif(alt_signal_only==True):
                        new_df[idx][j] = 0
                    else:
                        new_df[idx][j][0] = 0
                        new_df[idx][j][1] = 0
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

def process_data(posfile, file, categorical="False"):

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

    #refpos = pd.Series(refpos[1], index=range(len(refpos[1])))

    if(do_parallel==False):
        results = convert_genotypes_to_int(indexes, file, categorical)
        print( len(results), len(results[0]), len(results[0][0]))

    else:
        chunks = chunk(indexes, ncores )

        pool = multiprocessing.Pool(ncores)

        results = pool.map(partial(convert_genotypes_to_int, posfile=posfile, file=file, categorical=categorical),chunks)

        pool.close()
        pool.join()

        #print(len(results), len(results[0]), len(results[0][0]) )

        #for i in range(len(results)):
        #    print(len(results[i]))

        #merge outputs from all processes, reshaping nested list
        results = [item for sublist in results for item in sublist]

    #print(len(results), len(results[0]), len(results[0][0]) )

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

    if(do_parallel_MAF == False and input_MAF==True):

         MAF_all_var = calculate_MAF_global_GPU(indexes, results, categorical)

    elif(input_MAF==True):
        chunks = chunk(indexes,ncores)

        pool = multiprocessing.Pool(ncores)

        MAF_all_var = pool.map(partial(calculate_MAF_global, inx=results, categorical=categorical),chunks)

        pool.close()
        pool.join()

        #merge outputs from all processes, reshaping nested list
        MAF_all_var = [item for sublist in MAF_all_var for item in sublist]

    global known_indexes

    if(input_MAF==True):
        global MAF_all_var_vector
        MAF_all_var_vector = []

        for i in range(len(MAF_all_var)):
            MAF_all_var_vector.append(MAF_all_var[i])
            MAF_all_var_vector.append(MAF_all_var[i])
            if(categorical==True):
                MAF_all_var_vector.append(MAF_all_var[i])
        
        print("ALLELE FREQUENCIES", MAF_all_var)
        print("LENGTH1", len(MAF_all_var))

        known_indexes = []
        for j in range(len(MAF_all_var)):
            if(MAF_all_var[j] != 0 ):
                #print("Adding known variant", j)
                known_indexes.append(j)
    else:
        known_indexes = []
        i=0
        while i < len(results[0]):
            if(results[0][i][0] != 0 or results[0][i][1] != 0):
                #print("Adding known variant", i)
                known_indexes.append(i)
            i=i+1

    #print("LKKKKKKKKKKKKKKKKKKKKK", known_indexes)

    stop = timeit.default_timer()
    print('Time to calculate MAF (sec): ', stop - start)

    return results

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


def read_MAF_file(file):

#    CHR         SNP   A1   A2          MAF  NCHROBS
#   9   rs1333039    G    C       0.3821    54330
#   9 rs138885769    T    C    0.0008099    54330
#   9 rs548022918    T    G     0.000589    54330
    pd.set_option('display.float_format', '{:.6f}'.format)

    maftable = pd.read_csv(file, sep='\s+', comment='#')
    maftable['MAF'] = maftable['MAF'].astype(float)

    global MAF_all_var
    
    MAF_all_var = maftable['MAF'].values.tolist()
    print("#####REF MAF#####",MAF_all_var)

#this calculates MAFs for all variants of a data set
def calculate_MAF_global(indexes, inx, categorical="False"):
    j=0
    if(do_parallel_MAF==True):
        getter = operator.itemgetter(indexes)
        x = list(map(list, map(getter, np.copy(inx))))
    else:
        x = inx
    MAF_list = []
    #print("LENGTH", len(x[0]))
    while j < (len(x[0])):
        ref = 0
        alt = 0
        MAF = 0        
        for i in range(len(x)):
            ref+=x[i][j][0]
            alt+=x[i][j][1] 
        if(ref==0):
            MAF=0
        elif(alt<=ref):
            MAF=alt/(ref+alt)
            #major=ref/len(y)
        else:
            MAF=ref/(ref+alt)
        MAF_list.append(MAF)
        j+=1
    return MAF_list


# In[17]:


#flattens a 3D array dataset into a 2D array
def flatten(mydata):
    #subjects, SNP, REF/ALT counts
    if(len(mydata.shape) == 3):
        mydata = np.reshape(mydata, (mydata.shape[0],-1))
    else:#do one hot encoding, depth=3 because missing (-1) is encoded to all zeroes
        mydata = tf.one_hot(indices=mydata, depth=3)
        mydata = tf.layers.flatten(mydata)#flattening to simplify calculations later (matmul, add, etc)
    return mydata


# In[18]:


#filters an imputed dataset (x) and it's respective WGS version (y) based on a upper and lower threshold for the MAFs provided
def filter_by_MAF(x,y, MAFs, threshold1=0, threshold2=1, known_indexes=[], categorical=False):
    
    #print("a")
    colsum=np.sum(y, axis=0)
    indexes_to_keep = []
    i = 0
    j = 0
    k = 0   
    #print("b")
    while i < len(MAFs):
        #print("c")
        #if(i in known_indexes):
        #    print(i, "already in input, excluding it.")
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
    #print("d")
    #print(len(indexes_to_keep))
    getter = operator.itemgetter(indexes_to_keep)
    filtered_data_x = list(map(list, map(getter, np.copy(x))))
    filtered_data_y = list(map(list, map(getter, np.copy(y))))
    
    return filtered_data_x, filtered_data_y

#calculates the correct number of predictions divided by total number of predictions, based on a upper and lower threshold for the MAFs provided
def accuracy_maf_threshold(x, y, MAFs, threshold1=0, threshold2=1, categorical=False):
    

    filtered_data_x, filtered_data_y = filter_by_MAF(x,y, MAFs, threshold1, threshold2, categorical)
    
    correct_prediction = np.equal( np.round( filtered_data_x ), np.round( filtered_data_y ) )
    accuracy_per_marker = np.mean(correct_prediction.astype(float), 0)
    
    accuracy = np.mean(accuracy_per_marker)
    standard_dev = statistics.stdev(accuracy_per_marker)
    standard_err = standard_dev/np.sqrt(len(filtered_data_x[0]))

    return accuracy, accuracy_per_marker, standard_err


def r2_maf_threshold(x, y, MAFs, threshold1=0, threshold2=1, categorical=False):
    

    filtered_data_x, filtered_data_y = filter_by_MAF(x,y, MAFs, threshold1, threshold2, categorical)
    r2 = []

    for i in range(len(filtered_data_x)): 
        r2tmp = linregress(filtered_data_x[i], filtered_data_y[i])[2]
        r2.append(r2tmp)

    r2mean = np.mean(r2)
    standard_dev = statistics.stdev(r2)
    standard_err = standard_dev/np.sqrt(len(r2))

    return r2mean, standard_err, r2


# In[19]:


#split input data into chunks so we can procces data in parallel batches
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


def f1_score(x, y, MAFs, threshold1, threshold2, categorical=False, sess=tf.Session()):


    y_pred, y_true = filter_by_MAF(x,y, MAFs, threshold1, threshold2, categorical)

    f1s = [0, 0, 0]
    vars = [0, 0, 0]
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
        mean, var = tf.nn.moments(f1, axes=axis)

        f1s[i] = tf.reduce_mean(f1)
        vars[i] = var
      

    weights = tf.reduce_sum(y_true, axis=0)
    weights = tf.divide(weights,tf.reduce_sum(weights))

    f1s[2] = tf.reduce_sum(tf.multiply(f1, weights))
    mean, vars[2] = tf.nn.moments(tf.multiply(f1, weights), axes=axis)

    micro, macro, weighted = sess.run(f1s)
    myvars1, myvars2, myvars3 = sess.run(vars)
    myvars1, myvars2, myvars3 = np.sqrt(myvars1), np.sqrt(myvars2), np.sqrt(myvars3)
    
    return micro, macro, weighted, myvars1, myvars2, myvars3


def calculate_ref_MAF(refname):
    #plink --vcf HRC.r1-1.EGA.GRCh37.chr9.haplotypes.9p21.3.vcf.clean4 --freq 

    result = sp.check_output("plink --vcf "+refname+" --freq --out "+refname, encoding='UTF-8', shell=True)

    read_MAF_file(refname+".frq")



# In[20]:

def generate_refpos(refname):

    #result = sp.check_output("cut -f1-5 "+inpos+" > "+refname+".pos", encoding='UTF-8', shell=True)
    #result = sp.check_output("cut -f1-5 "+refname+" > "+refname+".pos", encoding='UTF-8', shell=True)
    #my_cmd = "cut -f1-5 "+inpos+" > "+refname+".pos"
    my_cmd = "cut -f1-5 "+refname
    with open(refname+".pos", "w") as outfile:
        sp.run(my_cmd, stdout=outfile,  encoding='UTF-8', shell=True)

reffile=sys.argv[1]
inpos=sys.argv[1]+".pos"
infile=sys.argv[2]
imputedfile=sys.argv[3]
inwgs=sys.argv[4]

generate_refpos(reffile)

#process input file
input_df = process_data(inpos, infile)
my_known_indexes = known_indexes
print("KNOWN INDEXES:", my_known_indexes)

#Process ARIC WGS dataset as our ground truth, mapping its variants to the positions listed in HRC
new_df_obs = process_data(inpos, inwgs)
my_wgs_indexes = known_indexes


#Process imputed output for ARIC dataset, mapping its variants to the positions listed in HRC
new_df_imputed = process_data(inpos, imputedfile)

#Create a backup copy of our ground truth datasets
orig_new_df_obs = np.copy(new_df_obs)

#read_maf_file(freqfile)
calculate_ref_MAF(reffile)

print("REF MAF:", MAF_all_var)

#convert 3D array to 2D array, necessary for inference and accuracy calculations
new_df_obs = flatten(new_df_obs.copy())

(new_df_obs.shape)


#convert 3D array to 2D array, necessary for inference and accuracy calculations
new_df_imputed = flatten(new_df_imputed.copy())

(new_df_imputed.shape)



start = timeit.default_timer()

print("\n****\n")
print(new_df_obs)
print("\n****\n")
    
print("\n****\n")
print(new_df_imputed)

print("\n****\n")
    

#set parallel MAF calculation
do_parallel_MAF = True
#number of cores
ncores = multiprocessing.cpu_count() #for parallel processing
#set one index per variant

#Map calculated MAF values to the ones present in the ground truth dataset (WGS)
def map_MAFs(MAFs, mapped_indexes):
    i=0
    j=0
    new_MAFs = []
    while i < len(MAFs):
        if(j in mapped_indexes):
            new_MAFs.append(MAFs[i])
        i += 1
        j += 2
    return new_MAFs




print("Calculating accuracy for predictions...")

#-1 MAF means that this variant was part of input data, will be ignored for accuracy calc.
my_MAF_all_var = MAF_all_var
print("###########KNOWN INDEXES:", my_known_indexes)

for i in my_known_indexes:
    my_MAF_all_var[i] = -1

for i in range(len(my_MAF_all_var)):
    if(i not in my_wgs_indexes):
        my_MAF_all_var[i] = -1

accuracy_imputed, am4v, standard_err = accuracy_maf_threshold(new_df_imputed, new_df_obs, my_MAF_all_var, threshold1=0, threshold2=1, categorical=False)
accuracy_imputed_lowMAF, _, standard_err_low = accuracy_maf_threshold(new_df_imputed, new_df_obs, my_MAF_all_var, threshold1=0, threshold2=0.005, categorical=False)
accuracy_imputed_highMAF, _, standard_err_high = accuracy_maf_threshold(new_df_imputed, new_df_obs, my_MAF_all_var, threshold1=0.005, threshold2=1, categorical=False)

r2_imputed, r2_standard_err, r2s = r2_maf_threshold(new_df_imputed, new_df_obs, my_MAF_all_var, threshold1=0, threshold2=1, categorical=False)
r2_imputed_lowMAF, r2_standard_err_low, _ = r2_maf_threshold(new_df_imputed, new_df_obs, my_MAF_all_var, threshold1=0, threshold2=0.005, categorical=False)
r2_imputed_highMAF, r2_standard_err_high, _ = r2_maf_threshold(new_df_imputed, new_df_obs, my_MAF_all_var, threshold1=0.005, threshold2=1, categorical=False)

f1s_full = f1_score(new_df_imputed, new_df_obs, my_MAF_all_var, threshold1=0, threshold2=1, categorical=False)
f1s_lowMAF = f1_score(new_df_imputed, new_df_obs, my_MAF_all_var, threshold1=0, threshold2=0.005, categorical=False)
f1s_highMAF = f1_score(new_df_imputed, new_df_obs, my_MAF_all_var, threshold1=0.005, threshold2=1, categorical=False)



stop = timeit.default_timer()

print('Time to calculate MAF and accuracy (sec): ', stop - start)

print("accuracy per variant")
print(am4v)
print("MAF per variant")
print(my_MAF_all_var)

print("r2 per sample")
print(r2s)

print("LABEL: acc_0-1", "acc_0-0.005", "acc_0.005-1")
print("ACC_RESULT:", accuracy_imputed, accuracy_imputed_lowMAF, accuracy_imputed_highMAF)
print("ACC_RESULT_STDERR:", standard_err, standard_err_low, standard_err_high)

print("LABEL: F1_0-1", "F1_0-0.005", "F1_0.005-1")

print("F1_RESULT:", f1s_full[0], f1s_lowMAF[0], f1s_highMAF[0])
print("F1_RESULT_STDERR:", f1s_full[3], f1s_lowMAF[3], f1s_highMAF[3])

print("F1_RESULT_MICRO:", f1s_full[1], f1s_lowMAF[1], f1s_highMAF[1])
print("F1_RESULT_MICRO_STDERR:", f1s_full[4], f1s_lowMAF[4], f1s_highMAF[4])

print("F1_RESULT_MACRO:", f1s_full[2], f1s_lowMAF[2], f1s_highMAF[2])
print("F1_RESULT_MACRO_STDERR:", f1s_full[5], f1s_lowMAF[5], f1s_highMAF[5])


print("LABEL: r2_0-1", "r2_0-0.005", "r2_0.005-1")
print("R2_RESULT:", r2_imputed, r2_imputed_lowMAF, r2_imputed_highMAF)
print("R2_RESULT_STDERR:", r2_standard_err, r2_standard_err_low, r2_standard_err_high)
