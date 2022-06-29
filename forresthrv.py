#!/usr/bin/env python
# coding: utf-8

# 
# <h2>Music Listening Physiological Data (Cardiac and Respiratory Traces)</h2>
# <h2>OpenFMRI Study Forrest dataset, task001 Forrest Gump Audio Movie in German (8 Segments)</h2>
# <h2>Michael A. Casey - Dartmouth College, Data and Code: 2013 - 2019</h2>
# <h2>The Physiological Data:</h2>
# 
# This dataset contains files for 20 subjects, 8 runs per subject. This folder contains only the physiological data.
# 
# Location sub[ID]/physio/task001_run00[1-8]/physio.txt.gz 
#     
# Physiological data were:
# 
# - truncated to start with the first MRI trigger pulse and to end one volume acquisition duration after the last trig-ger pulse. Data are provided in a four-column (MRI trigger, respira-tory trace, cardiac trace and oxygen saturation), space-delimited text file for each run. A log file of the automated conversion procedure is provided in the same directory (conversion.log). Sampling rate for the majority of all participants is 200 Hz.
# 
# - recorded simultaneously with the fMRI data acquisition using a custom setup25 and in-house recording software (written in Python). A Nonin 8600 FO pulse oxymeter (Nonin Medical, Inc, Plymouth, MN, USA) was used to measure cardiac trace and oxygen saturation, and a Siemens respiratory belt connected to a pressure sensor (Honeywell 40PC001B1A) captured the respiratory trace. Analog signals were digitized (12 bit analog digital converter, National Instruments USB-6008) at a sampling rate of 200 Hz. The digitizer also logged the volume acquisition trigger pulse emitted by the MRI scanner.
# 
# - down-sampled to <b>100 Hz</b> and truncated to start with the first MRI trigger pulse and to end one volume acquisition duration after the last trigger pulse. Data are provided in a four-column (MRI trigger, respiratory trace, cardiac trace and oxygen saturation), space-delimited text file for each movie segment. A log file of the automated conversion procedure is provided in the same directory (conversion.log).
# 
# <h2>References:</h2>
# 
# Michael Casey, Music of the 7Ts: Predicting and Decoding Multivoxel fMRI Responses with Acoustic, Schematic, and Categorical Music Features, <i>Frontiers in Psychology: Cognition</i>, 2017
# 
# Michael Hanke, Richard Dinga, Christian Häusler, J. Swaroop Guntupalli, Michael Casey, Falko R. Kaule, Jörg Stadler, High-resolution 7-Tesla fMRI data on the perception of musical genres, <i>F1000 Research</i>, 2015.
# 
# Michael Hanke, Florian J. Baumgartner, Pierre Ibe, Falko R. Kaule, Stefan Pollmann, Oliver Speck, Wolf Zinke & Jörg Stadler, A high-resolution 7-Tesla fMRI dataset from complex natural stimulation with an audio movie, <i>Scientific Data</i>, Volume 1, Article number: 140003 (2014) <br />
# 
# 
# 
# <h2>Things to do:</h2>
# <ul> 
#     <li> import emotion annotations
#     </ul>
#     

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import biosppy
import pyhrv
import pyhrv.time_domain as td
from scipy.signal import resample_poly
from opensignalsreader import OpenSignalsReader
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel, ttest_ind
from scipy.signal import find_peaks
from scipy.fftpack import fftshift
import glob
import pickle
import datetime

n_subjects=20
n_runs=8
data=None # global physiological data

def setup_environment():
    plt.interactive(True)
    plt.matplotlib.rcParams['figure.figsize']=(12,6)
    # Check that physiological data is available
    flist = sorted(glob.glob('sub0??/physio/task001_run00?/physio.txt.gz'))
    if not len(flist):
        print "WARNING: NO PHYSIOLOGY FILES FOUND."
        print "DID YOU SEPARATE THE NOTEBOOK FROM THE DATA DIRECTORY?"
    else:
        print len(flist), "PHYSIOLOGICAL DATA FILES FOUND"
    return flist

flist = setup_environment()

def get_subject_run(subj=1, run=1, t_range=[0,60], plotting=False, zscore_data=False, cardiac=True, respiratory=True, sr=100):
    # subj - 1..20, [1]  (converts 1-based to 0-based indexing)
    # run  - 1..8 [1]  (converts 1-based to 0-based indexing)
    # t_range - time-range: lower, upper in seconds [0,60]
    # plotting - plot the data [True]
    # zscore_data - zscore the data columns [True]
    # cardiac - plot the cardiac trace [True]
    # repiratory - plot the respiratory trace [True]
    #
    # return data for subject, run, and time-range
    plt.rcParams["figure.figsize"] = [12, 6]
    data = np.loadtxt(flist[(subj-1)*n_runs+(run-1)])
    if zscore_data:
        data = data - data.mean(0)
        data[:,2:] = data[:,2:] / data[:,2:].std(0) 
    if t_range is not None:
        data = data[t_range[0]*sr:t_range[1]*sr]
    else:
        t_range=[0, len(data)/sr]
    if plotting:
        plt.figure()
        if cardiac and respiratory:
            plt.plot(data[:,np.array([3,2])]) # Respiratory and Cardiac data
            plt.title('Subject %d Run %d %3.2fs-%3.2fs Cardiac and Respiratory Traces'%(subj,run,t_range[0],t_range[1]), fontsize=20)
            plt.legend(['Respiratory','Cardiac'],fontsize=20)
        elif cardiac:
            plt.plot(data[:,2]) # Cardiac (ECG) data
            plt.title('Subject %d Run %d %3.2fs-%3.2fs Cardiac Trace'%(subj,run,t_range[0],t_range[1]), fontsize=20)
        elif respiratory:
            plt.plot(data[:,3]) # Respiratory data
            plt.title('Subject %d Run %d %3.2fs-%3.2fs Respiratory Trace'%(subj,run,t_range[0],t_range[1]), fontsize=20)
        if(len(data)/sr) < 50: # heuristic for number of x-axis ticks
            plt.xticks(np.arange(0,len(data)+1,sr),t_range[0]+np.arange(0,(len(data)+1)/sr)) # label the x-axis correctly
        else:
            sf=10 # scale factor
            plt.xticks(np.arange(0,len(data),sr*sf),t_range[0]+np.arange(0,(len(data)+sf)/(sr*sf))*sf) # label the x-axis correctly
        plt.xlabel('Time (s)', fontsize=20)
        plt.ylabel('Volts (mV)',fontsize=20)
        plt.axis('tight')
        plt.grid() 
    return data
    

def data_test_plot(subj=11, run=3, t_range=[60,90], **kwargs):
    data=get_subject_run(subj,run,t_range,**kwargs)
    return data

# DATA PREPARATION
def load_all_subj_data(cache=True):
    global data
    data=[]
    if cache:
        print "Loading studyforrest task001 CACHED physiological data..."        
        try:
            with open("./task001rawdata.pickle","rb") as f:
                data = pickle.load(f)
        except:
            cache=False
    if not cache: # catch failed attampt to open cache file
        print "Loading studyforrest task001 ORIG physiological data..."
        for subj in range(1,21):
            data.append([])
            print "subj%03d:"%subj,
            for run in range(1,9):
                data[-1].append(get_subject_run(subj,run,None))
                print "%6d"%len(data[-1][-1]),
            print
        with open("./task001rawdata.pickle","wb") as f:
            pickle.dump(data,f)
    return data

def calc_min_run_lens(data):
    # to compensate for varying-length run data, due to differing acquisition end times,
    # truncate all subjects' run r data to length of min subject run r
    min_run_lens=[]
    for run in range(0,n_runs):
        min_run_lens.append(len(data[0][run])) # default to subject 1
        for subj in range(0,n_subjects):
            if len(data[subj][run]<min_run_lens[-1]):
                min_run_lens[-1]=len(data[subj][run])
        #print "len run", run+1, "=", min_run_lens[run]
    return min_run_lens

def zscore(sig):
    newsig=sig.copy()
    newsig -= newsig.mean()
    newsig /= newsig.std()
    return newsig

def get_hr_acorr(sig, sr=100, lo_hr=40, hi_hr=160, num_peaks=5):
    x = zscore(sig)
    a = fftshift(np.correlate(x,x,"same"))[:len(x)/2]
    peaks, _ = find_peaks(np.array(a),height=0)
    peaks = np.r_[0,peaks][:num_peaks]
    d = np.diff(peaks)
    test_hr = (60.0*sr)/d
    idx = np.where((test_hr>lo_hr) & (test_hr<hi_hr))[0] # constrain to BPM range lo...hi
    rpeaks = peaks[np.r_[0,idx+1]]
    dd = np.diff(rpeaks)
    hr = (60.0*sr)/dd
    if np.any(np.isnan(hr)):
        raise ValueError("hr is nan")
    return {'hr_mean': hr.mean(), 'hr_var': hr.std()}
            
def get_hr_pyhrv(sig,sr=100,show=False):
    # Get R-peaks series using biosppy
    # Calculate hrv using pyhrv gamboa_segmenter
    sig = zscore(sig)
    ecg_filt = biosppy.signals.ecg.ecg(sig, sampling_rate=sr, show=False)[1]
    rpeaks = biosppy.signals.ecg.gamboa_segmenter(ecg_filt, sampling_rate=sr)[0]
    hr_params = td.hr_parameters(rpeaks=rpeaks)
    if show:
        plt.plot(ecg_filt)
        print rpeaks
        n=len(rpeaks)
        plt.plot([rpeaks,rpeaks],[np.ones(n)*ecg_filt.min(),np.ones(n)*ecg_filt.max()])
        plt.title('R Peaks')
    return hr_params

def concatenate_subj_hr_segments(subj, data, sr=100, HR_TYPE='hr_mean', ECG_CHANNEL=2):
    # Given subject, get hr_mean or hr_std series for all runs
    #
    # inputs:
    #     sr  - physiological data sample rate
    global hr_len, hr_step
    min_run_lens = np.array(calc_min_run_lens(data))
    sig = np.zeros(min_run_lens.sum())
    t=0
    # concatenate physiological signal for all runs
    hr_dat=[]
    hr_err=0
    for run in range(n_runs):
        d=data[subj][run][:min_run_lens[run], ECG_CHANNEL] # ECG data
        sig[t:t+len(d)] = d[:]
        t+=len(d)
    for t in range(0, len(sig)-hr_len*sr, hr_step*sr):
        hr_dat.append(get_hr_acorr(sig[t:t+hr_len*sr])[HR_TYPE])
    return hr_dat, hr_err

# HR EXTRACTION
# Extract HR(V) data for all subjects, 1min segments, 2 second skip
def extract_hrv_all(cache=True, hr_win=20, hr_hop=2, show_errs=False):
    global hr_data, hr_error, hr_len, hr_step, data
    hr_len, hr_step = hr_win, hr_hop
    print "hr_len:", hr_len, "hr_step:", hr_step
    dlist=sorted(glob.glob('hr_data_%d_%d*.pickle'%(hr_len,hr_step)))
    if not cache or not len(dlist):
        if data is None:
            load_all_subj_data() # only need to load this once
        hr_data=[]
        hr_error=[]
        for subj in range(n_subjects):
            print "subj%03d"%subj,
            hr_dat, hr_err = concatenate_subj_hr_segments(subj, data)
            hr_data.append(hr_dat)
            hr_error.append(hr_err)

        # CACHE THE DATA, TO MAKE IT LOADABLE
        today = datetime.date.today()
        pickle.dump((hr_data, hr_error), open('hr_data_%d_%d_%d-%d-%d.pickle'%(hr_len,hr_step,today.year,today.month,today.day),'wb'))
    else:
        print "Loading cache file:", dlist[-1]
        hr_data, hr_error = pickle.load(open(dlist[-1],'rb')) # load the latest version of cached data
    hr_error = np.array(hr_error)
    if show_errs:
        # Show missing data errors
        plt.bar(np.arange(1,len(hr_error)+1), hr_error)
        plt.title('Missing data points', fontsize=20)
        plt.xticks(np.arange(1,len(hr_error)+1),fontsize=20)
        plt.xlabel('Subject id',fontsize=20)
        plt.ylabel('#Missing (nan)', fontsize=20)
        plt.axis('tight')
        plt.grid()
    return hr_data, hr_error

# EVALUATION BY MOVIE SEGMENT RETRIEVAL BY HR VARIATION
def corr2_coeff(A,B):
    # Fast correlation matrix computation:
    # subtract rowwise mean of input arrays from arrays
    # dot product and norm
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);
    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def get_corr_matrices(subj, sr=100, clean=False):
    # Compute Segment-to-Segment Cross-Correlation Similarity Matrices to All Subjects
    # input:
    #   subj - subject id (0..n_subjects-1)
    #   sr - physiological data sample rate [100]
    #   clean - remove subjects with too many errors [False]
    # globals:
    #   match_len  - match window len in seconds [match_len]
    #   match_step - match window shift in seconds [match_step]
    #   hr_data    - extracted hr_data
    #   hr_error   - missing hr_data counts
    others = np.setdiff1d(np.arange(n_subjects),subj) # all the other subjects
    n_segs = len(hr_data[subj])-match_len/match_step
    outliers = np.where(hr_error>999)[0] # if clean, remove subjects with too many errors
    if clean and subj in outliers:
        print "WARNING: subj%03d is an outlier, missing data errors>999"%subj
    scores = []
    # Construct similarity matrices between subj and all others
    tgt=np.zeros((n_segs,match_len/match_step))
    print "subj%03d:"%subj,
    for k in np.arange(n_segs):
        tgt[k,:] = np.array(hr_data[subj][k:k+match_len/match_step])
    for other in others: # remove target subj, CHANGED FROM INCORRECT arange(n_subjects):
        if not clean or other not in outliers:
            print other,
            src=np.zeros((n_segs,match_len/match_step))
            for k in np.arange(n_segs):
                src[k,:] = np.array(hr_data[other][k:k+match_len/match_step])
            scores.append(corr2_coeff(tgt,src))    
        else:
             print "(%d)"%other,
    # Quick and dirty nan fix due to degenrate segments with std()==0 in corr(a,b)
    for subj in np.arange(len(scores)):
        if np.any(np.isnan(scores[subj])):
            print "!", # indicates that one or both of hr_data win had zero variance
            scores[subj][np.isnan(scores[subj])]=0
    print
    return np.array(scores)

def retrieve_segments_precision_null(scores):
    # For each segment, return the position of the segment as a sorted list
    N=scores[0].shape[0]
    precision=np.zeros(N)
    null=np.zeros(N)
    SS=scores.mean(0) # mean correlation score for other (not target) subjects
    for k in np.arange(N):
        precision[k]=1.0/(np.where(np.argsort(SS[k,:])[::-1]==k)[0]+1)
        null[k]=1.0/(np.where(np.random.permutation(np.arange(N))==k)[0]+1)
    return precision, null

def prec_rec_subj(target_subj, test_fun=ttest_rel, show=False, clean=False):
    # Information retrieval and evaluation by precision and recall
    # mean between-subject segment cross-correlations, with random-permutation null
    # inpunts:
    #   target_subj - zero-based subject index (0..19)
    #   test_fun - which statistical test to use [ttest_rel]
    #   show - whether to plot results graphs [False]
    #   clean - remove subjects with too many errors [False]
    scores=get_corr_matrices(target_subj, clean=clean) # Get the segment correlation matrices for a subject
    precision, null = retrieve_segments_precision_null(scores) # Calculate precision-recall and null
    tt = test_fun(np.sort(precision)[::-1], np.sort(null)[::-1]) 
    if(show):
        # Show the mean segment correlation matrix
        plt.figure()
        plt.imshow(scores.mean(0), cmap=plt.cm.jet)
        plt.colorbar()
        plt.title('Mean XCorr Matrix, Subj%03d to All Subjects'%target_subj, fontsize=20)
        plt.xlabel('segment',fontsize=16)
        plt.ylabel('segment',fontsize=16)
        # log precision of true and null models
        plt.figure()
        N=scores[0].shape[0]
        eps=0.0000001
        f = lambda x: np.log10(x+eps)
        plt.plot(f(np.sort(precision)[::-1]))
        plt.plot(f(np.sort(null)[::-1]+eps))
        plt.plot([0,len(precision)],[f(2.0/N),f(2.0/N)])
        plt.legend(['prec','null','baseline'],fontsize=16)
        plt.title('subj%03d log precision, recall=1.0, T=%1.3f, p=%1.3e'%(target_subj,tt[0],tt[1]), fontsize=20)
        plt.axis('tight')
        plt.xlabel('Sorted retrieved segments',fontsize=20)
        plt.ylabel('Log10(precision)',fontsize=20)
        plt.grid()
        # Histogram of true and null distributions
        plt.figure()
        plt.hist([f(precision),f(null)],30)
        plt.grid()
        plt.title('hist subj%03d log precision, recall=1.0, T=%1.3f, p=%1.3e'%(target_subj,tt[0],tt[1]), fontsize=20)
        plt.xlabel('Log10(precision)',fontsize=20)
        plt.ylabel('Frequency (#segments)',fontsize=20)
        plt.legend(['prec','null'])
    return tt, precision, null

def prec_null_all(test_fun=ttest_rel):
    tt_res = []
    prec=[]
    null=[]
    for subj in np.arange(n_subjects):
        r, p, n = prec_rec_subj(target_subj=subj, test_fun=test_fun, clean=True)
        tt_res.append(r)
        prec.append(p)
        null.append(n)
    tt_res=np.array(tt_res)
    prec=np.array(prec)
    null=np.array(null)
    return prec, null, tt_res

def evaluate_prec_all(test_fun=ttest_rel, match_win=60, match_hop=2, show=True, log_precision=True, limit=None):
    global match_len, match_step
    match_len, match_step =  match_win, match_hop
    print "match_len:", match_len, "match_step:", match_step
    prec, null, tt_res = prec_null_all(test_fun)
    prec2=prec #[np.where(hr_error<1000)[0]]
    null2=null #[np.where(hr_error<1000)[0]]
    n,N=prec2.shape # num subjects, num segments
    pm=prec2.mean(0)
    nm=null2.mean(0)
    rankm = (1/prec2).mean(0)
    nullm = (1/null2).mean(0)
    N = limit if limit is not None else N    
    limit = N if limit is None else limit
    avg_tt=tt_res.mean(0) #[np.where(hr_error<1000)[0]].mean(0)
    baseline = np.array([(1.0/(np.random.randint(N)+1)) for _ in range(1000)]).mean() # sample mean
    print "avg prec=%6.6f"%prec2.mean()
    print "avg null=%6.6f"%null2.mean()
    print "baseline=%6.6f"%baseline
    print
    print "avg rank=%f"%(1./prec2).mean()
    print "avg null=%f"%(1./null2).mean()
    print "baseline=%f"%(N/2.)
    f = lambda x: np.log10(x) if log_precision else lambda x: x
    if(show): # avg prec plot
        plt.figure()
        idx = np.argsort(pm)[::-1][:limit]
        plt.plot(np.arange(N), f(pm[idx]), linewidth=4)
        ye=prec2[:,idx].std(0)/np.sqrt(n)
        plt.fill_between(np.arange(N),f(pm[idx]-ye), f(pm[idx]+ye),alpha=0.5)
        idx = np.argsort(nm)[::-1][:limit]
        plt.plot(np.arange(N), f(nm[idx]), linewidth=4)
        ye=null2[:,idx].std(0)/np.sqrt(n)
        plt.fill_between(np.arange(N),f(nm[idx]-ye), f(nm[idx]+ye),alpha=0.5)
        plt.plot([0,N],[f(baseline),f(baseline)],'g--',linewidth=2)
        plt.axis('tight')
        plt.grid()
        plt.legend(['HRV','Null','Baseline'],fontsize=16)
        plt.title('Mean precision (hw=%d, mw=%d), recall=1.0, T=%1.3f, p=%1.3e'%(hr_len, match_len, avg_tt[0],avg_tt[1]), fontsize=20)
        plt.xlabel('Sorted retrieved segments',fontsize=20)
        plt.ylabel('Log10 Precision',fontsize=20)
        plt.show()
        # avg rank plot of tru and null distributions
        plt.figure()
        idx = np.argsort(rankm)[:limit]
        plt.plot(np.arange(N), rankm[idx], linewidth=4)
        ye=1./prec2[:,idx].std(0)/np.sqrt(n)
        plt.fill_between(np.arange(N),rankm[idx]-ye, rankm[idx]+ye,alpha=0.5)
        idx = np.argsort(nullm)[:limit]
        plt.plot(np.arange(N), nullm[idx],linewidth=4) 
        ye=1./null2[:,idx].std(0)/np.sqrt(n)
        plt.fill_between(np.arange(N),nullm[idx]-ye,nullm[idx]+ye,alpha=0.5)
        plt.plot([0,N],[N/2.0,N/2.0],'g--',linewidth=2)
        plt.axis('tight')
        plt.grid()
        plt.legend(['HRV','Null','Baseline'],fontsize=16)
        plt.title('Retrieval rank (hw=%d, mw=%d), recall=1.0, T=%1.3f, p=%1.3e'%(hr_len, match_len, avg_tt[0],avg_tt[1]), fontsize=20)
        #axis('tight')
        plt.xlabel('Sorted retrieved segments',fontsize=20)
        plt.ylabel('Segment rank',fontsize=20)
        plt.show()
        # hist avg precision of true and null distributions
        plt.figure()
        axes = plt.gca()        
        plt.hist([f(pm[:limit]),f(nm[:limit])],30)
        plt.grid()
        plt.title('Precision hist (hw=%d, mw=%d), recall=1.0, T=%1.3f, p=%1.3e'%(hr_len, match_len, avg_tt[0],avg_tt[1]), fontsize=20)
        plt.xlabel('Log10(precision)',fontsize=20)
        plt.ylabel('Frequency (#segments)',fontsize=20)
        plt.legend(['HRV','Null'],fontsize=16)
        plt.plot([f(baseline),f(baseline)],[0,axes.get_ylim()[1]],'g--',linewidth=2)        
        plt.show()
    return prec, null, tt_res


if __name__=="__main__":
    extract_hrv_all(cache=True) # Global data needed for analysis
    prec, null, tt_res=evaluate_prec_all(test_fun=wilcoxon, limit=100)
    print "All T staistics:"
    print tt_res
    print "Removing outliers:"
    print "Valid T statistics:"
    print tt_res
    print "Mean T staistic:"
    print tt_res.mean(0)    
