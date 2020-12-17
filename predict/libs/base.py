#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 ETRI (Minkyu Lee)

import numpy
from etri_dist.libs import sigproc
from scipy.fftpack import dct
import os.path
def set_cmvn_file(path):
    if os.path.exists(path+'/cmvn.ark'):
        import kaldiio
        import numpy as np
        cmvn = kaldiio.load_mat(path+'/cmvn.ark')
        count = cmvn[0][-1]
        mean =cmvn[0,:-1]/count
        var = (cmvn[1,:-1]/count)-mean*mean
        scale = 1 / np.sqrt(var)
        offset = -(mean*scale)
        norm = np.zeros((2, cmvn[0].shape[0]-1))
        norm[0,:] = offset
        norm[1,:] = scale
        print('cmvn.ark file apllied,inputdim=%d'%(cmvn[0].shape[0]-1))
        return norm,cmvn[0].shape[0]-1
    else: 
        print('Default cmvn apllied')
        norm = [[-3.42167211,-3.19438577,-3.38188171,-3.70518327,-3.95481634,-4.08967972,
-4.12971735,-4.0177989,-4.05439854,-4.11131907,-4.2040782,-4.20991182,
-4.25162649,-4.25907564,-4.2473011,-4.2863965,-4.3228898,-4.34782124,
-4.42950296,-4.39487934,-4.36633348,-4.50143957,-4.48567581,-4.5968647,
-4.61216831,-4.68406868,-4.68915033,-4.70958185,-4.69221592,-4.70501041,
-4.70832491,-4.72276783,-4.74502897,-4.77747059,-4.79214573,-4.81906843,
-4.84250784,-4.8643012,-4.88663578,-4.85466433,-4.90646744,-4.9041872,
-4.9521184,-4.97165966,-5.01090717,-5.0324893,-5.03520489,-5.03818893,
-5.04275227,-5.06600761,-5.08489704,-5.11085701,-5.12284422,-5.12537432,
-5.10954142,-5.08986282,-5.09612083,-5.12694502,-5.16363811,-5.19640732,
-5.22519541,-5.21797276,-5.21604729,-5.2105999,-5.21371508,-5.21609163,
-5.2056222,-5.19626617,-5.16277838,-5.13859081,-5.13667679,-5.15312576,
-5.17222881,-5.1936388,-5.22146034,-5.23832226,-5.24389744,-5.21634912,
-5.15253687,-5.05822802,1.25118387,0.16807194,0.02456923],
[0.3435652,0.30806524,0.2948626,0.29855329,0.29850823,0.29500216,
0.2900461,0.28056651,0.28067291,0.28453702,0.28764045,0.28579083,
0.28413242,0.28140688,0.27958646,0.28081656,0.28304908,0.28531724,
0.28741103,0.28793833,0.28851834,0.293441,0.29677734,0.30205214,
0.30518064,0.30842769,0.31117955,0.31127203,0.31129918,0.31215218,
0.31162351,0.31246269,0.31293857,0.31346714,0.31359836,0.31413645,
0.31463048,0.31555009,0.31622899,0.31533957,0.31715053,0.31806079,
0.31910229,0.31948549,0.31972486,0.3182689,0.31538239,0.31367698,
0.31298089,0.31383485,0.31637794,0.31893483,0.320057,0.31951809,
0.31782046,0.31567478,0.31514621,0.31691712,0.3202112,0.32393128,
0.32680854,0.32837763,0.33002022,0.33165351,0.33369759,0.33539012,
0.33612099,0.3356232,0.33299479,0.33120826,0.3311016,0.33190542,
0.33274376,0.33311793,0.33442715,0.33595425,0.33788115,0.34010333,
0.3433814,0.34954873,2.91277742,2.19889498,4.09453058]]

        return norm,83



def cmvn(vec, variance_normalization=False):
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        variance_normalization (bool): If the variance
            normilization should be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    eps = 2**-30
    rows, cols = vec.shape

    # Mean calculation
    #norm = numpy.mean(vec, axis=0)
    norm=[13.81728912,13.54220955,14.5613793,15.45506153,16.28197078,16.77583828,16.90248914,16.51130705,16.87707883,17.03003926,16.79243714,16.4319049,16.15078832,15.96410727,15.86211735,15.88430905,15.91035622,15.74871705,15.63217505,15.18196422,14.87927356,14.97845328,14.62023821,14.54376859,14.36037709,14.37890261,14.05186802,13.95491892,13.78801275,13.7417198,13.70090885,13.63907513,13.5986479,13.55647996,13.57488933,13.62006698,13.72976808,13.72190318,13.70704903,13.61857512,13.68904373,13.65855143,13.75306085,13.70118232,13.68455553,13.64148073,13.56307018,13.55783733,13.44710216,13.30385999,13.23176361,13.24240552,13.24202188,13.22154549,13.1852984,13.2220598,13.33818141,13.46509443,13.44225796,13.33508423,13.23343752,13.02002618,12.86639199,12.83257406,12.92551667,12.9394715,12.87757082,12.89940534,12.94605788,12.93834487,12.83259154,12.71292629,12.62831123,12.61561601,12.54721791,12.15011781,11.30001299,9.98615348,8.61970199,7.56689922]
    norm_vec = numpy.tile(norm, (rows, 1))

    # Mean subtraction
    mean_subtracted = vec - norm_vec

    # Variance normalization
    if variance_normalization:
        #stdev = numpy.std(mean_subtracted, axis=0)
        stdev = [2.77170399,2.36850564,2.64998414,2.80786705,2.96376364,3.16694759,3.38130528,3.90046123,3.89960683,3.75648588,3.80324647,3.81267306,3.85492083,3.96901255,4.10301255,4.19317926,4.14094211,4.11957733,4.18141569,4.19893117,4.10962309,3.96179855,3.79471732,3.68831649,3.53129423,3.3899461,3.42984116,3.46188679,3.45592937,3.38961382,3.36126416,3.34760663,3.36526951,3.42112789,3.44533324,3.45941405,3.45994202,3.57684512,3.64303491,3.6141617,3.65694041,3.67959744,3.65586664,3.65669462,3.66385247,3.64272663,3.58584162,3.5918153,3.5033288,3.35176385,3.29142179,3.33101401,3.3453287,3.33631761,3.34699373,3.37557506,3.48191781,3.5997266,3.59247739,3.52937215,3.4241821,3.28394983,3.14243689,3.12374424,3.25172066,3.29622535,3.26740819,3.31936797,3.41201299,3.46992348,3.40404082,3.21726981,3.17939876,3.35834759,3.48727169,3.50188143,3.41396138,3.20734311,2.97026716,3.08520177]
        stdev_vec = numpy.tile(stdev, (rows, 1))
        output = mean_subtracted / (stdev_vec + eps)
    else:
        output = mean_subtracted

    return output
def cmvn2(vec,in_norm=None, variance_normalization=False,dim=80):
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        variance_normalization (bool): If the variance
            normilization should be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    rows,cols = vec.shape
    if in_norm is None:
        norm = [[-3.42167211,-3.19438577,-3.38188171,-3.70518327,-3.95481634,-4.08967972,
-4.12971735,-4.0177989,-4.05439854,-4.11131907,-4.2040782,-4.20991182,
-4.25162649,-4.25907564,-4.2473011,-4.2863965,-4.3228898,-4.34782124,
-4.42950296,-4.39487934,-4.36633348,-4.50143957,-4.48567581,-4.5968647,
-4.61216831,-4.68406868,-4.68915033,-4.70958185,-4.69221592,-4.70501041,
-4.70832491,-4.72276783,-4.74502897,-4.77747059,-4.79214573,-4.81906843,
-4.84250784,-4.8643012,-4.88663578,-4.85466433,-4.90646744,-4.9041872,
-4.9521184,-4.97165966,-5.01090717,-5.0324893,-5.03520489,-5.03818893,
-5.04275227,-5.06600761,-5.08489704,-5.11085701,-5.12284422,-5.12537432,
-5.10954142,-5.08986282,-5.09612083,-5.12694502,-5.16363811,-5.19640732,
-5.22519541,-5.21797276,-5.21604729,-5.2105999,-5.21371508,-5.21609163,
-5.2056222,-5.19626617,-5.16277838,-5.13859081,-5.13667679,-5.15312576,
-5.17222881,-5.1936388,-5.22146034,-5.23832226,-5.24389744,-5.21634912,
-5.15253687,-5.05822802,1.25118387,0.16807194,0.02456923],
[0.3435652,0.30806524,0.2948626,0.29855329,0.29850823,0.29500216,
0.2900461,0.28056651,0.28067291,0.28453702,0.28764045,0.28579083,
0.28413242,0.28140688,0.27958646,0.28081656,0.28304908,0.28531724,
0.28741103,0.28793833,0.28851834,0.293441,0.29677734,0.30205214,
0.30518064,0.30842769,0.31117955,0.31127203,0.31129918,0.31215218,
0.31162351,0.31246269,0.31293857,0.31346714,0.31359836,0.31413645,
0.31463048,0.31555009,0.31622899,0.31533957,0.31715053,0.31806079,
0.31910229,0.31948549,0.31972486,0.3182689,0.31538239,0.31367698,
0.31298089,0.31383485,0.31637794,0.31893483,0.320057,0.31951809,
0.31782046,0.31567478,0.31514621,0.31691712,0.3202112,0.32393128,
0.32680854,0.32837763,0.33002022,0.33165351,0.33369759,0.33539012,
0.33612099,0.3356232,0.33299479,0.33120826,0.3311016,0.33190542,
0.33274376,0.33311793,0.33442715,0.33595425,0.33788115,0.34010333,
0.3433814,0.34954873,2.91277742,2.19889498,4.09453058]]
    else:
        norm = in_norm        
    
    norm_vec = numpy.tile(norm[0][:dim],(rows,1))
   
    stdev_vec = numpy.tile(norm[1][:dim],(rows,1))
   
    vec = vec * stdev_vec
    vec += norm_vec

    return vec

def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=23,nfft=512,lowfreq=20,highfreq=None,dither=1.0,remove_dc_offset=True,preemph=0.97,
         ceplifter=22,useEnergy=True,wintype='povey'):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,dither,remove_dc_offset,preemph,wintype)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if useEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=40,nfft=512,lowfreq=0,highfreq=None,dither=1.0,remove_dc_offset=True, preemph=0.97, 
          wintype='hamming'):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
     winfunc=lambda x:numpy.ones((x,))   
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    frames,raw_frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, dither, preemph, remove_dc_offset, wintype)
    pspec = sigproc.powspec(frames,nfft) # nearly the same until this part
    energy = numpy.sum(raw_frames**2,1) # this stores the raw energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy

def logfbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=40,nfft=512,lowfreq=64,highfreq=None,dither=1.0,remove_dc_offset=True,preemph=0.97,wintype='hamming'):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,dither, remove_dc_offset,preemph,wintype)
    return numpy.log(feat)

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 1127 * numpy.log(1+hz/700.0)


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (numpy.exp(mel/1127.0)-1)

def get_filterbanks(nfilt=26,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)

    # check kaldi/src/feat/Mel-computations.h    
    fbank = numpy.zeros([nfilt,nfft//2+1])
    mel_freq_delta = (highmel-lowmel)/(nfilt+1)
    for j in range(0,nfilt):
        leftmel = lowmel+j*mel_freq_delta
        centermel = lowmel+(j+1)*mel_freq_delta
        rightmel = lowmel+(j+2)*mel_freq_delta
        for i in range(0,nfft//2):
            mel=hz2mel(i*samplerate/nfft)
            if mel>leftmel and mel<rightmel:
                if mel<centermel:
                    fbank[j,i]=(mel-leftmel)/(centermel-leftmel)
                else:
                    fbank[j,i]=(rightmel-mel)/(rightmel-centermel)
    return fbank

def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat
