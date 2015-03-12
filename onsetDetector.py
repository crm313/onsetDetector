#!/usr/bin/env python
'''
Chris Miller - 11 March 2015

onsetDetector.py - detects onsets in wav file and writes onset times (in seconds)
                        to a csv called 'calculatedOnsets'

call function from bash shell as:

$ python onsetDetector.py filename.wav

'''

import numpy as np
from scipy.io import wavfile
from scipy import signal
import math
import sys
import csv

def computeNoveltyFunction(filepath,windowSize,hopSize):
   '''
       computeNoveltyFunction: computes log energy derivative novelty function
        
           INPUTS:
               filepath         - filepath to .wav file for analysis
               windowSize       - window size for analysis (samples)
               hopSize          - hop size for analysis (samples)
        
           OUTPUTS:
               LEderivative     - log energy derivative novelty function
               timeNovelty      - time vector for n_t_le (seconds)
               fsNovelty        - sampling frequency of n_t_le (Hz)
    '''

# initialization
   fs, sig = wavfile.read(filepath)
   sig = sig / (2.**15)
   L = len(sig)
   time = np.linspace(1/fs, L/fs, L)
   window = signal.hamming(windowSize)
   noveltyFunctionLength = int(math.floor(L/hopSize))
   localEnergy = np.zeros(noveltyFunctionLength)
   # timeNovelty array is one sample shorter to account for numeric differentiation
   # later in function ----> len(LEderivative) = len(logEnergy) -1
   timeNovelty = np.zeros(noveltyFunctionLength - 1)

   # pad end of signal with zeros
   pad = np.zeros(windowSize)
   sig = np.append(sig,pad)
 
# compute local energy in each window
   m = 1
   # account for drastically smaller hop sizes
   if hopSize < windowSize/2:
        m = windowSize/hopSize
   while m < noveltyFunctionLength:
        windowedSignal = np.zeros(windowSize)
        localEnergySum = 0
        for n in range((-windowSize/2),(windowSize/2 - 1)):
            localEnergySum = localEnergySum + (((sig[n + m*hopSize])**2) * window[n + windowSize/2])
        localEnergy[m] = (1./windowSize)*localEnergySum
        # construct time vector
        if m < noveltyFunctionLength - 1:
            timeNovelty[m] = time[m*hopSize]
        m = m+1

# take log and then differentiate
   logEnergy = np.zeros(len(localEnergy))
   for i in range(1, len(localEnergy)):
      logEnergy[i] = math.log(localEnergy[i] + np.spacing(1))
   LEderivative = np.diff(logEnergy)
   fsNovelty = fs/hopSize
   return LEderivative, timeNovelty, fsNovelty 

################################################

def onsetsFromNovelty(noveltyFunction,noveltyTimeArray,fsNovelty,LPcutoff,medFilterLength,offset):

   '''
       onsetsFromNovelty: retrieve onsets from novelty function through peak picking
        
           INPUTS:
               noveltyFunction          - novelty function
               noveltyTimeArray         - time array for novelty function (seconds)
               fsNovelty                - sampling frequency of novelty function (Hz)
               LPcutoff                 - cutoff frequency for smoothing filter (Hz)
               medFilterLength          - length of median filter for adaptive threshold (samples)
               offset                   - adaptive threshold offset
        
           OUTPUTS:
               onsetAmplitude           - onset amplitude
               onsetTime                - onset time (seconds)
   '''

   L = len(noveltyFunction)
# smooth novelty function with smoothing filter
   wN = LPcutoff/(fsNovelty/2.)
   b, a = signal.butter(1,wN)
   smoothedNovelty = signal.filtfilt(b,a,noveltyFunction)

# normalize
   normFactor = np.amax(smoothedNovelty)
   normNovelty = np.zeros(L)
   for i in range(1, L):
       normNovelty[i] = smoothedNovelty[i]/normFactor

# adaptive thresholing
   threshold = np.zeros(L)
   for m in range(1,L):
      localRange = np.arange((m - medFilterLength/2),(m + medFilterLength/2))
      localValues = np.zeros(len(localRange))
      for k in range(1,len(localRange)):
         if localRange[k] <= 0:
             localValues[k] = 0
         elif localRange[k] > 0:
             localValues[k] = 0
         else:
             localValues[k] = normNovelty[localRange[k]]
      medianValue = np.median(localValues) 
      threshold[m] = offset + np.around(medianValue)

   # threshold novelty function
   reducedNovelty = np.zeros(L)
   for m in range(1,L):
      if normNovelty[m] - threshold[m] > 0:
          reducedNovelty[m] = normNovelty[m]
      else:
          reducedNovelty[m] = 0
   
# peak picking

   dN = np.diff(reducedNovelty)
   peakCount = 0
   location = []
   for i in range(1,len(dN)):
      if i == 1:
          if np.sign(dN[i]) < 0:
              location.append(i)
              peakCount = peakCount + 1
      else:
          if np.sign(dN[i - 1]) - np.sign(dN[i]) > 1:
              location.append(i)
              peakCount = peakCount + 1
   if not location:
      # no onsets detected!
      onsetAmplitude = []
      onsetTime = []
   else:
      onsetAmplitude = np.zeros(len(location))
      onsetTime = np.zeros(len(location))
      for i in range(1,len(location)):
         onsetAmplitude[i] = smoothedNovelty[location[i]]
         onsetTime[i] = timeNovelty[location[i]]
   return onsetAmplitude, onsetTime
          
   
################################################
################################################
################################################
################################################
################################################
################################################

if __name__=='__main__':
    filename = str(sys.argv[1])
    # default onsetDetector parameters
    windowSize = 1024 #samples
    hopSize = 512 #samples
    LPcutoff = 4 #Hz
    medFilterLength = 8 #samples
    offset = 0.01
    LEderivative, timeNovelty, fsNovelty = computeNoveltyFunction(filename,windowSize,hopSize)
    onsetAmplitude, onsetTime = onsetsFromNovelty(LEderivative,timeNovelty,fsNovelty,LPcutoff,medFilterLength,offset)
    outputFilename = "calculatedOnsets"
    with open(outputFilename, "wb") as file:
        writer = csv.writer(file)
        writer.writerow(onsetTime)
        

