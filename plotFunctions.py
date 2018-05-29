from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

def plotResidual(rnnEstimator, currEstimator, simpEstimator, sampleType):
    bins = np.linspace(-1, 1, 100)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title('')
    ax.set_ylabel('Events')
    ax.set_xlabel('(Reco E - True E)/True E')
    plt.hist(simpEstimator, bins, color='b', alpha=0.9, histtype='step',lw=2,label='CalE')
    plt.hist(currEstimator, bins, color='g', alpha=0.9, histtype='step',lw=2,label='3A')
    plt.hist(rnnEstimator, bins, color='r', alpha=0.9, histtype='step',lw=2,label='LSTM')
    ax.legend(loc='right',frameon=False)
    plt.savefig(sampleType+'Comparison.pdf',dpi = 1000)


def plotTrueSpectra(trueSpectra, sampleType):
    bins = np.linspace(0, 20, 200)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title('')
    ax.set_ylabel('Events')
    ax.set_xlabel('True E')
    plt.hist(trueSpectra, bins, color='r', alpha=0.9, histtype='step',lw=2,label='TruE')
    plt.savefig(sampleType+'TrueSpec.pdf',dpi = 1000)

    spectraHisto=np.zeroes(100)

    for i in trueSpectra:
        spectraHisto[int(i*10)]++

    spectraHisto=1./spectraHisto;

    np.save(spectraHisto,sampleType+'TrueSpecWeight.np')


def plot2DEnergyResponse(inputArray, estimatorTitle, recoTitle, fileName, isNorm, ax0, ax1, ay0, ay1):

    if isNorm:
        inputArray = inputArray.astype('float') / inputArray.sum(axis=0)[np.newaxis,:]

    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_ylabel(recoTitle)
    ax.set_xlabel('True Energy')
    ax.set_title(estimatorTitle)
    aspectRatio=float(ax1-ax0)/float(ay1-ay0)
    print (aspectRatio)
    #0.05
    plt.imshow(inputArray,cmap='gist_heat_r',interpolation='none',extent=[ax0,ax1,ay0,ay1],aspect=aspectRatio,vmin=0)
    plt.savefig(fileName,dpi = 1000)
