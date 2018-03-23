import matplotlib.pyplot as plt

def plotResidual(rnnEstimator, currentAnalysis Estimator, simpleEstimator, sampleType):
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

def plot2DEnergyResponse(inputArray, estimatorTitle, recoTitle, fileName, isNorm):

    if isNorm:
        inputArray = inputArray.astype('float') / inputArray.sum(axis=0)[np.newaxis,:]

    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_ylabel(recoTitle)
    ax.set_xlabel('True Energy')
    ax.set_title(estimatorTitle)
    plt.imshow(X_hist,cmap='gist_heat_r',interpolation='none',extent=[1,5,-50,50],aspect=0.05,vmin=0)
    plt.savefig(fileName,dpi = 1000)
