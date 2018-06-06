'''Useful functions for plotting the performance of our regression networks'''

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

def plot_residual(rnn_estimator, curr_estimator, simp_estimator, sample_type):
    '''Plot 1D residual for an RNN estimator and leading alternatives.'''
    bins = np.linspace(-1, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('')
    ax.set_ylabel('Events')
    ax.set_xlabel('(Reco E - True E)/True E')
    plt.hist(simp_estimator, bins, color='b', alpha=0.9, histtype='step', lw=2, label='CalE')
    plt.hist(curr_estimator, bins, color='g', alpha=0.9, histtype='step', lw=2, label='3A')
    plt.hist(rnn_estimator, bins, color='r', alpha=0.9, histtype='step', lw=2, label='LSTM')
    ax.legend(loc='right', frameon=False)
    plt.savefig(sample_type+'Comparison.png', dpi=1000)


def plot_true_spectra(true_spectra, sample_type):
    '''Plot a 1D true spectra and save it's inverse to a numpy array on disk.
    Array saved for reweighting loss calculation in training.'''
    bins = np.linspace(0, 5, 50)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('')
    ax.set_ylabel('Events')
    ax.set_xlabel('True E')
    plt.hist(true_spectra, bins, color='r', alpha=0.9, histtype='step', lw=2, label='TruE')
    plt.savefig(sample_type+'TrueSpec.png', dpi=1000)

    spectra_histo = np.zeros(50)

    for i in true_spectra:
        spectra_histo[int(i*10)] = spectra_histo[int(i*10)]+1

    for i in range(0, 50):
        if spectra_histo[i] > 0:
            spectra_histo[i] = 1./spectra_histo[i]

    np.save(sample_type+'TrueSpecWeight', spectra_histo)

    plt.savefig(sample_type+'TrueSpec.png', dpi=1000)



def plot2D_energy_response(input_array, estimator_title, reco_title, file_name,
                           is_norm, ax0, ax1, ay0, ay1):
    '''Plot2D residual of regression network vs. true variable.'''
    if is_norm:
        input_array = input_array.astype('float') / input_array.sum(axis=0)[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_ylabel(reco_title)
    ax.set_xlabel('True Energy')
    ax.set_title(estimator_title)
    aspect_ratio = float(ax1-ax0)/float(ay1-ay0)

    plt.imshow(input_array, cmap='gist_heat_r', interpolation='none',
               extent=[ax0, ax1, ay0, ay1], aspect=aspect_ratio, vmin=0)
    plt.savefig(file_name, dpi=1000)
