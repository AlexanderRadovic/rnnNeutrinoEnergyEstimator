# RNN Neutrino Energy Estimator
The repository contains a number of python scripts designed to help develop an RNN based approach to estimating
neutrino interaction energys. The core concept is to iterate through all reconstructed objects associate with a
particular neutrino interaction, and to then produce and estimate of the parent neutrinos energy.

As the physics goal of an oscillation measurement rely on measuring the changes to the underlying true energy
distribution we would like to avoid bias as a function of true energy. We can avoid this by reweighting our loss
calculation to simulate training on a flat neutrino energy flux, a solution which is developed here.

## Validation, plotting scripts:
- **validation_plots** Make plots to assess network performance, comparing to other leading approaches.
- **spectra_plot** Make a plot of the true energy spectra, creates weights so we can remove bias as a function of true energy.
- **plot_functions** Contains important plotmaking functions from the previous two modules.

## Training Scripts
- **nova_rnn_training.py** trains the actual network, take the sample and whether or not to reweight the loss as arguments
- **new_loss_functions.py** module to store prototypes of alternative loss functions

## TODO:
- Try adding attention to the network
- Try more event level variables, such as the output of our event ID CNN
- Try a more complex network after concatenation.
