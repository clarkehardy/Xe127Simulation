import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import histlite as hl
import pandas as pd
import LightMap
import sys
import os
import pickle
import argparse
import time
import gzip
from plot_data import plot_lm_rz,proj2d,make_figs,plot_results

# ***********************************************************************************************************
# SET BASIC RUN OPTIONS HERE
# ***********************************************************************************************************

# save figures here
sim_dir = os.getenv('SIM_DIR')
path = '{}/outputs/'.format(sim_dir)

'''
Arguments:
description    :    The description that will be appended to file names to identify this run
standoff       :    The fiducial cut as a distance from the nearest wall, in mm
events         :    The number of events to be used to reconstruct the lightmap
fit_type       :    The type of fit to apply to the data. Currently 'NN' or 'KS'
-train         :    Run the full NN training. Default is False
-make_plots    :    Show the plots produced rather than just saving them
-both_peaks    :    Train on both Xe-127 peaks. Default is just high energy peak
-learning_rate :    Learning rate input for NN
-layers        :    List of number of nodes per layer. One input for each layer
-batch_size    :    Batch size used in NN training
-ensemble_size :    Number of NNs in the ensemble
-sigma         :    Smoothing length scale for kernel smoother
-seed          :    Random state used to get subset of events from input files
-validation    :    Use extra calibration events to compute validation loss
-input_files   :    List of processed simulation files
'''
parser = argparse.ArgumentParser()
parser.add_argument('-input_files',type=str,nargs='+')
parser.add_argument('-train',action='store_true',default=False)
parser.add_argument('-make_plots',action='store_true',default=False)
parser.add_argument('-both_peaks',action='store_true',default=False)
parser.add_argument('-learning_rate',type=float)
parser.add_argument('-layers',type=int,nargs='+')
parser.add_argument('-batch_size',type=int)
parser.add_argument('-ensemble_size',type=int)
parser.add_argument('-sigma',type=float)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-validation',action='store_true',default=False)
parser.add_argument('description',type=str)
parser.add_argument('standoff',type=float)
parser.add_argument('events',type=int)
parser.add_argument('fit_type',type=str)

args = parser.parse_args()
input_files = args.input_files
train = args.train
make_plots = args.make_plots
both_peaks = args.both_peaks
standoff = args.standoff
name = args.description
events = args.events
fit_type = args.fit_type
learning_rate = args.learning_rate
layers = args.layers
batch_size = args.batch_size
ensemble_size = args.ensemble_size
sigma = args.sigma
seed = args.seed
validation = args.validation

if fit_type!='NN' and fit_type!='KS':
    print('\nFit type not recognized. Exiting.\n')
    sys.exit()

if not make_plots:
    mpl.use('pdf')
    
# **********************************************************************************************************
# DEFINE ANY FUNCTIONS AND SET PLOTTING OPTIONS
# **********************************************************************************************************

# cut to select high energy peak
cl_slope = -0.1875
def peak_sep(x):
    return cl_slope*x+3960

# cut out any other peaks
def cl_cut(x,y):
    return y>100

# set plotting style
plt.rc('figure', dpi=200, figsize=(4,3), facecolor='w')
plt.rc('savefig', dpi=200, facecolor='w')
plt.rc('lines', linewidth=3)
pkw = dict(cmap='viridis',vmin=0., vmax=.5)

# *********************************************************************************************************
# LOAD TPC AND TRUE LIGHTMAP MODEL
# *********************************************************************************************************

# load TPC used for training
print('\nLoading TPC geometry used for lightmap training...\n')
with open(sim_dir+'tpc.pkl', 'rb') as handle:
    tpc = pickle.load(handle)
print(tpc)

# redefine TPC as reduced volume within field rings and between cathode and anode
tpc.r = 566.65
tpc.zmax = -402.97 #tpc.zmax-19.#1199#17#19.
tpc.zmin = -1585.97#tpc.zmax-1183.#3#21#1183.

# load model
print('\nLoading true lightmap model...\n')
lm_true = LightMap.load_model(sim_dir+'true-lm', 'LightMapHistRZ')
print(lm_true, '\n')

# plot the original lightmap
fig,ax = plt.subplots(figsize=(3.5,5))
d = plot_lm_rz(ax,lm_true,tpc)
ax.set_title('Truth Lightmap')
if make_plots:
    fig.tight_layout()
fig.savefig(path+'original.png',bbox_inches='tight')

# *********************************************************************************************************
# LOOP THROUGH ALL DATASETS
# *********************************************************************************************************

# loop through and collect the specified number of events
data = []
print('Collecting events from {:d} processed simulation files...\n'.format(len(input_files)))
for data_file in input_files:
    input_file = gzip.open(data_file,'rb')
    this_df = pickle.load(input_file)
    data.append(this_df)
    input_file.close()

# add to pandas dataframe
data = pd.concat(data,ignore_index=True)

# get a sample of events from the full set
val_factor = 1
val_string = ''
if validation:
    val_factor = 2
    val_string = 'and {:d} events for validation '.format(events)

print('Sampling {:d} events for calibration '.format(events)+val_string+'using seed {:d}...\n'.format(seed))
data = data.sample(val_factor*events, random_state=seed)
    
# compute z from the drift time and TPC dimensions
# drift velocity from 2021 sensitivity paper
# TPC center + (TPC length)/2 - TPC top to anode
# 1022.6 + 1277/2 - 18.87
data['z'] = -402.97 - data.weighted_drift.values*1.709

# *****************************************************************************************************
# APPLY CUTS AND DETERMINE QUANTITIES FOR NN TRAINING
# *****************************************************************************************************

print('Applying quality cuts to the data...\n')

# cut events larger than a given size (not usually used)
data_size = len(data.index)
evt_size_cut = 5000000.
cuts = data['event_radius']<evt_size_cut
after_size = len(data[cuts].index)

# cut events with no charge signal
cuts = cuts & ~(data['evt_charge_including_noise']==0)
after_elec = len(data[cuts].index)

# cut events with NaN z values
cuts = cuts & ~(np.isnan(data['z']))
after_drift = len(data[cuts].index)

# cut events with no photons produced
cuts = cuts & ~(data['Observed Light']==0)
after_photon = len(data[cuts].index)

# apply fiducial cut
zlim = [tpc.zmin+standoff,tpc.zmax-standoff]
rlim = [0,tpc.r-standoff]
inside_z = abs(data.z.values-(zlim[1]-zlim[0])/2.-zlim[0])>(zlim[1]-zlim[0])/2.
inside_r = abs(data.weighted_radius.values-(rlim[1]-rlim[0])/2.-rlim[0])>(rlim[1]-rlim[0])/2.
cuts = cuts & (~inside_z & ~inside_r)
after_fiducial = len(data[cuts].index)

# sample based on number of photons generated
qe = 0.186

# separate low and high energy peaks
data['peak'] = np.ones(len(data['Observed Light']))
peak_cond = peak_sep(data.evt_charge_including_noise.values) < data['Observed Light']
data.loc[peak_cond,'peak'] = 2

# cut out data that is not in one of the peaks
cut_cond = cl_cut(data.evt_charge_including_noise.values,data['Observed Light'])
cuts = cuts & cut_cond
after_chargelight = len(data[cuts].index)

# print information about size of events
med_size = np.median(data['event_radius'][cuts])
vol_TPC = np.pi*tpc.r**2*(tpc.zmax-tpc.zmin)
print('Median event size: {:.2f} mm'.format(med_size))
print('Number of events at which there is overlap: {:.0f}'.format(vol_TPC/med_size**3))

# print results of cuts with efficiency
print('Events before event size cut: '+str(data_size))
print('Events after event size cut: '+str(after_size))
print('Event size cut efficiency: {:.1f} %'.format(after_size*100./data_size))
print('Events after thermal electron cut: '+str(after_elec))
print('Thermal electron cut efficiency: {:.1f} %'.format(after_elec*100./after_size))
print('Events after z quality cut: '+str(after_drift))
print('z quality cut efficiency: {:.1f} %'.format(after_drift*100./after_elec))
print('Events after photon cut: '+str(after_photon))
print('Photon cut efficiency: {:.1f} %'.format(after_photon*100./after_drift))
print('Events after fiducial cut: '+str(after_fiducial))
print('Fiducial cut efficiency: {:.1f} %'.format(after_fiducial*100./after_photon))
print('Events after charge/light cut: '+str(after_chargelight))
print('Charge/light cut efficiency: {:.1f} %\n'.format(after_chargelight*100./after_fiducial))

# compute mean number of photons for each peak
peaks = np.array((0,0))
for j in range(2):
    peaks[j] = np.mean(data['Observed Light'][(data['peak']==j+1) & cuts])
    
# compute the efficiency using the predicted mean of each peak
print('Computing the efficiency from the data selected...\n')
mean_eff = np.mean(data['Observed Light'][(data.peak.values==2) & cuts]/data['fInitNOP'][(data.peak.values==2) & cuts])
data['eff'] = data['Observed Light']*mean_eff/(qe*peaks[np.array(data.peak.values-1,dtype=int)])

np.savetxt(path+'effic_'+name+'.txt',np.array(data.eff.values))

# *****************************************************************************************************
# PLOT ALL DATA BEFORE TRAINING
# *****************************************************************************************************

if make_plots:
    print('Saving some relevant plots in {:s}\n'.format(path))
    make_figs(tpc,lm_true,data,cuts,path,name,rlim,zlim,peak_sep)
    plt.show()

if not train:
    sys.exit()

# *****************************************************************************************************
# FIT A LIGHTMAP MODEL TO THE DATA
# *****************************************************************************************************

# create separate validation dataset
validation_split = 0
if validation:
    validation_split = 0.5

# define new training set
if both_peaks:
    train_again = data.weighted_x.values[cuts], data.weighted_y.values[cuts], data.z.values[cuts], data.eff.values[cuts]
    print('Training on both peaks with {:d} events total.\n'.format(int(round(len(train_again[0])*(1-validation_split)))))
else:
    train_again = data.weighted_x.values[(data['peak']==2) & cuts],data.weighted_y.values[(data['peak']==2) & cuts],\
                  data.z.values[(data['peak']==2) & cuts],data.eff.values[(data['peak']==2) & cuts]
    print('Training on one peak with {:d} events total.\n'.format(int(round(len(train_again[0])*(1-validation_split)))))

# define neural net lightmap
if fit_type=='NN':
    if layers==None:
        layers = [512, 256, 128, 64, 32]
    if learning_rate==None:
        learning_rate = 0.0004
    if ensemble_size==None:
        ensemble_size = 3
    if batch_size==None:
        batch_size = 64
    lm_again = LightMap.total.LightMapNN(tpc, epochs=10, batch_size=batch_size,
                                         hidden_layers=layers, lr=learning_rate,
                                         validation_split=validation_split)

# define Gaussian kernel smoothing lightmap
if fit_type=='KS':
    ensemble_size = 1
    if sigma==None:
        sigma = 50
    lm_again = LightMap.total.LightMapKS(tpc, sigma, points=50, batch_size=1000)

# fit the lightmap
print('Fitting a lightmap to the data...\n')
times = []
for i in range(ensemble_size):
    starttime = time.time()
    lm_again.fit(*train_again)
    endtime = time.time()
    times.append(endtime-starttime)

# save losses for NN model
if fit_type=='NN':
    losses = []
    val_losses = []
    for i in range(ensemble_size):
        losses.append(np.array(lm_again.histories[i].history['loss']))
        if validation:
            val_losses.append(np.array(lm_again.histories[i].history['val_loss']))
else:
    losses = None
    val_losses = None

# save fitted lightmap
LightMap.save_model(path+'LightMap_'+name,lm_again.kind,lm_again)

# *****************************************************************************************************
# PLOT FINAL LIGHTMAP FOR SIMULATED DATA AND SAVE RESULTS
# *****************************************************************************************************

# make results plots and compute fitting metrics
print('\nPlotting and saving the results...\n')
mean,var,h_true,h_again,h_true_uniform,h_again_uniform = plot_results(tpc,lm_true,lm_again,rlim,zlim,path,name)

# print fitting results
print('Fitting results')
print('-----------------------------')
print('Mean: {:.6f}'.format(mean))
print('Standard deviation: {:.6f}'.format(np.sqrt(var)))
print('-----------------------------\n')

# pickle parameters
params_list = [[name,
                fit_type,
                events,
                standoff,
                int(both_peaks)+1,
                np.array(layers),
                learning_rate,
                ensemble_size,
                batch_size,
                sigma,
                len(train_again[0]),
                np.array(times),
                np.array(losses),
                np.array(val_losses),
                np.sqrt(var),
                mean,
                h_true,
                h_again,
                h_true_uniform,
                h_again_uniform
                ]]

columns = ['name',
           'fit_type',
           'nominal_events',
           'fid_cut',
           'num_peaks',
           'layers',
           'learning_rate',
           'ensemble_size',
           'batch_size',
           'sigma',
           'num_events',
           'times',
           'losses',
           'val_losses',
           'accuracy_std_dev',
           'accuracy_mean',
           'hist_true',
           'hist_again',
           'hist_true_uniform',
           'hist_again_uniform'
           ]

params = pd.DataFrame(params_list,columns=columns)
params.to_pickle(path + name + '_results.pkl',compression='gzip')

"""
# save data to make reconstruction pipeline plot
weighted_radius = data.weighted_radius.values[cuts]
z = data.z.values[cuts]
eff = data.eff.values[cuts]
tpc_r = tpc.r
tpc_zmin = tpc.zmin
tpc_zmax = tpc.zmax
plot_stuff = weighted_radius,z,eff,tpc_r,tpc_zmin,tpc_zmax
pickle.dump(plot_stuff,open('sample_data.pkl','wb'))
"""

print('Results saved to {:s}'.format(path+name+'_results.pkl'))

if make_plots:
    plt.show()
