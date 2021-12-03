import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import histlite as hl
import uproot as up
import pandas as pd
import sys
import os
import pickle
import copy
import argparse
import time
from ComputeObservables import ComputeObservedLight
from ComputeObservables import ComputeObservedCharge


##############################################################################
# Get arguments
##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-input_file',type=str)
parser.add_argument('-output_dir',type=str)
parser.add_argument('-num_events',type=int)
parser.add_argument('-pads',type=bool,default=False)
args = parser.parse_args()
input_file = args.input_file
output_dir = args.output_dir
num_events = args.num_events
pads_flag = args.pads

filename = input_file.split('/')[-1].split('.')[0]

print('Name of file: {}'.format(filename))

##############################################################################
# Open file
##############################################################################
start_time = time.time()

if not input_file.endswith('.root'):
   print('\n\nERROR: input file is not a ROOT file.\n\n')
   sys.exit()

ElecFile = up.open(input_file)
elec = ElecFile['Event/Elec/ElecEvent']
sim = ElecFile['Event/Sim/SimEvent']

columns_to_get = [x for x in sim.allkeys() if sim[x].interpretation is not None and x != b"fBits"]
dfsim = sim.arrays(columns_to_get,outputtype=pd.DataFrame, entrystart=0, entrystop=num_events)


elec_columns_to_get = [ 'fElecChannels.fChannelCharge',\
                        'fElecChannels.fChannelLocalId',\
                        'fElecChannels.fChannelNoiseTag',\
                        'fElecChannels.fChannelTime',\
                        'fElecChannels.fXPosition',\
                        'fElecChannels.fYPosition',\
                        'fNTE',]

#columns_to_get = [x for x in elec.allkeys() if elec[x].interpretation is not None and x != b"fBits"]
dfelec = elec.arrays( elec_columns_to_get, outputtype=pd.DataFrame, entrystart=0, entrystop=num_events )

print('Finished loading {} at {:5.5}s'.format(filename,time.time()-start_time))


##############################################################################
# From sim data, compute observed charge and light 
##############################################################################
observed_light = ComputeObservedLight( dfsim )
observed_charge = ComputeObservedCharge( dfelec, pads_flag )

observables_dict = {}
for key, value in observed_light.items():
   observables_dict[key] = value

for key, value in observed_charge.items():
   observables_dict[key] = value

output_df = pd.DataFrame(observables_dict)

output_df.to_pickle(output_dir + filename + '_REDUCED.pkl',compression='gzip')
