import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import histlite as hl
import pickle,sys,os,argparse,LightMap
from Analysis import Input as In
from Utilities import Print, Initialize

# set plotting style
plt.rc('figure', dpi=200, figsize=(4,3), facecolor='w')
plt.rc('savefig', dpi=200, facecolor='w')
plt.rc('lines', linewidth=1.5)
pkw = dict(cmap='viridis',vmin=0., vmax=.5)

# function to plot LightMaps
def plot_lm_rz(ax, lm, tpc, theta=0, vectorize=False, cbar=True):
    f = lambda r, z: lm(r, (theta if vectorize else np.repeat(theta, z.size)), z, cyl=True)
    rang = (0, tpc.r), (tpc.zmin, tpc.zmax)
    h = hl.hist_from_eval(f, vectorize=vectorize, bins=1000, range=rang)
    d = hl.plot2d(ax, h, cbar=cbar, **pkw)
    if cbar:
        d['colorbar'].set_label(r'Collection Efficiency')
    ax.set_xlabel(r'$r$ (mm)')
    ax.set_ylabel(r'$z$ (mm)')
    ax.set_aspect('equal')
    return d

# get environment variables that define paths
sim_dir = os.getenv('SIM_DIR')

# command line arguments to choose which calibration data to use
parser = argparse.ArgumentParser()
parser.add_argument('-read_again',action='store_true',default=False)
args = parser.parse_args()
read_again = args.read_again

# ***********************************************************************************************************
# READ SIMULATION DATA FROM CHROMA HERE
# ***********************************************************************************************************

if read_again:
    # get list of all simulation files as array
    filelist = []
    with open('filelist.txt','r') as infile:
        for line in infile:
            filelist.append('{}/../akojamil/chroma/data/nexo/2020_sensitivity_lightmap_new/'.format(sim_dir)+line[:-1])

    # read out keys necessary for lightmap training
    Input = pd.DataFrame()
    Input['files'] = filelist
    Input['num'] = -1
    LMap = In.LightMap(Input)
    Keys = ['Origin', 'NumDetected'] 
    LMap.GetData(Keys=Keys, Files=filelist, Multi=False)
    LMap.PrintEfficiency()
    LMap.Reshape()
    with open('{}/LMap.pkl'.format(sim_dir),'wb') as pickle_file:
        pickle.dump(LMap,pickle_file)

else:
    with open('{}/LMap.pkl'.format(sim_dir),'rb') as infile:
        LMap = pickle.load(infile)

# extract efficiency and position information
efficiency = np.array(LMap.Data['Efficiency'])/100.
origin = np.array(LMap.Data['Origin'])
x,y,z = [origin[:,i] for i in range(3)]

# shift z to match coordinates from GEANT4 (from Ako)
z = z - 894.59

# define full TPC geometry
z_top = -384.1
z_bottom = -1661.1
r_ring = 626.65

# create cylindrical TPC from data
tpc = LightMap.TPC(r_ring,z_bottom,z_top)
print(tpc)

# data to create LightMapHistRZ object from
train = x,y,z,efficiency

# ***********************************************************************************************************
# SAVE LIGHTMAP OBJECT
# ***********************************************************************************************************

# pickle TPC for use in retraining
with open('{}/tpc.pkl'.format(sim_dir),'wb') as handle:
    pickle.dump(tpc, handle, protocol=pickle.HIGHEST_PROTOCOL)

# define LightMapHistRZ object
lm_true = LightMap.total.LightMapHistRZ(tpc,nr=2500,nz=5000,smooth=1)

# fit the lightmap to the Chroma data
lm_true.fit(*train)
print(lm_true)

# plot the result
fig,ax = plt.subplots(figsize=(3.5,5))
plot_lm_rz(ax,lm_true,tpc)
plt.tight_layout()
plt.show()

# saving model
model_dir = '{}/true-lm'.format(sim_dir)
LightMap.save_model(model_dir, lm_true.kind, lm_true)
