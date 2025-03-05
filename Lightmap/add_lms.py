import numpy as np
import argparse
import LightMap
import os
import gzip
import pickle
import histlite as hl

sim_dir = os.getenv('SIM_DIR')

parser = argparse.ArgumentParser()
parser.add_argument('name',type=str)
parser.add_argument('standoff',type=float)
args = parser.parse_args()
name = args.name
standoff = args.standoff

# load TPC used for training
print('\nLoading TPC geometry used for lightmap training...\n')
with open(sim_dir+'tpc.pkl', 'rb') as handle:
    tpc = pickle.load(handle)
print(tpc)

# load true lightmap model
lm_true = LightMap.load_model(sim_dir+'true-lm', 'LightMapHistRZ')
print(lm_true)

# redefine TPC as reduced volume within field rings and between cathode and anode
tpc.r = 566.65
tpc.zmax = -402.97 #tpc.zmax-19.#1199#17#19.
tpc.zmin = -1585.97#tpc.zmax-1183.#3#21#1183.

result_file = sim_dir+'outputs/'+name+'_results.pkl'
print(result_file)

# loop through and get all results files
result = gzip.open(result_file,'rb')
this_df = pickle.load(result)
print(this_df)

zlim = [tpc.zmin+standoff,tpc.zmax-standoff]
rlim = [0,tpc.r-standoff]

lms = LightMap.load_model(sim_dir+'outputs/LightMap_'+name, 'LightMapNN')
print(lms)
hidden_layers = lms.hidden_layers
epochs = lms.epochs
batch_size = lms.batch_size

errs = []
means = []
for lm_model in lms.models:
    rang = (rlim[0],rlim[1]), (0, 2*np.pi), (zlim[0],zlim[1])
    lm = LightMap.total.LightMapNN(tpc,epochs=epochs,batch_size=batch_size,hidden_layers=hidden_layers)
    lm.models = [lm_model]
    f = lambda r, theta, z: lm(r, theta, z, cyl=True)
    h_again_3d = hl.hist_from_eval(f, vectorize=False, bins=50, range=rang)
    f = lambda r, theta, z: lm_true(r, theta, z, cyl=True)
    h_true_3d = hl.hist_from_eval(f, vectorize=False, bins=50, range=rang)
    var = np.var(h_true_3d.values/h_again_3d.values)
    errs.append(np.sqrt(var))
    mean = np.mean(h_true_3d.values/h_again_3d.values)
    means.append(np.sqrt(mean))
    print('')
    print(lm)
    print(var)
    print(mean)
    del lm

this_df['fid_cut'] = [standoff]
this_df['std_array'] = [np.array(errs)]
this_df['mean_array'] = [np.array(means)]

print(this_df)
this_df.to_pickle(sim_dir+'outputs/'+name+'_results_new.pkl',compression='gzip')
