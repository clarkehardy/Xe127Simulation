import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pickle,sys,os,argparse,LightMap
from Analysis import Input as In
from Utilities import Print, Initialize

# get environment variables that define paths
home_dir = os.getenv('WORKING_DIR')
data_dir = os.getenv('DATA_DIR')

# command line arguments to choose which calibration data to use
parser = argparse.ArgumentParser()
parser.add_argument('-chroma',action='store_true',default=False)
parser.add_argument('-read_again',action='store_true',default=False)
args = parser.parse_args()
print(args)
chroma = args.chroma
read_again = args.read_again

if not chroma:
    # ***********************************************************************************************************
    # READ PHOTON BOMB DATA FROM GEANT4 HERE
    # ***********************************************************************************************************
    # import and read simulation files
    filenames = [
        '{}/blenardo/lightmap_example/PhotonBomb_gps_ActiveOptical_seed{}.nEXOevents.root'.format(data_dir, i)
        for i in range(1, 1001)]

    # read data and compute efficiency
    data = LightMap.read_files(filenames[:25])
    print(data.head())
    data['eff'] = data['fNOP'] / 100.0

    # create cylindrical TPC from data
    tpc = LightMap.TPC(data.r.max() + 0.1, data.z.min() - 0.1, data.z.max() + 0.1)
    print(tpc)

    # data to train neural net on
    train = data.x.values, data.y.values, data.z.values, data.eff.values

else:
    # ***********************************************************************************************************
    # READ SIMULATION DATA FROM CHROMA HERE
    # ***********************************************************************************************************
    if read_again:
        # get list of all simulation files as array
        filelist = []
        with open('filelist.txt','r') as infile:
            for line in infile:
                filelist.append('{}/akojamil/chroma/data/nexo/2020_sensitivity_lightmap_new/'.format(data_dir)+line[:-1])
            
        # read out keys necessary for lightmap training
        Input = pd.DataFrame()
        Input['files'] = filelist
        Input['num'] = -1
        LMap = In.LightMap(Input)
        Keys = ['Origin', 'NumDetected'] 
        LMap.GetData(Keys=Keys, Files=filelist, Multi=False)
        LMap.PrintEfficiency()
        LMap.Reshape()
        with open('{}/lm-analysis/LMap.pkl'.format(home_dir),'wb') as pickle_file:
            pickle.dump(LMap,pickle_file)

    else:
        with open('{}/lm-analysis/LMap.pkl'.format(home_dir),'rb') as infile:
            LMap = pickle.load(infile)

    # extract efficiency and position information
    efficiency = np.array(LMap.Data['Efficiency'])/100.
    origin = np.array(LMap.Data['Origin'])
    x,y,z = [origin[:,i] for i in range(3)]

    # create cylindrical TPC from data
    tpc = LightMap.TPC(np.sqrt(x**2+y**2).max(),z.min(),z.max())

    # data to train neural net on
    train = x,y,z,efficiency

# ***********************************************************************************************************
# SAVE TRAINED NN LIGHTMAP
# ***********************************************************************************************************
# pickle TPC for use in retraining
with open('{}/lm-analysis/tpc.pkl'.format(home_dir),'wb') as handle:
    pickle.dump(tpc, handle, protocol=pickle.HIGHEST_PROTOCOL)

# neural network LightMap fitting
layers = [512, 256, 128, 64, 32]
lm_nn = LightMap.total.LightMapNN(tpc, epochs=1, batch_size=32768, hidden_layers=layers)

# ensemble method seems to work best here -> take mean of 3 NN's
for i in range(3):
    lm_nn.fit(*train)
print(lm_nn)

# saving model
model_dir = '{}/lm-analysis'.format(home_dir)
LightMap.save_model(model_dir, lm_nn.kind, lm_nn)
