python3 RunDetSim_new.py \
--evtmax 10000 \
--seed $1 \
--run ./Macros/NEXO_Xe127_uniform.mac \
--digioutput $SIM_DIR/g4andcharge_10kevts_xe127_seed$1_10msEL.root \
--padsize 6. \
--wpFile /usr/gapps/nexo/nexo-offline/data/singleStripWP6mm.root \
--tilemap /usr/gapps/nexo/nexo-offline/data/tilesMap_6mm.txt \
--localmap /usr/gapps/nexo/nexo-offline/data/localChannelsMap_6mm.txt \
--noiselib /usr/gapps/nexo/nexo-offline/data/noise_lib_100e.root \
--skipEThreshold 0. \
--sampling 0.5 \
--eleclife 10000.
