#!/bin/bash
source /usr/gapps/nexo/setup.sh
source /g/g92/hardy27/lm-analysis/setup.sh
cd /g/g92/hardy27/nexo-offline/build
source setup.sh
cd Cards
python3 ./RunDetSim.py --evtmax $1 --seed $2 --run ./examples/TPCVessel_220Rn.mac --output $DATA_DIR/rn220_sims/TPCVessel_220Rn_$3.root > $DATA_DIR/rn220_sims/TPCVessel_220Rn_$3.out 2>$DATA_DIR/rn220_sims/TPCVessel_220Rn_$3.err
