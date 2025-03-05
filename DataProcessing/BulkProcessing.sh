#!/bin/bash

for arg
do python DataProcessing.py -num_events 10000 -output_dir $SIM_DIR -input_file $arg
done
