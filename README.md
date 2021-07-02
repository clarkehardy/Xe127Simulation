# Xe127Simulation

This repository contains the scripts to simulate and analyze xenon-127 calibration data using `nexo-offline`.

## Setup

Running this code requires access to some existing files, including the nEXO lightmap and TPC geometry. Specify the path to the folder where this data is stored using the `$SIM_DIR` environment variable.

```python
export SIM_DIR="path/to/folder/"
```

## Usage

[Cards](https://github.com/clarkehardy/lm-analysis/tree/master/Cards) contains the Giant4 macros as well as scripts to run `nexo-offline` and submit multiple jobs to the SLURM queue.

[DataProcessing](https://github.com/clarkehardy/lm-analysis/tree/master/DataProcessing) contains Python scripts to process the raw Root files and produce the reduced data that can be used for electron lifetime and lightmap analysis.

[Lightmap](https://github.com/clarkehardy/lm-analysis/tree/master/Lightmap) contains the scripts to reconstruct the nEXO lightmap from the Xe-127 calibration data. This requires the `LightMap` package from `nexo-offline`.
