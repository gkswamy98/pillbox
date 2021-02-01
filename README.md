# pillbox 💊
Contains PyTorch implementation of the AdVIL and AdRIL algorithms.

## Setup
To install dependencies, run:
```bash
conda env create -f environment.yml
```

## Running Experiments
To train an expert, run:
```bash
python experts/train.py -e env_name
```

To train a learner, run:
```bash
python learners/train.py -a algo_name -e env_name -n num_runs
```
This package supports training via:
- Behavioral Cloning
- AdVIL
- SQIL
- GAIL
- AdRIL

## Visualizing Results
Run:
```bash
jupyter notebook
```
and open up vis.ipynb.
