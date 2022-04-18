# pillbox 💊
Contains PyTorch implementation of the MIMIC(Replay Estimation), AdVIL, AdRIL, and DAeQuIL algorithms.

## Setup
To install dependencies, run:
```bash
conda env create -f environment.yml
```

In addition, for mimic (replay estimation), there were some added packages all listed in the req.txt file. After creating the conda environment, install those necessary in that file [where necessary] to run this algorithm

### Adding in Submodule
This repo also relies on a modified stablesbaselines3 (https://github.com/DLR-RM/stable-baselines3) repo. Within this repo, download/clone the contents then add the file named
```stable_baselines3```
into the file named learners.

The path should look as follows
- experts
- learners
   - stable_baselines3
   - [other contents of learners folder]
- .gitignore
- README
- environment.yml
- vis.ipynb

## Running Experiments
To train an expert, run:
```bash
python experts/train.py -e env_name
```

To train a learner, run:
```bash
python learners/train.py -a algo_name -e env_name -n num_runs
```
For "mimicmd" specifically:
1. a. Determine whether to generate new BC simulated data or load existing [set bool 'load_d3']

   b. Determine whether to train new query nets for the soft oracles or load bool [set param 'load_query']

2. Run mimic-md with desired oracle [set 'distance_metric' param]
```bash
python learners/train.py -a "mimicmd" -e [env] -n [number of times]
```

This package supports training via:
- Behavioral Cloning
- AdVIL
- SQIL
- GAIL
- AdRIL

We also support a comparison of the following algorithms that require an interactive expert on both OpenAI Gym and a custom environment:
- DAgger
- DAeQuIL

To explore these algorithms and environments, run:
```bash
jupyter notebook
```
and open up learners/imm.ipynb.

## Visualizing Results
Run:
```bash
jupyter notebook
```
and open up vis.ipynb.

## Citing our Work
```
@article{swamy2021moments,
  author       = {Gokul Swamy and Sanjiban Choudhury and J. Andrew Bagnell and Zhiwei Steven Wu},
  title        = {Of Moments and Matching: A Game-Theoretic Framework for Closing the Imitation Gap},
  conference   = {Proceedings of the 38th International Conference on Machine Learning},
  url          = {https://arxiv.org/abs/2103.03236},
}
```
