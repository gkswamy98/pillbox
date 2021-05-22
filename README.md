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

## Citing our Work
```
@article{swamy2021moments,
  author       = {Gokul Swamy, Sanjiban Choudhury, J. Andrew Bagnell, Zhiwei Steven Wu},
  title        = {Of Moments and Matching: Trade-offs and Treatments in Imitation Learning},
  conference   = {Proceedings of the 38th International Conference on Machine Learning},
  url          = {https://arxiv.org/abs/2103.03236},
}
```
