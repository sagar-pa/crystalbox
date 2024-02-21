# CC-RL

This repo is the extension of the origional PCC-RL implmentation with a few notable changes: discreet action spaces, observation normalization with clipping, smart trace sampling, reward normalization and random starts.

## Installation
```bash
git clone https://github.com/sagar-pa/crystalbox.git
cd ./crystalbox/congestion_control
pip install -e .[extra]
```
### For Auxiliary/Shared Loss evaluation install
```bash
pip install git+https://github.com/sagar-pa/auxiliary_sb3.git
```

## Usage

The installed congestion control environment can be used as the following.
```python

import gym
import cc_rl

env = gym.make("cc-rl-v0")
obs = env.reset()
done = False
while not done:
  action = int(input())
  reward, obs, done, info = env.step(action)
  print(reward)
  print(obs)
  
```

### Conducting CrystalBox experiments
To run the code to complete the experiments done in the paper, first train the controller. This can be done going into training/ and running trainer.py with the standard random trace sampling config. To do this, ray, the multiprocessing library (used for synchronization across algorithm processes) must be started.
```bash
  cd training
  ray start --head
  python train.py --sampling_func random
```

If you wish to run the joint training controller as well, run that with:
```bash
  python train_auxiliary.py --sampling_func random --auxiliary_coef 1;
  python train_auxiliary.py --sampling_func random --auxiliary_coef 10;
  python train_auxiliary.py --sampling_func random --auxiliary_coef 50;
```


Once the controllers are trained, we can stop ray and go to crystalbox to start creating the datasets of states/actions/rewards
```bash
  ray stop
  cd ./../crystalbox
  python create_datasets.py --create_dataset --create_diff_dataset
```
If you trained the jointly trained controllers and also wish to evaluate them now, add --create_aux_datasets to the arguments like so:
```bash
  python create_datasets.py --create_dataset --create_diff_dataset --create_aux_datasets
```

Once the datasets are created, it is time to train crystalbox.
```bash
  python crystalbox_train.py --train --test
```

Finally, we can evaluate our models against sampling baselines
```bash
  python sampling_baselines.py --test
```

Optionally, we may also evaluate the jointly trained crystalbox variants with:
```bash
  python evaluate_shared_loss.py --auxiliary_coef 1;
  python evaluate_shared_loss.py --auxiliary_coef 10;
  python evaluate_shared_loss.py --auxiliary_coef 50;
```

These commands will create pickle files in the data directory in crystalbox/, which can then be loaded and used for plotting.