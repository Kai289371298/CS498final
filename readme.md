The wandb key has been commented out in BC.py for privacy; fill in "XXXXXXX" with your key to make it runnable.

## Dependency 

torch==1.10.0

stable-baselines3==1.6.2

scipy 1.8.0

scikit-learn 1.2.2

matplotlib

numpy

mujoco_py==2.1.2.14 (and mujoco210)

imageio==2.16.1

gym==0.21.0

dm-control==1.0.7

dmc2gym (https://github.com/denisyarats/dmc2gym)

wandb

## How to Run

For the comments below, (A, B) \in {(walker, walk), (reacher, easy), (cartpole, swingup)}

**generation of dataset**

python SAC_expert.py --train_agent 2 --domain_name A --task_name B

**running BC**

python BC.py --domain_name A --task_name B --encode_method simclr / vicreg / vinn --actor_method bc / nn_euccore_soft / nn_mancore_hard --arch resnet18

python BC.py --domain_name A --task_name B --encode_method da --aug crop / translate --actor_method bc --arch resnet18

use --load 1 if you have trained representations before and want to directly start BC/NN. 

