## Pre-requisites
1. Install MuJoCo as explained in [here](https://github.com/ashBabu/Utilities/wiki/Useful#install-mujoco-youtube)
2. Install ```mujoco-py``` as explained in [here](https://github.com/ashBabu/Utilities/wiki/Useful#install-mujoco-py) or [mujoco-py github](https://github.com/openai/mujoco-py#install-mujoco)
3. Tensorflow 2.x to run DDPG

## Installation
<link rel="stylesheet" type="text/css" href="style.css">

1. Clone the repository and inside ``` spaceRobot_RL ``` folder (where the ```setup.py``` is), run
``` pip install -e . ```

## Test Installation

``` python
import gym
import spacecraftRobot
env = gym.make('SpaceRobot-v0')
```
## Test DDPG
```python train_spaceRobot.py```
<!-- ![spacecraftRobot](free_floating_spacecraft.png?raw=true) -->

## Test POLO 
Plan Online Learn Offline is a value function learning paper using trajectory optimization.
Details availabe at [POLO](https://sites.google.com/view/polo-mpc#h.p_tuHcQ8dmvkmr)
1. ```cd POLO```
2. ```python job_script1.py```

**Note**: Needs ```mujoco-py```


<img style="float: left;" title="Free Floating Spacecraft with Robot Arm" src="free_floating_spacecraft.png" alt="spacecraftRobot" width="400" height="400"/>



   The debris could be stationary or rotating about some arbitrary axis. The aim is to capture the debris and bring it back to a station.

   The big cube is assumed to be the spacecraft and carries a robot arm which has seven rotary joints and has a gripper with two linear joints.
   
### Reward:
Currently it is composed of three parts. 

dist = distance between end-effector and debris  # to reach near the debris

act = 0.0001 * np.dot(action, action) # to obtain minimal actuator torques

rw_vel = 0.001 * sum of squares of base linear and angular velocites # for reaction-less control

reward = -(dist + act + rw_vel)

**Note:** As of now there is no reward for grasping
### Current issues:
   
    1. How to set reward for grasping? 
    Is it could be just negative of the distance between the end-effector and debris 
    or add a camera and get images at every instant of time?

### Ideas
1. Demonstrated replay buffer
2. Reaction-less control

