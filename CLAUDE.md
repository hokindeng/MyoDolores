# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

MyoAvatar is a comprehensive biomechanics and robotics research platform using **git submodules** for modular development:

### Core Submodules
1. **ASAP/** - Humanoid whole-body motion tracking and control using HumanoidVerse framework
2. **extreme-parkour/** - Legged robot parkour training with Isaac Gym  
3. **myo_api/** - MuJoCo-based API for motion capture, biomechanics simulation, and model manipulation
4. **myo_retarget/** - Motion retargeting pipeline for converting motion capture to joint angles
5. **myo_model_internal/** - MyoSkeleton universal human model and biomechanical assets
6. **unitree_rl_gym/** - Unitree robot (Go2, G1, H1, H1_2) reinforcement learning training environment

### Training Data
7. **myo_data/** - Comprehensive motion capture datasets (8,000+ motions from HAA500, AIST, dance, animation, game motion, kungfu, performance)

## Submodule Management

**Initialize all submodules:**
```bash
git submodule update --init --recursive
```

**Update submodules:**
```bash
git submodule update --remote
```

**Work on specific submodule:**
```bash
cd ASAP  # or any submodule
git checkout main
git pull origin main
# Make changes, commit, push as normal
```

## Key Commands

### ASAP (HumanoidVerse)

**Setup:**
```bash
# Create conda environment
conda create -n hvgym python=3.8
conda activate hvgym

# Install dependencies
pip install -e ASAP/
pip install -e ASAP/isaac_utils/
```

**Training:**
```bash
# Motion tracking with myo_data
python ASAP/humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
robot.motion.motion_file="../myo_data/HAA500_output/Badminton_Underswing/Badminton_Underswing_00.h5" \
num_envs=4096 \
project_name=MotionTracking \
experiment_name=MyoData_Badminton

# Dance motion training with AIST dataset
python ASAP/humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
robot.motion.motion_file="../myo_data/aist_output/dance_001/dance_001_00.h5" \
num_envs=4096 \
project_name=MotionTracking \
experiment_name=MyoData_Dance

# Locomotion training  
python ASAP/humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096
```

**Evaluation:**
```bash
python ASAP/humanoidverse/eval_agent.py \
+checkpoint=logs/MotionTracking/[timestamp]-[experiment]/model_5800.pt
```

### Extreme Parkour

**Setup:**
```bash
# Create conda environment
conda create -n parkour python=3.8
conda activate parkour

# Install dependencies
cd extreme-parkour
pip install -e rsl_rl/
pip install -e legged_gym/
```

**Training:**
```bash
cd extreme-parkour/legged_gym/scripts

# Train base policy (8-10 hours on RTX 3090)
python train.py --exptid xxx-xx-WHATEVER --device cuda:0

# Train distillation policy (5-10 hours on RTX 3090)  
python train.py --exptid yyy-yy-WHATEVER --device cuda:0 --resume --resumeid xxx-xx --delay --use_camera
```

**Evaluation:**
```bash
cd extreme-parkour/legged_gym/scripts

# Play base policy
python play.py --exptid xxx-xx

# Play distillation policy
python play.py --exptid yyy-yy --delay --use_camera

# Save models for deployment
python save_jit.py --exptid xxx-xx
```

### Myo API

**Setup:**
```bash
# Install with GPU support
pip install -e 'myo_api[gpu, test]'
```

**Key Features:**
- **MuJoCo (mj)**: CPU-based physics simulation with scaling, marker attachment, motion storage
- **MuJoCo XLA (mjx)**: GPU-accelerated parallel simulation
- **MuJoCo Spec (mjs)**: Model creation and modification using MjSpec API
- **Utils**: H5 handling, logging, tensor operations, quaternion math

### Unitree RL Gym

**Setup:**
```bash
# Install dependencies (see unitree_rl_gym/doc/setup_en.md)
conda create -n unitree python=3.8
conda activate unitree
```

**Training:**
```bash
cd unitree_rl_gym/legged_gym/scripts

# Train (Go2, G1, H1, H1_2 supported)
python train.py --task=g1 --headless=True --num_envs=4096

# Play/visualize
python play.py --task=g1

# Sim2Sim (Mujoco)
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml

# Sim2Real (Physical deployment)
python deploy/deploy_real/deploy_real.py {net_interface} g1.yaml
```

### Myo Retarget

**Setup:**
```bash
# Install with GPU support for JAX-based optimization
pip install myo_retarget[gpu]
```

**Key Features:**
- Motion capture to joint angle retargeting
- GPU-accelerated optimization with JAX
- Biomechanical trajectory evaluation
- Streaming motion processing

## Architecture Overview

### ASAP/HumanoidVerse Framework

- **Multi-simulator support**: Isaac Gym, Isaac Sim, Genesis
- **Modular design**: Separates simulators, tasks, and algorithms
- **Configuration system**: Hydra-based config management with YAML files
- **Core components**:
  - `humanoidverse/train_agent.py` - Main training entry point
  - `humanoidverse/eval_agent.py` - Evaluation and policy export
  - `humanoidverse/envs/` - Environment implementations (locomotion, motion tracking)
  - `humanoidverse/agents/` - RL algorithms (PPO, etc.)
  - `humanoidverse/simulator/` - Simulator backends
  - `humanoidverse/config/` - Hydra configuration files

### Extreme Parkour

- **Isaac Gym based**: Legged robot parkour training
- **Two-stage training**: Base policy → distillation policy with camera
- **Core components**:
  - `legged_gym/scripts/train.py` - Training script
  - `legged_gym/scripts/play.py` - Evaluation and visualization
  - `legged_gym/envs/` - Environment definitions
  - `rsl_rl/` - RL training library

### Myo API Framework

- **Multi-backend support**: MuJoCo CPU, MuJoCo XLA (GPU), MuJoCo Spec
- **Modular design**: Core, scaling, marker, motion modules
- **Core components**:
  - `myo_api/mj/` - CPU-based MuJoCo API
  - `myo_api/mjx/` - GPU-accelerated MuJoCo XLA
  - `myo_api/mjs/` - MuJoCo Spec model editing
  - `myo_api/utils/` - Utilities for data handling

### Myo Data

- **Datasets**: HAA500, AIST, dance, animation, game motion, kungfu, performance
- **Format**: H5 files with motion capture data
- **Processing**: Scaled models and retargeted motions
- **Structure**: Organized by dataset type with processed outputs

### Myo Model Internal

- **MyoSkeleton**: Universal human skeletal model
- **Components**:
  - `myo_model/myoskeleton/` - Core skeletal models
  - `myo_model/markerset/` - Marker set definitions
  - `myo_model/meshes_hd/` - High-detail mesh assets
- **Formats**: XML models, STL meshes, marker sets

### Unitree RL Gym

- **Multi-robot support**: Go2, G1, H1, H1_2
- **Sim2Real pipeline**: Isaac Gym → Mujoco → Real deployment
- **Core components**:
  - `legged_gym/scripts/` - Training and evaluation scripts
  - `deploy/` - Deployment for simulation and real robots
  - Robot-specific configurations and pre-trained models

### Myo Retarget

- **Retargeting pipeline**: 3D points → joint angles
- **GPU acceleration**: JAX-based optimization
- **Core components**:
  - `myo_retarget/retargeter/` - Core retargeting algorithms
  - `myo_retarget/trajectory_eval/` - Motion evaluation
  - `myo_retarget/artifacts/` - Pre-trained models and configs

## Configuration System

ASAP uses Hydra for configuration management. Key config categories:
- `+simulator=` - Choose simulator (isaacgym, isaacsim, genesis)
- `+exp=` - Experiment type (locomotion, motion_tracking)
- `+robot=` - Robot model (g1, h1, etc.)
- `+rewards=` - Reward function configuration
- `+obs=` - Observation space configuration
- `+terrain=` - Terrain settings

## Development Notes

### ASAP/HumanoidVerse
- **Simulator switching**: Change simulator with one line (`+simulator=isaacgym`)
- **Multi-environment support**: Isaac Gym, Isaac Sim, Genesis
- **Headless training**: Use `headless=True` for faster training
- **Wandb integration**: Built-in experiment tracking
- **ONNX export**: Policy export for deployment
- **Motion retargeting**: Human motion → robot motion pipeline

### Cross-Platform Integration
- **Motion pipeline**: Raw MoCap → myo_retarget → myo_data → ASAP/unitree_rl_gym training
- **Model sharing**: MyoSkeleton (myo_model_internal) used across myo_api and myo_retarget
- **Sim2Real**: Unitree RL Gym supports Isaac Gym → Mujoco → Real robot deployment
- **Data formats**: H5 for motion data, XML for models, PT for trained policies
- **Submodule integration**: Each component can be developed independently while maintaining compatibility

### Key Dependencies
- **ASAP**: Isaac Gym/Sim/Genesis, Hydra, Wandb
- **Extreme Parkour**: Isaac Gym, RSL-RL
- **Myo API**: MuJoCo 3.2.4, MuJoCo-MJX, JAX
- **Myo Retarget**: JAX, Equinox, Optax, MuJoCo-MJX
- **Unitree RL Gym**: Isaac Gym, Unitree SDK2

## Memories

- to memorize