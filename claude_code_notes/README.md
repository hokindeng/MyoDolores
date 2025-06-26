# MyoDolores: Comprehensive Biomechanics & Robotics Platform

🚀 **Complete setup in 3 steps:** `git clone → ./scripts/setup_environment.sh → ./scripts/sync_data.sh --all`

MyoDolores is a unified platform for biomechanics research, motion analysis, and humanoid robotics training, featuring 28,000+ motion capture sequences and state-of-the-art RL frameworks.

## 🎯 Key Features

- **🎭 Massive Motion Dataset**: 28,441 H5 motion files across 8 domains (200GB)
- **🤖 Multi-Robot Support**: Unitree (Go2, G1, H1, H1_2), humanoid models
- **🏃 Motion Tracking**: Real-time human motion → robot motion retargeting
- **🎮 RL Training**: Isaac Gym, Genesis, and Isaac Sim integration
- **🧬 Biomechanics**: MuJoCo-based physics simulation and analysis
- **☁️ Cloud-Ready**: Automated AWS S3 data sync and validation

## 📊 Dataset Overview

| Dataset | Size | Files | Domain | Description |
|---------|------|-------|---------|-------------|
| **game_motion_output** | 87GB | 8,000+ | Gaming | Combat, exploration, interactive motions |
| **HAA500_output** | 57GB | 3,000+ | Human Actions | Walking, running, sports, daily activities |
| **aist_output** | 21GB | 1,500+ | Dance | Professional choreographed sequences |
| **kungfu_output** | 15GB | 1,200+ | Martial Arts | Kung fu, Tai chi, combat techniques |
| **humman_output** | 11GB | 800+ | Biomechanics | Scientific motion analysis data |
| **perform_output** | 6.9GB | 600+ | Performance | Theatrical and stage motions |
| **animation_output** | 4.7GB | 1,000+ | Animation | Stylized character animations |
| **dance_output** | 2.4GB | 400+ | Dance | Extended dance and movement |

## ⚡ Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Clone repository with submodules
git clone --recursive <your-repo-url> MyoDolores
cd MyoDolores

# One-command setup (creates conda env, installs dependencies)
./scripts/setup_environment.sh --gpu  # For GPU systems
# or
./scripts/setup_environment.sh        # For CPU-only systems

# Sync all motion data (~200GB download)
./scripts/sync_data.sh --all

# Validate everything is working
./scripts/setup_validation.sh
```

### Option 2: Manual Setup
```bash
# 1. Environment setup
conda create -n myodolores python=3.8 -y
conda activate myodolores

# 2. Install core packages
pip install mujoco==3.2.4 numpy h5py torch jax

# 3. Install frameworks
pip install -e myo_api/
pip install -e ASAP/
pip install -e myo_retarget/

# 4. Configure AWS and sync data
aws configure
./scripts/sync_data.sh --all
```

## 🏗️ Architecture

```
MyoDolores/
├── 🎯 ASAP/                    # HumanoidVerse RL training framework
├── 🦘 extreme-parkour/        # Legged robot parkour with Isaac Gym
├── 🧬 myo_api/               # MuJoCo-based biomechanics API
├── 🎪 myo_retarget/          # Motion retargeting pipeline
├── 🏃 myo_model_internal/    # MyoSkeleton universal human model
├── 🤖 unitree_rl_gym/        # Unitree robot training environment
├── 📊 myo_data/              # 28K+ motion sequences (git ignored)
└── 🛠️ scripts/               # Automated setup and validation tools
```

## 🚀 Training Examples

### Motion Tracking with ASAP
```bash
conda activate myodolores
cd ASAP

# Train humanoid to follow human walking motion
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  robot.motion.motion_file="../myo_data/HAA500_output/Walking/Walking_Normal_01.h5" \
  num_envs=4096 \
  project_name=MotionTracking \
  experiment_name=HumanWalking

# Train with dance motion
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  robot.motion.motion_file="../myo_data/aist_output/dance_ballet/ballet_sequence_01.h5" \
  num_envs=2048
```

### Locomotion Training
```bash
# Train G1 humanoid for general locomotion
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=locomotion \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  num_envs=4096
```

### Unitree Robot Training
```bash
cd unitree_rl_gym/legged_gym/scripts

# Train Go2 quadruped
python train.py --task=go2 --headless=True --num_envs=4096

# Train G1 humanoid
python train.py --task=g1 --headless=True --num_envs=2048

# Sim2Real deployment
python deploy/deploy_real/deploy_real.py eth0 g1.yaml
```

### Motion Retargeting
```bash
cd myo_retarget

# Retarget human motion to robot
python retarget_motion.py \
  --input ../myo_data/HAA500_output/Running/Running_Fast_01.h5 \
  --target_model robot_g1.xml \
  --output retargeted_running.h5
```

## 🛠️ Development Tools

### Validation and Diagnostics
```bash
# Comprehensive system validation
./scripts/setup_validation.sh

# Quick environment check
./scripts/setup_validation.sh --quick

# Automated issue fixing
./scripts/setup_validation.sh --fix

# Data integrity verification
./scripts/verify_data.sh

# Detailed data statistics
./scripts/verify_data.sh --stats
```

### Data Management
```bash
# Sync specific datasets
./scripts/sync_data.sh HAA500_output
./scripts/sync_data.sh aist_output

# Preview sync operations
./scripts/sync_data.sh --dry-run --all

# Force re-download
./scripts/sync_data.sh --force HAA500_output

# List available datasets
./scripts/sync_data.sh --list
```

### Environment Management
```bash
# Activate environment
conda activate myodolores

# Or use project activation script
source activate_env.sh

# Export environment for sharing
conda env export > environment.yml

# Recreate environment
conda env create -f environment.yml
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **[SETUP_GUIDE.md](SETUP_GUIDE.md)** | 📋 Complete step-by-step setup instructions |
| **[CLAUDE.md](CLAUDE.md)** | 🤖 Project overview and framework commands |
| **[DATA_SETUP.md](DATA_SETUP.md)** | 📊 Data synchronization and management |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | 🔧 Common issues and solutions |
| **[myo_data/DATASETS.md](myo_data/DATASETS.md)** | 📈 Detailed dataset descriptions |

## 🎮 Supported Platforms

### Simulators
- **Isaac Gym**: High-performance parallel training
- **Isaac Sim**: Photorealistic simulation
- **Genesis**: Next-generation physics simulation
- **MuJoCo**: Precise biomechanical modeling

### Robots
- **Unitree Go2**: Quadruped with advanced mobility
- **Unitree G1**: Humanoid with dexterous manipulation
- **Unitree H1/H1_2**: Research humanoid platforms
- **MyoSkeleton**: Universal human biomechanical model

### Training Frameworks
- **ASAP/HumanoidVerse**: Multi-simulator RL platform
- **Extreme Parkour**: Legged robot agility training
- **Unitree RL Gym**: Production robot training
- **Myo Retarget**: Motion adaptation pipeline

## 💻 System Requirements

### Minimum (CPU Training)
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows WSL2
- **Memory**: 8GB RAM
- **Storage**: 250GB free space
- **Python**: 3.8+

### Recommended (GPU Training)  
- **GPU**: NVIDIA RTX 3090+ (24GB VRAM)
- **Memory**: 32GB RAM
- **Storage**: SSD with 500GB+ free space
- **CUDA**: 11.8+

## 🚨 Common Issues & Solutions

### Environment Issues
```bash
# Conda not found
./scripts/setup_environment.sh  # Installs conda automatically

# Python version conflicts  
conda activate myodolores
python --version  # Should show 3.8+

# Package conflicts
./scripts/setup_environment.sh --force  # Clean reinstall
```

### Data Issues
```bash
# AWS credentials not configured
aws configure

# Data sync fails
./scripts/sync_data.sh HAA500_output  # Sync specific dataset
./scripts/verify_data.sh  # Check data integrity

# Insufficient disk space
du -sh myo_data/*/  # Check dataset sizes
rm -rf myo_data/game_motion_output  # Remove largest dataset if needed
```

### Training Issues
```bash
# CUDA out of memory
num_envs=2048  # Reduce environment count

# Motion file not found
./scripts/verify_data.sh HAA500_output  # Verify data
find myo_data/ -name "*.h5" | head -5  # List available files

# Isaac Gym not found
# Isaac Gym requires manual installation from NVIDIA
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
./scripts/setup_environment.sh --dev

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

### Adding New Datasets
1. Add dataset to S3 bucket with proper structure
2. Update `EXPECTED_DATASETS` in `scripts/verify_data.sh`
3. Update documentation in `myo_data/DATASETS.md`

### Framework Integration
1. Add submodule: `git submodule add <repo-url> framework_name/`
2. Update `setup_environment.sh` to install dependencies
3. Add training examples to documentation

## 📈 Performance Benchmarks

### Training Performance (RTX 3090)
- **ASAP Motion Tracking**: 4096 envs @ 60 FPS
- **Unitree Locomotion**: 8192 envs @ 120 FPS  
- **Extreme Parkour**: 4096 envs @ 45 FPS

### Data Loading Performance
- **Local SSD**: ~500 motions/sec
- **Network Storage**: ~100 motions/sec
- **S3 Sync**: ~150 Mbps typical throughput

## 🎓 Research Applications

### Published Work
- Motion tracking for humanoid robots
- Biomechanical analysis of athletic performance
- Real-time motion retargeting systems
- Legged robot agility training

### Potential Applications
- **Medical**: Rehabilitation robotics, gait analysis
- **Entertainment**: Motion capture, animation, gaming
- **Sports**: Performance analysis, training optimization
- **Robotics**: Human-robot interaction, assistive devices

## 📄 License

This project contains multiple components with different licenses:
- **ASAP**: MIT License
- **MuJoCo**: Apache 2.0 License  
- **Motion Data**: Various licenses (see `myo_data/DATASETS.md`)
- **Project Code**: MIT License

Please review individual component licenses before commercial use.

## 🙏 Acknowledgments

- **NVIDIA** for Isaac Gym and Isaac Sim
- **DeepMind** for MuJoCo physics engine
- **Unitree Robotics** for robot platforms and SDK
- **Motion Capture Communities** for dataset contributions

---

## 🚀 Getting Started Now

Ready to dive in? Choose your path:

### 🏃‍♂️ I want to start training immediately:
```bash
git clone --recursive <repo-url> MyoDolores && cd MyoDolores
./scripts/setup_environment.sh --gpu && ./scripts/sync_data.sh HAA500_output
conda activate myodolores && cd ASAP
python humanoidverse/train_agent.py +simulator=isaacgym +exp=locomotion
```

### 🔬 I want to analyze motion data:
```bash
git clone --recursive <repo-url> MyoDolores && cd MyoDolores  
./scripts/setup_environment.sh && ./scripts/sync_data.sh aist_output
conda activate myodolores
python -c "import myo_api as myo; traj = myo.mjTrajectory.load('myo_data/aist_output/dance_001/dance_001_00.h5')"
```

### 📚 I want to understand the platform:
Start with **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for comprehensive instructions, then explore **[CLAUDE.md](CLAUDE.md)** for framework details.

---

**Welcome to MyoDolores - where human motion meets robotic intelligence!** 🤖💃
