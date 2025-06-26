# MyoDolores Complete Setup Guide

This comprehensive guide will help you set up the MyoDolores biomechanics and robotics platform on any computer, from complete beginner to advanced user.

## 📋 Quick Start Checklist

For experienced users, run this validation script first:
```bash
./scripts/setup_validation.sh --setup
```

For step-by-step setup, continue reading below.

## 🎯 Overview

MyoDolores is a comprehensive platform for:
- **Motion Capture Analysis**: 28,000+ motion sequences across 8 datasets  
- **Humanoid Training**: ASAP/HumanoidVerse with Isaac Gym integration
- **Robot Control**: Unitree robots (Go2, G1, H1, H1_2) with sim2real pipeline
- **Biomechanics**: MuJoCo-based simulation and myo_api framework

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows with WSL2
- **Memory**: 8GB RAM (16GB recommended for training)
- **Storage**: 250GB free space (200GB for data + workspace)
- **CPU**: 4+ cores (8+ recommended)
- **Internet**: Stable connection for downloading datasets

### Recommended for Training
- **GPU**: NVIDIA RTX 3090+ with 24GB VRAM
- **Memory**: 32GB RAM
- **Storage**: SSD with 500GB+ free space
- **CPU**: 16+ cores

### Software Dependencies
- **Python**: 3.8+ (3.8 recommended)
- **Git**: 2.20+
- **AWS CLI**: v2.0+
- **Conda**: Miniconda or Anaconda
- **CUDA**: 11.8+ (for GPU training)

## 📁 Repository Structure

```
MyoDolores/
├── ASAP/                          # HumanoidVerse training framework
├── extreme-parkour/               # Legged robot parkour training  
├── myo_api/                       # MuJoCo API and biomechanics tools
├── myo_retarget/                  # Motion retargeting pipeline
├── myo_model_internal/            # MyoSkeleton universal human model
├── unitree_rl_gym/               # Unitree robot RL training
├── myo_data/                     # Motion capture datasets (git ignored)
│   ├── animation_output/         # Animation motions (4.7GB)
│   ├── aist_output/              # AIST dance dataset (21GB)  
│   ├── dance_output/             # Dance sequences (2.4GB)
│   ├── game_motion_output/       # Video game motions (87GB)
│   ├── HAA500_output/            # Human action analysis (57GB)
│   ├── humman_output/            # Human motion biomechanics (11GB)
│   ├── kungfu_output/            # Martial arts motions (15GB)
│   └── perform_output/           # Performance motions (6.9GB)
├── scripts/
│   ├── sync_data.sh              # Data synchronization tool
│   ├── setup_validation.sh       # Setup validation and automation
│   └── setup_environment.sh      # Environment configuration
├── CLAUDE.md                     # Project instructions and commands
├── DATA_SETUP.md                 # Data synchronization guide
├── SETUP_GUIDE.md                # This comprehensive setup guide
└── TROUBLESHOOTING.md            # Common issues and solutions
```

## 🚀 Installation Steps

### Step 1: System Preparation

#### On Ubuntu/Linux:
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git curl wget build-essential

# Install Python 3.8+ if not present
sudo apt install -y python3.8 python3.8-dev python3-pip

# Install conda (if not installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### On macOS:
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install essential tools
brew install git python@3.8

# Install conda
brew install --cask miniconda
echo 'export PATH="/opt/homebrew/Caskroom/miniconda/base/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### On Windows (WSL2):
```bash
# Install WSL2 and Ubuntu from Microsoft Store first
# Then follow Ubuntu instructions above
```

### Step 2: Clone Repository

```bash
# Clone the main repository
git clone <your-repo-url> MyoDolores
cd MyoDolores

# Initialize all submodules (this may take several minutes)
git submodule update --init --recursive

# Verify submodules are initialized
ls -la  # Should see ASAP/, extreme-parkour/, myo_api/, etc.
```

### Step 3: Environment Setup

#### Create and Activate Conda Environment
```bash
# Create environment with exact Python version
conda create -n myodolores python=3.8 -y

# Activate environment
conda activate myodolores

# Verify activation
echo $CONDA_DEFAULT_ENV  # Should output: myodolores
```

#### Install Core Dependencies
```bash
# Essential packages
pip install numpy scipy matplotlib h5py pyyaml

# MuJoCo (specific version for compatibility)
pip install mujoco==3.2.4

# Machine learning frameworks
pip install torch torchvision torchaudio  # CPU version
# OR for CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# JAX for GPU acceleration
pip install jax[cpu]  # CPU version
# OR for CUDA:
# pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Additional scientific packages
pip install pandas scikit-learn opencv-python
```

### Step 4: AWS CLI Setup

#### Install AWS CLI v2
```bash
# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# macOS
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
rm AWSCLIV2.pkg

# Verify installation
aws --version  # Should show aws-cli/2.x.x
```

#### Configure AWS Credentials
```bash
# Configure with your AWS credentials
aws configure

# Enter when prompted:
# AWS Access Key ID: [your-access-key]
# AWS Secret Access Key: [your-secret-key]  
# Default region name: us-west-2
# Default output format: json

# Test S3 access
aws s3 ls s3://myo-data/
```

### Step 5: Project-Specific Setup

#### Install Framework Dependencies

**ASAP/HumanoidVerse:**
```bash
cd ASAP
pip install -e .
pip install -e isaac_utils/
cd ..
```

**Myo API:**
```bash
cd myo_api
pip install -e .[gpu,test]  # Include GPU and test dependencies
cd ..
```

**Extreme Parkour (optional):**
```bash
cd extreme-parkour
pip install -e rsl_rl/
pip install -e legged_gym/
cd ..
```

**Unitree RL Gym (optional):**
```bash
cd unitree_rl_gym
# Follow specific setup in unitree_rl_gym/doc/setup_en.md
cd ..
```

**Myo Retarget:**
```bash
cd myo_retarget  
pip install -e .[gpu]  # GPU-accelerated retargeting
cd ..
```

### Step 6: Data Setup

#### Quick Data Sync (All Datasets)
```bash
# Make sync script executable
chmod +x scripts/sync_data.sh

# Sync all datasets (~200GB download)
./scripts/sync_data.sh --all

# Monitor progress (in another terminal)
watch -n 5 'du -sh myo_data/*/'
```

#### Selective Data Sync
```bash
# List available datasets
./scripts/sync_data.sh --list

# Sync specific datasets
./scripts/sync_data.sh HAA500_output
./scripts/sync_data.sh aist_output

# Preview sync operations
./scripts/sync_data.sh --dry-run --all
```

### Step 7: Validation

#### Run Comprehensive Validation
```bash
# Full system validation
./scripts/setup_validation.sh

# Quick validation 
./scripts/setup_validation.sh --quick

# Fix common issues automatically
./scripts/setup_validation.sh --fix

# Validate only data setup
./scripts/setup_validation.sh --data-only
```

#### Manual Verification
```bash
# Test Python environment
python -c "
import mujoco
import numpy as np
import h5py
print('✓ All core packages imported successfully')
print(f'✓ MuJoCo version: {mujoco.__version__}')
"

# Test data loading
python -c "
import sys
sys.path.append('myo_api')
import myo_api as myo
print('✓ Myo API imported successfully')
"

# Check data integrity
find myo_data/ -name "*.h5" | wc -l  # Should show ~28,000+ files
du -sh myo_data/*/  # Show sizes of each dataset
```

## 🏃‍♂️ Quick Start Examples

### Example 1: Load and Visualize Motion Data
```bash
cd myo_api
python examples/load_motion.py ../myo_data/HAA500_output/Walking/Walking_01.h5
```

### Example 2: Train Motion Tracking with ASAP
```bash
conda activate myodolores
cd ASAP

# Train with HAA500 walking motion
python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  robot.motion.motion_file="../myo_data/HAA500_output/Walking/Walking_Normal_01.h5" \
  num_envs=4096 \
  project_name=QuickStart \
  experiment_name=WalkingDemo
```

### Example 3: Motion Retargeting
```bash
cd myo_retarget
python examples/retarget_motion.py \
  --input ../myo_data/dance_output/ballet_01/ballet_01_00.h5 \
  --output retargeted_ballet.h5
```

## 🔧 Environment Configuration

### Conda Environment Management
```bash
# Activate environment (do this every session)
conda activate myodolores

# Save environment to file
conda env export > environment.yml

# Recreate environment from file (on new machine)
conda env create -f environment.yml

# Update environment
conda env update -f environment.yml

# Remove environment (if needed)
conda env remove -n myodolores
```

### Environment Variables (Optional)
```bash
# Add to ~/.bashrc or ~/.zshrc
export MYODOLORES_ROOT="/path/to/MyoDolores"
export MUJOCO_GL="egl"  # For headless rendering
export CUDA_VISIBLE_DEVICES="0"  # Specify GPU
export AWS_PROFILE="myodolores"  # Use specific AWS profile

# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc
```

## 📊 Data Management

### Dataset Information
| Dataset | Size | Files | Domain | Description |
|---------|------|-------|---------|-------------|
| **HAA500_output** | 57GB | 3,000+ | Human Actions | Walking, running, sports, daily activities |
| **game_motion_output** | 87GB | 8,000+ | Gaming | Combat, exploration, interactive motions |
| **aist_output** | 21GB | 1,500+ | Dance | AIST choreographed dance sequences |
| **kungfu_output** | 15GB | 1,200+ | Martial Arts | Kung fu, Tai chi, combat techniques |
| **humman_output** | 11GB | 800+ | Biomechanics | Scientific motion analysis data |
| **perform_output** | 6.9GB | 600+ | Performance | Theatrical and stage performance |
| **animation_output** | 4.7GB | 1,000+ | Animation | Stylized animation motions |
| **dance_output** | 2.4GB | 400+ | Dance | Extended dance collection |

### Data Sync Options
```bash
# Sync all datasets (recommended)
./scripts/sync_data.sh --all

# Sync by priority (for limited storage)
./scripts/sync_data.sh HAA500_output    # Essential human motions
./scripts/sync_data.sh aist_output      # High-quality dance  
./scripts/sync_data.sh dance_output     # Additional dance

# Force re-download (if data corrupted)
./scripts/sync_data.sh --force HAA500_output

# Use specific AWS profile
./scripts/sync_data.sh --profile myodolores --all
```

## 🎮 Training Workflows

### 1. Motion Tracking Training
```bash
# Basic locomotion tracking
python ASAP/humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  robot.motion.motion_file="../myo_data/HAA500_output/Walking/Walking_Normal_01.h5"

# Dance motion tracking  
python ASAP/humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  robot.motion.motion_file="../myo_data/aist_output/dance_ballet/ballet_01.h5"
```

### 2. Locomotion Training
```bash
python ASAP/humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=locomotion \
  +robot=g1/g1_29dof_anneal_23dof \
  num_envs=4096
```

### 3. Unitree Robot Training
```bash
cd unitree_rl_gym/legged_gym/scripts
python train.py --task=g1 --headless=True --num_envs=4096
```

## 🐛 Common Issues and Solutions

### Issue: Conda environment not activating
```bash
# Solution: Initialize conda in shell
conda init bash  # or conda init zsh
source ~/.bashrc  # or source ~/.zshrc
conda activate myodolores
```

### Issue: AWS credentials not working
```bash
# Solution: Reconfigure AWS
aws configure list  # Check current config
aws sts get-caller-identity  # Test credentials
aws configure  # Reconfigure if needed
```

### Issue: MuJoCo rendering issues
```bash
# Solution: Set environment variable
export MUJOCO_GL=egl  # For headless
export MUJOCO_GL=glfw  # For display
```

### Issue: Out of memory during training
```bash
# Solution: Reduce batch size
# In training script, modify:
num_envs=2048  # Instead of 4096
```

### Issue: Data sync fails
```bash
# Solution: Check connectivity and retry
aws s3 ls s3://myo-data/  # Test S3 access
./scripts/sync_data.sh --dry-run HAA500_output  # Preview
./scripts/sync_data.sh HAA500_output  # AWS will resume automatically
```

## 📝 Verification Checklist

After setup, verify these items work:

- [ ] **Environment**: `conda activate myodolores` succeeds
- [ ] **Python**: `python -c "import mujoco, numpy, h5py"` succeeds  
- [ ] **AWS**: `aws s3 ls s3://myo-data/` lists datasets
- [ ] **Data**: `find myo_data/ -name "*.h5" | wc -l` shows 28,000+ files
- [ ] **Scripts**: `./scripts/setup_validation.sh` shows mostly green checkmarks
- [ ] **Loading**: Can load and visualize motion files
- [ ] **Training**: Can start a basic training run

## 🆘 Getting Help

1. **Run validation**: `./scripts/setup_validation.sh --verbose`
2. **Check logs**: Review `setup_validation.log` for details
3. **Read documentation**: `CLAUDE.md`, `DATA_SETUP.md`
4. **Common issues**: See `TROUBLESHOOTING.md`
5. **Auto-fix**: Try `./scripts/setup_validation.sh --fix`

## 🔄 Regular Maintenance

### Weekly Tasks
```bash
# Update submodules
git submodule update --remote

# Check for data updates
./scripts/sync_data.sh --list

# Validate setup
./scripts/setup_validation.sh --quick
```

### Before Training Sessions
```bash
# Activate environment
conda activate myodolores

# Check GPU status (if using GPU)
nvidia-smi

# Verify data integrity  
./scripts/setup_validation.sh --data-only
```

---

**🎉 Congratulations!** Your MyoDolores setup is complete. You now have access to 28,000+ motion sequences and a comprehensive biomechanics platform. Start with the examples above or dive into the specific framework documentation in `CLAUDE.md`.

For the most up-to-date information and advanced configurations, always refer to the project documentation and run the validation scripts.