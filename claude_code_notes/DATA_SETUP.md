# MyoData Setup Guide

This guide explains how to set up and sync the motion capture datasets for the MyoDolores project.

## Overview

The MyoData collection contains 8,000+ motion sequences across 8 different categories:

- **animation_output** - Animation and computer graphics motions
- **aist_output** - AIST dance dataset motions  
- **dance_output** - Dance and choreography sequences
- **game_motion_output** - Video game character motions
- **HAA500_output** - HAA500 dataset with 500 human actions
- **humman_output** - Human motion analysis sequences
- **kungfu_output** - Kung fu and martial arts motions
- **perform_output** - Performance and theatrical motions

## Directory Structure

```
MyoDolores/
├── myo_data/                    # ← Local data directory (git ignored)
│   ├── animation_output/
│   ├── aist_output/
│   ├── dance_output/
│   ├── game_motion_output/
│   ├── HAA500_output/
│   ├── humman_output/
│   ├── kungfu_output/
│   └── perform_output/
├── scripts/
│   └── sync_data.sh            # ← Data sync script
└── DATA_SETUP.md               # ← This file
```

## Prerequisites

### 1. AWS CLI Installation

```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version
```

### 2. AWS Credentials Configuration

Configure your AWS credentials with access to the `myo-data` S3 bucket:

```bash
# Configure default profile
aws configure

# Or configure a specific profile
aws configure --profile myodolores
```

Required information:
- **Access Key ID**: Your AWS access key
- **Secret Access Key**: Your AWS secret key  
- **Default region**: `us-west-2` (recommended)
- **Output format**: `json`

### 3. Verify S3 Access

Test access to the S3 bucket:

```bash
# List bucket contents
aws s3 ls s3://myo-data/

# Check specific dataset
aws s3 ls s3://myo-data/HAA500_output/ --recursive | head -10
```

## Data Sync Usage

### Quick Start

```bash
# Make script executable (if not already)
chmod +x scripts/sync_data.sh

# Sync all datasets
./scripts/sync_data.sh --all

# Sync specific dataset
./scripts/sync_data.sh HAA500_output
```

### Sync Options

| Option | Description | Example |
|--------|-------------|---------|
| `--all` | Sync all datasets | `./scripts/sync_data.sh --all` |
| `--list` | List available datasets | `./scripts/sync_data.sh --list` |
| `--dry-run` | Preview sync without downloading | `./scripts/sync_data.sh --dry-run --all` |
| `--force` | Force re-download existing files | `./scripts/sync_data.sh --force HAA500_output` |
| `--profile` | Use specific AWS profile | `./scripts/sync_data.sh --profile myodolores --all` |

### Examples

```bash
# Preview what would be downloaded for HAA500 dataset
./scripts/sync_data.sh --dry-run HAA500_output

# Sync dance dataset with custom AWS profile
./scripts/sync_data.sh --profile myodolores dance_output

# Force re-sync all datasets (overwrites local files)
./scripts/sync_data.sh --force --all

# List all available datasets and their status
./scripts/sync_data.sh --list
```

## Dataset Information

### File Formats

All datasets contain H5 files with the following structure:

```python
# H5 file structure (example)
motion_file.h5:
├── time          # Time stamps for each frame
├── qpos          # Joint positions (quaternions + joint angles)
├── qvel          # Joint velocities
├── ctrl          # Control signals/torques
└── markers       # 3D marker positions (if available)
```

### Dataset Sizes (Approximate)

| Dataset | Size | Files | Description |
|---------|------|-------|-------------|
| HAA500_output | ~2GB | 500+ | Human action analysis dataset |
| aist_output | ~1.5GB | 300+ | AIST dance dataset |
| dance_output | ~3GB | 800+ | Dance and choreography |
| animation_output | ~2.5GB | 600+ | Animation motions |
| game_motion_output | ~1.8GB | 400+ | Video game character motions |
| perform_output | ~2.2GB | 500+ | Performance and theatrical |
| kungfu_output | ~1.2GB | 300+ | Martial arts motions |
| humman_output | ~1.8GB | 400+ | Human motion analysis |

**Total: ~16GB, 4,000+ files**

## Usage in Training

### ASAP/HumanoidVerse Training

```bash
# Train with HAA500 badminton motion
python ASAP/humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
robot.motion.motion_file="../myo_data/HAA500_output/Badminton_Underswing/Badminton_Underswing_00.h5" \
num_envs=4096 \
project_name=MotionTracking \
experiment_name=MyoData_Badminton

# Train with AIST dance motion
python ASAP/humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
robot.motion.motion_file="../myo_data/aist_output/dance_001/dance_001_00.h5" \
num_envs=4096
```

### Myo API Loading

```python
import myo_api as myo

# Load motion file
trajectory = myo.mjTrajectory.load("myo_data/HAA500_output/Walking/Walking_01.h5")

# Access motion data
qpos = trajectory.qpos  # Joint positions over time
qvel = trajectory.qvel  # Joint velocities over time
time = trajectory.time  # Time stamps
```

## Troubleshooting

### Common Issues

1. **AWS credentials not configured**
   ```bash
   # Error: Unable to locate credentials
   aws configure --profile myodolores
   ```

2. **Permission denied on S3 bucket**
   ```bash
   # Check your AWS permissions
   aws s3 ls s3://myo-data/ --profile myodolores
   ```

3. **Disk space issues**
   ```bash
   # Check available space (need ~20GB free)
   df -h .
   
   # Sync individual datasets instead of all
   ./scripts/sync_data.sh HAA500_output
   ```

4. **Network interruption during sync**
   ```bash
   # Resume interrupted sync
   ./scripts/sync_data.sh HAA500_output
   # (AWS S3 sync automatically resumes from where it left off)
   ```

### Verification

After syncing, verify your data:

```bash
# Check dataset sizes
du -sh myo_data/*/

# Count files in each dataset
find myo_data/ -name "*.h5" | wc -l

# Test loading a motion file
python -c "
import myo_api as myo
traj = myo.mjTrajectory.load('myo_data/HAA500_output/Walking/Walking_01.h5')
print(f'Motion duration: {traj.time[-1]:.2f}s')
print(f'Frames: {len(traj.time)}')
"
```

## Next Steps

After setting up the data:

1. **Training**: Use the motion files for RL training with ASAP or Unitree RL Gym
2. **Analysis**: Load motions with myo_api for biomechanical analysis
3. **Retargeting**: Use myo_retarget to adapt motions for different models
4. **Demo**: Create keyboard-controlled demos using the motion data

For training examples, see `CLAUDE.md` and the respective framework documentation.