# MyoAvatar: Comprehensive Biomechanics and Robotics Platform

MyoAvatar is an integrated research platform for biomechanics simulation, motion capture processing, and robot control. It combines state-of-the-art humanoid motion tracking, legged robot parkour training, and comprehensive motion retargeting capabilities.

## Repository Structure

```
MyoAvatar/
├── ASAP/                    # Humanoid motion tracking (HumanoidVerse)
├── extreme-parkour/         # Legged robot parkour training
├── myo_api/                 # MuJoCo-based biomechanics API
├── myo_retarget/           # Motion retargeting pipeline
├── myo_model_internal/     # MyoSkeleton universal human model
├── unitree_rl_gym/         # Unitree robot training environment
├── myo_data/               # Training datasets (HAA500, AIST, etc.)
├── CLAUDE.md               # Detailed development documentation
└── README.md               # This file
```

## Submodules

This repository uses git submodules for modular development. Each component can be developed independently while maintaining integration.

### Initialize Submodules
```bash
git submodule update --init --recursive
```

### Update All Submodules
```bash
git submodule update --remote
```

## Quick Start

### 1. Environment Setup

**For ASAP (HumanoidVerse):**
```bash
conda create -n hvgym python=3.8
conda activate hvgym
cd ASAP && pip install -e . && pip install -e isaac_utils/
```

**For Extreme Parkour:**
```bash
conda create -n parkour python=3.8
conda activate parkour
cd extreme-parkour && pip install -e rsl_rl/ && pip install -e legged_gym/
```

**For Myo Components:**
```bash
conda create -n myo python=3.8
conda activate myo
cd myo_api && pip install -e '.[gpu,test]'
cd ../myo_retarget && pip install -e '.[gpu]'
```

### 2. Training Examples

**Humanoid Motion Tracking:**
```bash
cd ASAP
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
robot.motion.motion_file="../myo_data/HAA500_output/Badminton_Underswing/Badminton_Underswing_00.h5"
```

**Legged Robot Parkour:**
```bash
cd extreme-parkour/legged_gym/scripts
python train.py --exptid cr7-parkour --device cuda:0
```

**Unitree Robot Training:**
```bash
cd unitree_rl_gym/legged_gym/scripts
python train.py --task=g1 --headless=True
```

## Key Features

- **Multi-Simulator Support**: Isaac Gym, Isaac Sim, Genesis, MuJoCo
- **Comprehensive Datasets**: 8,000+ motion capture sequences
- **Robot Platforms**: Humanoids (G1, H1), Quadrupeds (Go2, A1)
- **Sim2Real Pipeline**: Simulation to real robot deployment
- **GPU Acceleration**: JAX-based optimization and MuJoCo-MJX

## Integration Pipeline

```
Raw MoCap → myo_retarget → myo_data → ASAP/Unitree Training → Deployed Policies
     ↓           ↓            ↓              ↓                    ↓
  C3D/BVH    Joint Angles   H5 Files    Trained Models     Real Robots
```

## Research Papers

- **ASAP**: "Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills" (RSS 2025)
- **Extreme Parkour**: "Extreme Parkour with Legged Robots" (arXiv:2309.14341)

## License

See individual submodule repositories for specific licenses. Most components use MIT or BSD licenses for research use.

## Citation

If you use MyoAvatar in your research, please cite the relevant papers:

```bibtex
@article{he2025asap,
  title={ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills},
  author={He, Tairan and Gao, Jiawei and others},
  journal={Robotics: Science and Systems (RSS)},
  year={2025}
}

@article{cheng2023parkour,
  title={Extreme Parkour with Legged Robots},
  author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
  journal={arXiv preprint arXiv:2309.14341},
  year={2023}
}
```

## Support

For detailed development guidance and component-specific documentation, see [CLAUDE.md](CLAUDE.md).
