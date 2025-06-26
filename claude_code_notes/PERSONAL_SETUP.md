# MyoDolores Personal Computer Setup

🎮 **Complete guide to set up keyboard-controlled humanoid on your personal computer**

## 🚀 Quick Start (Recommended)

```bash
# 1. Clone the repository
git clone <your-repo-url> MyoDolores
cd MyoDolores

# 2. Run automated setup
chmod +x setup_myodolores.sh
./setup_myodolores.sh

# 3. Start the demo
./quick_start.sh
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows WSL2
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 5GB free space
- **CPU**: Any modern CPU (4+ cores recommended)

### Recommended for Fast Training
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **RAM**: 8GB+
- **CPU**: 8+ cores

### Software Dependencies (Auto-installed)
- Python 3.8+
- Conda/Miniconda
- PyTorch
- MuJoCo 3.2.4
- NumPy, SciPy, Matplotlib

## 🛠️ Manual Installation

If the automated script doesn't work, follow these manual steps:

### Step 1: Install Conda

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**macOS:**
```bash
curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash miniconda.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Create Environment

```bash
conda create -n myodolores python=3.8 -y
conda activate myodolores
```

### Step 3: Install Dependencies

```bash
# Core packages
pip install numpy scipy matplotlib h5py pyyaml Pillow

# MuJoCo
pip install mujoco==3.2.4

# PyTorch (choose one)
# For GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Demo dependencies
pip install termcolor
```

### Step 4: Install Project

```bash
cd MyoDolores
cd myo_api && pip install -e . && cd ..
```

## 🎮 Usage Guide

### Training the Policy

```bash
# Activate environment
conda activate myodolores

# Train the keyboard control policy (10-30 minutes)
python train_keyboard_control.py
```

**Training Output:**
```
Training on device: cuda
GPU: NVIDIA GeForce RTX 3080
Environment: 256 envs, 133 actions, 278 observations
Starting training: 1000 iterations
Iteration    0: Avg Reward =    1.069
Iteration   50: Avg Reward =    0.793
...
Training complete! Saved final model as 'keyboard_policy_final.pt'
```

### Testing the Policy

```bash
# Test the trained policy (headless)
python test_keyboard_demo.py
```

**Test Output:**
```
MyoDolores Keyboard Control Test
========================================
Testing: Forward [1.0, 0.0, 0.0]
  Step   0: Height=1.70m, Velocity=[0.00 0.00]
  Step  25: Height=1.69m, Velocity=[0.15 -0.21]
✅ Test completed!
The trained policy is working!
```

### Interactive Demo

```bash
# Run interactive demo (needs display)
python keyboard_demo.py
```

**Controls:**
- **↑** Forward
- **↓** Backward  
- **←** Turn left
- **→** Turn right
- **Space** Stop
- **Esc** Quit

## 📁 Project Structure

```
MyoDolores/
├── train_keyboard_control.py      # RL training script
├── keyboard_demo.py               # Interactive demo
├── test_keyboard_demo.py          # Headless testing
├── keyboard_policy_final.pt       # Trained policy (created after training)
├── setup_myodolores.sh           # Automated setup script
├── activate_myodolores.sh         # Environment activation
├── quick_start.sh                 # Quick start demo
├── myo_model_internal/            # MyoSkeleton humanoid model
├── myo_api/                       # MuJoCo API
└── logs/                          # Training logs
```

## ⚙️ Configuration Options

### Training Parameters

Edit `train_keyboard_control.py`:

```python
# Training configuration
num_envs = 256        # Number of parallel environments (reduce for less memory)
learning_rate = 3e-4  # Learning rate
num_iterations = 1000 # Training iterations (increase for better policy)
```

### Hardware-Specific Settings

**For CPU-only training:**
```python
device = 'cpu'
num_envs = 64  # Reduce for CPU
```

**For limited GPU memory:**
```python
num_envs = 128  # Reduce batch size
```

**For high-end GPUs:**
```python
num_envs = 512  # Increase for faster training
```

## 🐛 Troubleshooting

### Common Issues

**1. MuJoCo rendering errors:**
```bash
export MUJOCO_GL=osmesa  # CPU rendering
# or
export MUJOCO_GL=egl     # Headless GPU rendering
```

**2. CUDA out of memory:**
```python
# Reduce num_envs in train_keyboard_control.py
num_envs = 128  # or even 64
```

**3. Import errors:**
```bash
# Make sure environment is activated
conda activate myodolores

# Check Python path
export PYTHONPATH="$PWD/myo_api:$PYTHONPATH"
```

**4. Model file not found:**
```bash
# Verify the model exists
ls myo_model_internal/myo_model/myoskeleton/myoskeleton_with_motors.xml

# If missing, check git submodules
git submodule update --init --recursive
```

### Performance Issues

**Slow training:**
- Use GPU if available
- Reduce `num_envs` 
- Close other applications

**Robot falls over quickly:**
- Train longer (increase `num_iterations`)
- Adjust reward weights in training script
- Use smaller velocity commands for testing

### Environment Issues

**Conda not found:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc  # or source ~/.zshrc
```

**Wrong Python version:**
```bash
# Create new environment with correct Python
conda create -n myodolores python=3.8 -y
conda activate myodolores
```

## 📊 Expected Performance

### Training Time
- **GPU (RTX 3080)**: 10-15 minutes
- **GPU (GTX 1060)**: 20-30 minutes  
- **CPU (8 cores)**: 45-60 minutes
- **CPU (4 cores)**: 60-90 minutes

### Policy Performance
- **Forward walking**: Good stability for 2-3 seconds
- **Turning**: Partial success, may fall during sharp turns
- **Balance**: Maintains upright posture initially
- **Responsiveness**: Real-time 50Hz control

### Hardware Usage
- **GPU Memory**: 2-4GB during training
- **RAM**: 4-8GB during training
- **Storage**: ~1MB for trained policy
- **CPU**: High usage during training, moderate during demo

## 🔧 Advanced Configuration

### Custom Reward Functions

Edit the reward calculation in `train_keyboard_control.py`:

```python
def _calculate_rewards(self, commands):
    # Modify these weights
    self.reward_scales = {
        'velocity_tracking': 2.0,    # Increase for better command following
        'upright': 1.0,              # Increase for better balance
        'joint_limits': -0.1,        # Increase penalty for joint limits
        'energy': -0.001             # Increase for more efficient movement
    }
```

### Training on Different Hardware

**High-end GPU (RTX 4090):**
```python
num_envs = 1024
num_iterations = 2000
```

**Laptop/Integrated Graphics:**
```python
device = 'cpu'
num_envs = 32
num_iterations = 500
```

## 📈 Next Steps

Once you have the basic demo working:

1. **Improve Training**: Increase iterations, tune rewards
2. **Add Features**: Stairs, obstacles, different terrains
3. **Better Control**: Add joystick/gamepad support
4. **Advanced Motions**: Running, jumping, dancing
5. **Real Robot**: Deploy to physical humanoid robot

## 🤝 Contributing

Found a bug or want to improve the setup?

1. Fork the repository
2. Make your changes
3. Test on your system
4. Submit a pull request

## 📄 License

This project is licensed under MIT License. See LICENSE file for details.

---

**🎉 Enjoy your keyboard-controlled humanoid robot!** 

For questions or issues, please check the troubleshooting section or create an issue in the repository.