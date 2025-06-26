# MyoDolores Troubleshooting Guide

This guide provides solutions to common issues encountered when setting up and using the MyoDolores platform.

## 🚨 Quick Diagnostic Tools

Before diving into specific issues, use these diagnostic tools:

```bash
# Comprehensive system validation
./scripts/setup_validation.sh

# Quick environment check
./scripts/setup_validation.sh --quick

# Data integrity verification
./scripts/verify_data.sh

# Check specific dataset
./scripts/verify_data.sh HAA500_output
```

## 📋 Common Issue Categories

- [Environment Setup Issues](#environment-setup-issues)
- [Data Sync Problems](#data-sync-problems)
- [Training Issues](#training-issues)
- [GPU and CUDA Problems](#gpu-and-cuda-problems)
- [Memory and Performance Issues](#memory-and-performance-issues)
- [Import and Package Errors](#import-and-package-errors)
- [File and Permission Issues](#file-and-permission-issues)

---

## 🔧 Environment Setup Issues

### Issue: Conda environment not activating

**Symptoms:**
```bash
conda activate myodolores
# Error: Could not find conda environment
```

**Solutions:**

1. **Initialize conda in your shell:**
   ```bash
   conda init bash  # or conda init zsh for zsh
   source ~/.bashrc  # or source ~/.zshrc
   conda activate myodolores
   ```

2. **Check if environment exists:**
   ```bash
   conda env list
   # If myodolores not listed, create it:
   conda create -n myodolores python=3.8 -y
   ```

3. **Recreate environment from scratch:**
   ```bash
   ./scripts/setup_environment.sh --force
   ```

### Issue: Python version conflicts

**Symptoms:**
```bash
python --version
# Python 2.7.x (wrong version)
```

**Solutions:**

1. **Use python3 explicitly:**
   ```bash
   python3 --version
   which python3
   # Add alias to ~/.bashrc:
   echo 'alias python=python3' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Check conda environment Python:**
   ```bash
   conda activate myodolores
   which python
   python --version  # Should show Python 3.8+
   ```

### Issue: "conda: command not found"

**Solutions:**

1. **Install Miniconda:**
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b
   echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Use the automated setup:**
   ```bash
   ./scripts/setup_environment.sh
   ```

---

## 📊 Data Sync Problems

### Issue: AWS credentials not configured

**Symptoms:**
```bash
./scripts/sync_data.sh --all
# Error: Unable to locate credentials
```

**Solutions:**

1. **Configure AWS credentials:**
   ```bash
   aws configure
   # Enter your AWS Access Key ID
   # Enter your AWS Secret Access Key
   # Default region: us-west-2
   # Default output format: json
   ```

2. **Test AWS access:**
   ```bash
   aws sts get-caller-identity
   aws s3 ls s3://myo-data/
   ```

3. **Use specific AWS profile:**
   ```bash
   aws configure --profile myodolores
   ./scripts/sync_data.sh --profile myodolores --all
   ```

### Issue: S3 permission denied

**Symptoms:**
```bash
aws s3 ls s3://myo-data/
# An error occurred (AccessDenied) when calling the ListObjectsV2 operation
```

**Solutions:**

1. **Check your AWS account permissions**
2. **Contact your AWS administrator for S3 bucket access**
3. **Verify correct AWS profile:**
   ```bash
   aws configure list
   aws sts get-caller-identity
   ```

### Issue: Data sync interrupted/incomplete

**Symptoms:**
- Partial downloads
- Connection timeouts
- Inconsistent file counts

**Solutions:**

1. **Resume sync (AWS S3 automatically resumes):**
   ```bash
   ./scripts/sync_data.sh --all
   ```

2. **Sync specific dataset:**
   ```bash
   ./scripts/sync_data.sh HAA500_output
   ```

3. **Force complete re-download:**
   ```bash
   ./scripts/sync_data.sh --force HAA500_output
   ```

4. **Check network connectivity:**
   ```bash
   ping google.com
   wget -O /dev/null http://speedtest.tele2.net/100MB.zip
   ```

### Issue: Insufficient disk space

**Symptoms:**
```bash
# Error: No space left on device
df -h  # Shows 100% usage
```

**Solutions:**

1. **Check disk usage:**
   ```bash
   df -h
   du -sh myo_data/*/
   ```

2. **Clean up space:**
   ```bash
   # Remove large temporary files
   sudo apt autoremove
   sudo apt autoclean
   rm -rf ~/.cache/pip
   
   # Remove specific datasets if needed
   rm -rf myo_data/game_motion_output  # Largest dataset (87GB)
   ```

3. **Sync only essential datasets:**
   ```bash
   ./scripts/sync_data.sh HAA500_output
   ./scripts/sync_data.sh aist_output
   ```

---

## 🏋️ Training Issues

### Issue: Isaac Gym not found

**Symptoms:**
```bash
python ASAP/humanoidverse/train_agent.py
# ModuleNotFoundError: No module named 'isaacgym'
```

**Solutions:**

1. **Isaac Gym requires manual installation:**
   - Download from NVIDIA Developer Portal
   - Requires NVIDIA account and license agreement
   - Follow ASAP documentation for Isaac Gym setup

2. **Use alternative simulators:**
   ```bash
   # Use Isaac Sim instead
   python ASAP/humanoidverse/train_agent.py +simulator=isaacsim
   
   # Use Genesis (if available)
   python ASAP/humanoidverse/train_agent.py +simulator=genesis
   ```

### Issue: Training crashes with CUDA out of memory

**Symptoms:**
```bash
# RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce number of environments:**
   ```bash
   python ASAP/humanoidverse/train_agent.py num_envs=2048  # Instead of 4096
   python ASAP/humanoidverse/train_agent.py num_envs=1024  # For lower memory
   ```

2. **Check GPU memory:**
   ```bash
   nvidia-smi
   # Free up GPU memory:
   pkill -f python  # Kill other Python processes using GPU
   ```

3. **Use gradient accumulation:**
   ```bash
   # Modify training config to use smaller batches with accumulation
   ```

### Issue: Motion file not found

**Symptoms:**
```bash
# FileNotFoundError: ../myo_data/HAA500_output/Walking/Walking_01.h5
```

**Solutions:**

1. **Verify file exists:**
   ```bash
   ls -la myo_data/HAA500_output/Walking/
   find myo_data/ -name "*Walking*" -type f
   ```

2. **Use correct file path:**
   ```bash
   # Check available motions
   ./scripts/verify_data.sh --stats
   
   # Use existing file
   python ASAP/humanoidverse/train_agent.py \
     robot.motion.motion_file="../myo_data/HAA500_output/Badminton_Underswing/Badminton_Underswing_00.h5"
   ```

3. **Sync missing data:**
   ```bash
   ./scripts/sync_data.sh HAA500_output
   ```

---

## 🎮 GPU and CUDA Problems

### Issue: CUDA not available

**Symptoms:**
```python
import torch
print(torch.cuda.is_available())  # False
```

**Solutions:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   # If command not found, install driver:
   sudo apt update
   sudo apt install nvidia-driver-525  # Or latest version
   sudo reboot
   ```

2. **Install CUDA toolkit:**
   ```bash
   # Check CUDA version
   nvidia-smi  # Look for CUDA Version
   
   # Install PyTorch with matching CUDA version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
   ```

### Issue: MuJoCo rendering problems

**Symptoms:**
```bash
# Error: Could not initialize OpenGL context
# Black screen or no rendering
```

**Solutions:**

1. **Set rendering backend:**
   ```bash
   export MUJOCO_GL=egl  # For headless/server
   export MUJOCO_GL=glfw # For desktop with display
   export MUJOCO_GL=osmesa # CPU-only rendering
   ```

2. **Install rendering dependencies:**
   ```bash
   # For EGL (headless)
   sudo apt install libegl1-mesa-dev
   
   # For desktop rendering
   sudo apt install freeglut3-dev
   ```

3. **Test rendering:**
   ```bash
   python -c "
   import mujoco
   print('MuJoCo version:', mujoco.__version__)
   # Try to create a simple model
   "
   ```

---

## 💾 Memory and Performance Issues

### Issue: System runs out of memory

**Symptoms:**
```bash
# System becomes unresponsive
# "Killed" messages in terminal
free -h  # Shows very low available memory
```

**Solutions:**

1. **Monitor memory usage:**
   ```bash
   htop
   watch -n 1 'free -h'
   ```

2. **Reduce memory usage:**
   ```bash
   # Smaller training batches
   num_envs=1024  # Instead of 4096
   
   # Close unnecessary applications
   pkill -f chrome
   pkill -f firefox
   ```

3. **Add swap space:**
   ```bash
   # Create 8GB swap file
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   
   # Make permanent
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

### Issue: Slow training performance

**Solutions:**

1. **Check system resources:**
   ```bash
   nvidia-smi  # GPU utilization
   htop        # CPU usage
   iotop       # Disk I/O
   ```

2. **Optimize data loading:**
   ```bash
   # Move data to SSD if using HDD
   # Use data_loader_num_workers parameter in training
   ```

3. **Profile training:**
   ```bash
   # Use PyTorch profiler or Weights & Biases
   # Monitor training metrics in tensorboard
   ```

---

## 📦 Import and Package Errors

### Issue: "No module named 'mujoco'"

**Solutions:**

1. **Install MuJoCo:**
   ```bash
   conda activate myodolores
   pip install mujoco==3.2.4
   ```

2. **Verify installation:**
   ```bash
   python -c "import mujoco; print(mujoco.__version__)"
   ```

### Issue: "No module named 'myo_api'"

**Solutions:**

1. **Install in development mode:**
   ```bash
   cd myo_api
   pip install -e .[gpu,test]
   cd ..
   ```

2. **Add to Python path:**
   ```bash
   export PYTHONPATH="$PWD/myo_api:$PYTHONPATH"
   ```

### Issue: Package version conflicts

**Symptoms:**
```bash
# ERROR: package has incompatible requirements
```

**Solutions:**

1. **Create clean environment:**
   ```bash
   conda deactivate
   conda env remove -n myodolores
   ./scripts/setup_environment.sh --force
   ```

2. **Install packages one by one:**
   ```bash
   pip install numpy
   pip install mujoco==3.2.4
   pip install torch  # etc.
   ```

---

## 📁 File and Permission Issues

### Issue: Permission denied errors

**Symptoms:**
```bash
# bash: ./scripts/sync_data.sh: Permission denied
```

**Solutions:**

1. **Make scripts executable:**
   ```bash
   chmod +x scripts/*.sh
   ```

2. **Check file ownership:**
   ```bash
   ls -la scripts/
   # If needed:
   sudo chown $USER:$USER scripts/*.sh
   ```

### Issue: Git submodule problems

**Symptoms:**
```bash
# fatal: no submodule mapping found
# empty submodule directories
```

**Solutions:**

1. **Initialize submodules:**
   ```bash
   git submodule update --init --recursive
   ```

2. **Reset submodules:**
   ```bash
   git submodule deinit -f .
   git submodule update --init --recursive
   ```

3. **Clone with submodules:**
   ```bash
   git clone --recursive <repo-url>
   ```

---

## 🔍 Advanced Debugging

### Enable Verbose Logging

```bash
# Enable verbose output for all scripts
./scripts/setup_validation.sh --verbose
./scripts/verify_data.sh --verbose
./scripts/sync_data.sh --dry-run  # Preview operations

# Python debugging
export PYTHONPATH="$PWD:$PYTHONPATH"
python -u script.py  # Unbuffered output
python -X dev script.py  # Development mode with extra checks
```

### Check System Information

```bash
# System info
uname -a
lsb_release -a
free -h
df -h

# GPU info
nvidia-smi
lspci | grep -i nvidia

# Python environment
conda info
conda list
pip list

# Network connectivity
ping google.com
traceroute aws.amazon.com
```

### Log Analysis

```bash
# Check setup logs
tail -f setup_validation.log
tail -f environment_setup.log
tail -f data_verification.log

# System logs
sudo journalctl -f
tail -f /var/log/syslog
```

---

## 📞 Getting Additional Help

### Self-Diagnostic Checklist

Before seeking help, run through this checklist:

- [ ] **Environment**: `conda activate myodolores` works
- [ ] **Python**: `python -c "import mujoco, numpy, h5py"` succeeds
- [ ] **AWS**: `aws s3 ls s3://myo-data/` lists datasets
- [ ] **Data**: `find myo_data/ -name "*.h5" | wc -l` shows files
- [ ] **Scripts**: All scripts in `scripts/` are executable
- [ ] **Validation**: `./scripts/setup_validation.sh` mostly green
- [ ] **Verification**: `./scripts/verify_data.sh` passes

### Generate Diagnostic Report

```bash
# Create comprehensive diagnostic report
./scripts/setup_validation.sh --verbose > diagnostic_report.txt 2>&1
./scripts/verify_data.sh --stats >> diagnostic_report.txt 2>&1

# Add system information
echo "=== SYSTEM INFO ===" >> diagnostic_report.txt
uname -a >> diagnostic_report.txt
free -h >> diagnostic_report.txt
df -h >> diagnostic_report.txt
nvidia-smi >> diagnostic_report.txt 2>&1

# Share this file when asking for help
```

### Documentation References

- **Setup Guide**: `SETUP_GUIDE.md` - Complete installation instructions
- **Project Instructions**: `CLAUDE.md` - Framework-specific commands
- **Data Setup**: `DATA_SETUP.md` - Data synchronization details
- **Dataset Info**: `myo_data/DATASETS.md` - Dataset descriptions

### Common Commands Summary

```bash
# Environment management
conda activate myodolores
source activate_env.sh

# Data operations
./scripts/sync_data.sh --all
./scripts/verify_data.sh

# Validation and debugging
./scripts/setup_validation.sh
./scripts/setup_validation.sh --fix

# Training (examples)
cd ASAP
python humanoidverse/train_agent.py +simulator=isaacgym +exp=locomotion
```

---

## 🔄 Recovery Procedures

### Complete Reset and Reinstall

If everything fails, follow this complete reset procedure:

```bash
# 1. Backup any important data/configs
cp environment.yml ~/backup_env.yml
cp -r logs/ ~/backup_logs/

# 2. Remove conda environment
conda deactivate
conda env remove -n myodolores

# 3. Clean project directory
rm -rf myo_data/
git clean -fdx  # BE CAREFUL: removes all untracked files
git reset --hard HEAD

# 4. Reinstall everything
git submodule update --init --recursive
./scripts/setup_environment.sh --force --gpu  # or --cpu-only
./scripts/sync_data.sh --all

# 5. Validate
./scripts/setup_validation.sh
./scripts/verify_data.sh
```

### Partial Recovery

For specific component issues:

```bash
# Just re-sync data
rm -rf myo_data/
./scripts/sync_data.sh --all

# Just reinstall Python packages
conda activate myodolores
pip install --force-reinstall -r requirements.txt

# Just reset git submodules
git submodule deinit -f .
git submodule update --init --recursive
```

Remember: Most issues can be resolved by carefully following the error messages and using the diagnostic tools provided. When in doubt, start with `./scripts/setup_validation.sh --verbose` to identify the specific problem area.