#!/bin/bash
# MyoDolores Keyboard Control Setup Script
# Automated installation for personal computers

set -e  # Exit on any error

echo "========================================="
echo "MyoDolores Keyboard Control Setup"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOGFILE="myodolores_setup.log"
exec > >(tee -a "$LOGFILE")
exec 2>&1

echo -e "${BLUE}[INFO]${NC} Setup started at $(date)"
echo -e "${BLUE}[INFO]${NC} Log file: $LOGFILE"

# Check system requirements
echo -e "\n${BLUE}[STEP 1/8]${NC} Checking system requirements..."

# Check OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo -e "${GREEN}✓${NC} Linux detected"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    echo -e "${GREEN}✓${NC} macOS detected"
else
    echo -e "${RED}✗${NC} Unsupported OS: $OSTYPE"
    echo "This script supports Linux and macOS only"
    exit 1
fi

# Check for required commands
REQUIRED_COMMANDS=("git" "python3" "pip3")
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if command -v $cmd &> /dev/null; then
        echo -e "${GREEN}✓${NC} $cmd found"
    else
        echo -e "${RED}✗${NC} $cmd not found"
        echo "Please install $cmd and try again"
        exit 1
    fi
done

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION (>=3.8 required)"
else
    echo -e "${RED}✗${NC} Python $PYTHON_VERSION is too old (3.8+ required)"
    exit 1
fi

# Check for conda
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✓${NC} Conda found"
    HAS_CONDA=true
else
    echo -e "${YELLOW}!${NC} Conda not found - will install miniconda"
    HAS_CONDA=false
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}✓${NC} GPU detected: $GPU_INFO"
    HAS_GPU=true
else
    echo -e "${YELLOW}!${NC} No NVIDIA GPU detected - will use CPU training (slower)"
    HAS_GPU=false
fi

# Ask user confirmation
echo -e "\n${BLUE}[SETUP SUMMARY]${NC}"
echo "  OS: $OS"
echo "  Python: $PYTHON_VERSION"
echo "  Conda: $HAS_CONDA"
echo "  GPU: $HAS_GPU"
echo ""
read -p "Continue with installation? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled"
    exit 0
fi

# Install conda if needed
if [ "$HAS_CONDA" = false ]; then
    echo -e "\n${BLUE}[STEP 2/8]${NC} Installing Miniconda..."
    
    if [ "$OS" = "linux" ]; then
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    else
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi
    
    curl -o miniconda.sh $CONDA_URL
    bash miniconda.sh -b -p $HOME/miniconda3
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    echo -e "${GREEN}✓${NC} Miniconda installed"
else
    echo -e "\n${BLUE}[STEP 2/8]${NC} Conda already installed - skipping"
fi

# Create conda environment
echo -e "\n${BLUE}[STEP 3/8]${NC} Creating conda environment..."

if conda env list | grep -q "myodolores"; then
    echo -e "${YELLOW}!${NC} Environment 'myodolores' already exists"
    read -p "Remove and recreate? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n myodolores -y
    else
        echo "Using existing environment"
    fi
fi

if ! conda env list | grep -q "myodolores"; then
    conda create -n myodolores python=3.8 -y
    echo -e "${GREEN}✓${NC} Environment created"
fi

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate myodolores

# Install dependencies
echo -e "\n${BLUE}[STEP 4/8]${NC} Installing Python dependencies..."

# Core packages
pip install numpy scipy matplotlib h5py pyyaml Pillow

# MuJoCo
pip install mujoco==3.2.4

# PyTorch (CPU/GPU)
if [ "$HAS_GPU" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Additional packages for demo
pip install termcolor
if [ "$OS" = "linux" ]; then
    pip install keyboard || echo "Note: keyboard package optional for demo"
fi

echo -e "${GREEN}✓${NC} Dependencies installed"

# Clone/setup repository if needed
echo -e "\n${BLUE}[STEP 5/8]${NC} Setting up repository..."

REPO_DIR="MyoDolores"
if [ ! -d "$REPO_DIR" ]; then
    echo "Repository not found - please ensure MyoDolores directory exists"
    echo "This script should be run from within the MyoDolores repository"
    exit 1
fi

cd "$REPO_DIR"

# Install project packages
echo "Installing project packages..."
if [ -d "myo_api" ]; then
    cd myo_api && pip install -e . && cd ..
    echo -e "${GREEN}✓${NC} myo_api installed"
fi

# Download Isaac Gym (optional)
echo -e "\n${BLUE}[STEP 6/8]${NC} Isaac Gym setup..."
if [ ! -d "IsaacGymEnvs" ]; then
    echo "Downloading Isaac Gym environments..."
    git clone https://github.com/isaac-sim/IsaacGymEnvs.git
    cd IsaacGymEnvs && pip install -e . && cd ..
    
    # Try to download Isaac Gym itself
    echo "Attempting to download Isaac Gym..."
    wget -O isaac_gym.tar.gz "https://developer.nvidia.com/isaac-gym-preview-4" || echo "Isaac Gym download failed - manual installation required"
    
    if [ -f "isaac_gym.tar.gz" ]; then
        tar -xzf isaac_gym.tar.gz
        if [ -d "isaacgym/python" ]; then
            cd isaacgym/python && pip install -e . && cd ../..
            echo -e "${GREEN}✓${NC} Isaac Gym installed"
        fi
    fi
else
    echo -e "${GREEN}✓${NC} Isaac Gym already present"
fi

# Verify model files
echo -e "\n${BLUE}[STEP 7/8]${NC} Verifying model files..."

MODEL_FILE="myo_model_internal/myo_model/myoskeleton/myoskeleton_with_motors.xml"
if [ -f "$MODEL_FILE" ]; then
    echo -e "${GREEN}✓${NC} MyoSkeleton model found"
else
    echo -e "${RED}✗${NC} MyoSkeleton model not found at $MODEL_FILE"
    echo "Please ensure the myo_model_internal submodule is properly initialized"
fi

# Create activation script
echo -e "\n${BLUE}[STEP 8/8]${NC} Creating activation script..."

cat > activate_myodolores.sh << 'EOF'
#!/bin/bash
# MyoDolores Environment Activation Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Activating MyoDolores environment...${NC}"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate myodolores

# Set environment variables
export MUJOCO_GL=egl  # For headless rendering
export PYTHONPATH="$PWD/myo_api:$PYTHONPATH"

echo -e "${GREEN}✓ Environment activated${NC}"
echo ""
echo "Available commands:"
echo "  python train_keyboard_control.py  - Train the RL policy"
echo "  python test_keyboard_demo.py      - Test trained policy"
echo "  python keyboard_demo.py           - Interactive demo"
echo ""
echo "To deactivate: conda deactivate"
EOF

chmod +x activate_myodolores.sh

# Create quick start script
cat > quick_start.sh << 'EOF'
#!/bin/bash
# MyoDolores Quick Start Script

echo "MyoDolores Keyboard Control - Quick Start"
echo "========================================"

# Check if policy exists
if [ ! -f "keyboard_policy_final.pt" ]; then
    echo "No trained policy found. Starting training..."
    echo "This will take 10-30 minutes depending on your hardware."
    echo ""
    read -p "Continue with training? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        source activate_myodolores.sh
        python train_keyboard_control.py
    else
        echo "Training cancelled. Run 'python train_keyboard_control.py' manually when ready."
        exit 0
    fi
fi

echo ""
echo "Testing trained policy..."
source activate_myodolores.sh
python test_keyboard_demo.py

echo ""
echo "To run interactive demo: python keyboard_demo.py"
EOF

chmod +x quick_start.sh

# Final summary
echo -e "\n${GREEN}========================================="
echo -e "🎉 MyoDolores Setup Complete!"
echo -e "=========================================${NC}"

echo ""
echo "📁 Files created:"
echo "  • activate_myodolores.sh  - Environment activation"
echo "  • quick_start.sh          - Quick start demo"
echo "  • $LOGFILE         - Setup log"

echo ""
echo "🚀 Next steps:"
echo "  1. Run: ./quick_start.sh"
echo "  2. Or manually:"
echo "     source activate_myodolores.sh"
echo "     python train_keyboard_control.py    # Train policy (if needed)"
echo "     python test_keyboard_demo.py        # Test policy"
echo "     python keyboard_demo.py             # Interactive demo"

echo ""
echo "📋 Requirements for training:"
if [ "$HAS_GPU" = true ]; then
    echo "  • GPU training: 10-15 minutes"
else
    echo "  • CPU training: 30-60 minutes (slower but works)"
fi
echo "  • Memory: 4GB+ RAM recommended"
echo "  • Storage: Training uses ~1MB for policy file"

echo ""
echo "🔧 Troubleshooting:"
echo "  • Check setup log: $LOGFILE"
echo "  • GPU issues: Set MUJOCO_GL=osmesa for CPU rendering"
echo "  • Import errors: Ensure 'conda activate myodolores' was run"

echo ""
echo -e "${BLUE}Setup completed successfully at $(date)${NC}"