#!/bin/bash

# MyoDolores Environment Setup Script
# Automated environment configuration for the MyoDolores project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV_NAME="myodolores"
PYTHON_VERSION="3.8"
SETUP_LOG="environment_setup.log"

# Functions
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}              MyoDolores Environment Setup${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    echo -e "${CYAN}This script will set up the complete MyoDolores environment${NC}"
    echo -e "${CYAN}including conda environment, dependencies, and configurations.${NC}"
    echo ""
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -f, --force             Force reinstall even if environment exists"
    echo "  -g, --gpu               Install GPU-enabled packages"
    echo "  -c, --cpu-only          Install CPU-only packages (default)"
    echo "  -m, --minimal           Install minimal dependencies only"
    echo "  -d, --dev               Install development dependencies"
    echo "  --python VERSION        Specify Python version (default: 3.8)"
    echo "  --env-name NAME         Specify conda environment name (default: myodolores)"
    echo "  --skip-conda            Skip conda environment creation"
    echo "  --skip-aws              Skip AWS CLI installation"
    echo "  --skip-submodules       Skip submodule initialization"
    echo ""
    echo "Examples:"
    echo "  $0                      # Standard CPU installation"
    echo "  $0 --gpu                # Install with GPU support"
    echo "  $0 --minimal            # Minimal installation"
    echo "  $0 --dev --gpu          # Development installation with GPU"
}

log_message() {
    local level=$1
    local message=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" >> "$SETUP_LOG"
    if [[ "$verbose" == "true" ]]; then
        echo -e "${CYAN}[LOG]${NC} $message"
    fi
}

print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK")
            echo -e "${GREEN}✓${NC} $message"
            log_message "INFO" "SUCCESS: $message"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠${NC} $message"
            log_message "WARN" "WARNING: $message"
            ;;
        "ERROR")
            echo -e "${RED}✗${NC} $message"
            log_message "ERROR" "ERROR: $message"
            ;;
        "INFO")
            echo -e "${CYAN}ℹ${NC} $message"
            log_message "INFO" "$message"
            ;;
        "STEP")
            echo -e "${BLUE}➤${NC} $message"
            log_message "INFO" "STEP: $message"
            ;;
    esac
}

check_requirements() {
    print_status "STEP" "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "OK" "Operating System: Linux"
        OS_TYPE="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "OK" "Operating System: macOS"
        OS_TYPE="macos"
    else
        print_status "WARN" "Operating System: $OSTYPE (may have compatibility issues)"
        OS_TYPE="other"
    fi
    
    # Check internet connectivity
    if ping -c 1 google.com &> /dev/null; then
        print_status "OK" "Internet connectivity verified"
    else
        print_status "ERROR" "No internet connection - setup cannot continue"
        exit 1
    fi
    
    # Check available disk space
    local available_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -ge 20 ]]; then
        print_status "OK" "Disk space: ${available_space}GB available"
    else
        print_status "WARN" "Low disk space: ${available_space}GB (recommend 20GB+)"
    fi
    
    echo ""
}

install_system_dependencies() {
    if [[ "$skip_system" == "true" ]]; then
        print_status "INFO" "Skipping system dependencies installation"
        return
    fi
    
    print_status "STEP" "Installing system dependencies..."
    
    if [[ "$OS_TYPE" == "linux" ]]; then
        # Check if we can use sudo
        if sudo -n true 2>/dev/null; then
            print_status "INFO" "Installing Linux packages with sudo"
            sudo apt update
            sudo apt install -y curl wget build-essential git
            print_status "OK" "Linux system packages installed"
        else
            print_status "WARN" "Cannot use sudo - skipping system package installation"
            print_status "INFO" "Please ensure these packages are installed: curl wget build-essential git"
        fi
    elif [[ "$OS_TYPE" == "macos" ]]; then
        # Check if Homebrew is installed
        if command -v brew &> /dev/null; then
            print_status "INFO" "Using Homebrew to install packages"
            brew install git curl wget
            print_status "OK" "macOS packages installed via Homebrew"
        else
            print_status "WARN" "Homebrew not found - please install system dependencies manually"
            print_status "INFO" "Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fi
    fi
    
    echo ""
}

install_conda() {
    if [[ "$skip_conda" == "true" ]]; then
        print_status "INFO" "Skipping conda installation"
        return
    fi
    
    print_status "STEP" "Setting up conda environment..."
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_status "INFO" "Conda not found - installing Miniconda..."
        
        local conda_installer
        if [[ "$OS_TYPE" == "linux" ]]; then
            conda_installer="Miniconda3-latest-Linux-x86_64.sh"
        elif [[ "$OS_TYPE" == "macos" ]]; then
            if [[ $(uname -m) == "arm64" ]]; then
                conda_installer="Miniconda3-latest-MacOSX-arm64.sh"
            else
                conda_installer="Miniconda3-latest-MacOSX-x86_64.sh"
            fi
        else
            print_status "ERROR" "Unsupported OS for automatic conda installation"
            exit 1
        fi
        
        # Download and install conda
        wget "https://repo.anaconda.com/miniconda/$conda_installer" -O miniconda_installer.sh
        bash miniconda_installer.sh -b -p "$HOME/miniconda3"
        rm miniconda_installer.sh
        
        # Initialize conda
        "$HOME/miniconda3/bin/conda" init bash
        if [[ "$SHELL" == *"zsh"* ]]; then
            "$HOME/miniconda3/bin/conda" init zsh
        fi
        
        # Add to PATH for current session
        export PATH="$HOME/miniconda3/bin:$PATH"
        
        print_status "OK" "Miniconda installed successfully"
        print_status "INFO" "Please restart your shell or run: source ~/.bashrc"
    else
        print_status "OK" "Conda found: $(conda --version)"
    fi
    
    # Ensure conda is available
    if ! command -v conda &> /dev/null; then
        # Try to source conda
        if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        fi
    fi
    
    if ! command -v conda &> /dev/null; then
        print_status "ERROR" "Conda installation failed or not in PATH"
        exit 1
    fi
    
    echo ""
}

create_conda_environment() {
    if [[ "$skip_conda" == "true" ]]; then
        print_status "INFO" "Skipping conda environment creation"
        return
    fi
    
    print_status "STEP" "Creating conda environment '$CONDA_ENV_NAME'..."
    
    # Check if environment exists
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        if [[ "$force" == "true" ]]; then
            print_status "INFO" "Removing existing environment"
            conda env remove -n "$CONDA_ENV_NAME" -y
        else
            print_status "WARN" "Environment '$CONDA_ENV_NAME' already exists"
            print_status "INFO" "Use --force to recreate or --env-name to use different name"
            return
        fi
    fi
    
    # Create environment
    print_status "INFO" "Creating environment with Python $PYTHON_VERSION"
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    print_status "OK" "Conda environment '$CONDA_ENV_NAME' created"
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    print_status "OK" "Environment activated"
    
    echo ""
}

install_python_packages() {
    print_status "STEP" "Installing Python packages..."
    
    # Ensure we're in the right environment
    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]] && [[ "$skip_conda" != "true" ]]; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"
    fi
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    print_status "OK" "Updated pip, setuptools, wheel"
    
    # Essential packages
    print_status "INFO" "Installing essential packages..."
    pip install numpy scipy matplotlib pandas h5py pyyaml opencv-python pillow
    print_status "OK" "Essential packages installed"
    
    # MuJoCo (specific version for compatibility)
    print_status "INFO" "Installing MuJoCo..."
    pip install mujoco==3.2.4
    print_status "OK" "MuJoCo 3.2.4 installed"
    
    # Machine learning frameworks
    if [[ "$gpu" == "true" ]]; then
        print_status "INFO" "Installing GPU-enabled packages..."
        
        # PyTorch with CUDA
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        print_status "OK" "PyTorch with CUDA installed"
        
        # JAX with CUDA
        pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        print_status "OK" "JAX with CUDA installed"
        
    else
        print_status "INFO" "Installing CPU-only packages..."
        
        # PyTorch CPU
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        print_status "OK" "PyTorch (CPU) installed"
        
        # JAX CPU
        pip install jax[cpu]
        print_status "OK" "JAX (CPU) installed"
    fi
    
    # Additional ML packages
    if [[ "$minimal" != "true" ]]; then
        print_status "INFO" "Installing additional ML packages..."
        pip install scikit-learn tensorboard wandb
        print_status "OK" "Additional ML packages installed"
    fi
    
    # Development packages
    if [[ "$dev" == "true" ]]; then
        print_status "INFO" "Installing development packages..."
        pip install pytest black flake8 mypy jupyter ipython
        print_status "OK" "Development packages installed"
    fi
    
    echo ""
}

install_aws_cli() {
    if [[ "$skip_aws" == "true" ]]; then
        print_status "INFO" "Skipping AWS CLI installation"
        return
    fi
    
    print_status "STEP" "Installing AWS CLI..."
    
    if command -v aws &> /dev/null; then
        local aws_version=$(aws --version 2>&1 | cut -d' ' -f1 | cut -d'/' -f2)
        print_status "OK" "AWS CLI already installed: $aws_version"
        return
    fi
    
    if [[ "$OS_TYPE" == "linux" ]]; then
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        
        # Try with sudo first, fallback to local install
        if sudo ./aws/install 2>/dev/null; then
            print_status "OK" "AWS CLI installed system-wide"
        else
            print_status "INFO" "Installing AWS CLI to user directory"
            ./aws/install -i ~/.local/aws-cli -b ~/.local/bin
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        fi
        
        rm -rf aws awscliv2.zip
        
    elif [[ "$OS_TYPE" == "macos" ]]; then
        curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
        
        if sudo installer -pkg AWSCLIV2.pkg -target / 2>/dev/null; then
            print_status "OK" "AWS CLI installed system-wide"
        else
            print_status "WARN" "Could not install AWS CLI system-wide"
            print_status "INFO" "Please install manually: https://aws.amazon.com/cli/"
        fi
        
        rm AWSCLIV2.pkg
    fi
    
    # Verify installation
    if command -v aws &> /dev/null; then
        print_status "OK" "AWS CLI installation verified"
    else
        print_status "WARN" "AWS CLI not found in PATH - may need shell restart"
    fi
    
    echo ""
}

setup_project_structure() {
    print_status "STEP" "Setting up project structure..."
    
    cd "$PROJECT_ROOT"
    
    # Initialize git submodules
    if [[ "$skip_submodules" != "true" ]] && [[ -f ".gitmodules" ]]; then
        print_status "INFO" "Initializing git submodules (this may take several minutes)..."
        git submodule update --init --recursive
        print_status "OK" "Git submodules initialized"
    fi
    
    # Create myo_data directory structure
    mkdir -p myo_data/{animation_output,aist_output,dance_output,game_motion_output,HAA500_output,humman_output,kungfu_output,perform_output}
    print_status "OK" "Created myo_data directory structure"
    
    # Make scripts executable
    if [[ -f "scripts/sync_data.sh" ]]; then
        chmod +x scripts/sync_data.sh
        print_status "OK" "Made sync_data.sh executable"
    fi
    
    if [[ -f "scripts/setup_validation.sh" ]]; then
        chmod +x scripts/setup_validation.sh
        print_status "OK" "Made setup_validation.sh executable"
    fi
    
    echo ""
}

install_framework_dependencies() {
    if [[ "$minimal" == "true" ]]; then
        print_status "INFO" "Skipping framework dependencies (minimal install)"
        return
    fi
    
    print_status "STEP" "Installing framework dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # ASAP/HumanoidVerse
    if [[ -d "ASAP" ]]; then
        print_status "INFO" "Installing ASAP dependencies..."
        cd ASAP
        pip install -e . --quiet
        if [[ -d "isaac_utils" ]]; then
            pip install -e isaac_utils/ --quiet
        fi
        cd "$PROJECT_ROOT"
        print_status "OK" "ASAP dependencies installed"
    fi
    
    # Myo API
    if [[ -d "myo_api" ]]; then
        print_status "INFO" "Installing myo_api dependencies..."
        cd myo_api
        if [[ "$gpu" == "true" ]]; then
            pip install -e .[gpu,test] --quiet
        else
            pip install -e .[test] --quiet
        fi
        cd "$PROJECT_ROOT"
        print_status "OK" "myo_api dependencies installed"
    fi
    
    # Myo Retarget
    if [[ -d "myo_retarget" ]]; then
        print_status "INFO" "Installing myo_retarget dependencies..."
        cd myo_retarget
        if [[ "$gpu" == "true" ]]; then
            pip install -e .[gpu] --quiet
        else
            pip install -e . --quiet
        fi
        cd "$PROJECT_ROOT"
        print_status "OK" "myo_retarget dependencies installed"
    fi
    
    echo ""
}

create_environment_file() {
    print_status "STEP" "Creating environment configuration..."
    
    # Create environment.yml for easy recreation
    if [[ "$skip_conda" != "true" ]]; then
        conda env export > environment.yml
        print_status "OK" "Environment exported to environment.yml"
    fi
    
    # Create activation script
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# MyoDolores Environment Activation Script

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate myodolores
    echo "✓ Activated myodolores conda environment"
else
    echo "⚠ Conda not found"
fi

# Set environment variables
export MYODOLORES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$MYODOLORES_ROOT:$PYTHONPATH"
export MUJOCO_GL="egl"  # For headless rendering

echo "✓ MyoDolores environment ready"
echo "  Project root: $MYODOLORES_ROOT"
echo "  Python: $(which python)"
echo "  MuJoCo GL: $MUJOCO_GL"
EOF
    
    chmod +x activate_env.sh
    print_status "OK" "Created activation script: activate_env.sh"
    
    echo ""
}

print_completion_summary() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}                    SETUP COMPLETE!${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    
    print_status "OK" "MyoDolores environment setup completed successfully"
    echo ""
    
    echo -e "${CYAN}Environment Details:${NC}"
    echo -e "  • Conda environment: ${YELLOW}$CONDA_ENV_NAME${NC}"
    echo -e "  • Python version: ${YELLOW}$PYTHON_VERSION${NC}"
    echo -e "  • GPU support: ${YELLOW}$([ "$gpu" == "true" ] && echo "Enabled" || echo "Disabled")${NC}"
    echo -e "  • Installation type: ${YELLOW}$([ "$minimal" == "true" ] && echo "Minimal" || echo "Full")${NC}"
    echo ""
    
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "  1. ${YELLOW}Restart your shell or run:${NC} source ~/.bashrc"
    echo -e "  2. ${YELLOW}Activate environment:${NC} conda activate $CONDA_ENV_NAME"
    echo -e "  3. ${YELLOW}Or use activation script:${NC} source activate_env.sh"
    echo -e "  4. ${YELLOW}Validate setup:${NC} ./scripts/setup_validation.sh"
    echo -e "  5. ${YELLOW}Sync data:${NC} ./scripts/sync_data.sh --all"
    echo ""
    
    echo -e "${CYAN}Quick Test:${NC}"
    echo -e "  ${YELLOW}python -c \"import mujoco, numpy, h5py; print('✓ All packages working')\"${NC}"
    echo ""
    
    if [[ "$gpu" == "true" ]]; then
        echo -e "${CYAN}GPU Verification:${NC}"
        echo -e "  ${YELLOW}python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"${NC}"
        echo ""
    fi
    
    echo -e "${CYAN}Documentation:${NC}"
    echo -e "  • Complete guide: ${YELLOW}SETUP_GUIDE.md${NC}"
    echo -e "  • Project instructions: ${YELLOW}CLAUDE.md${NC}"
    echo -e "  • Data setup: ${YELLOW}DATA_SETUP.md${NC}"
    echo ""
    
    print_status "INFO" "Setup log saved to: $SETUP_LOG"
}

# Main execution
main() {
    local verbose=false
    local force=false
    local gpu=false
    local minimal=false
    local dev=false
    local skip_conda=false
    local skip_aws=false
    local skip_submodules=false
    local skip_system=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -f|--force)
                force=true
                shift
                ;;
            -g|--gpu)
                gpu=true
                shift
                ;;
            -c|--cpu-only)
                gpu=false
                shift
                ;;
            -m|--minimal)
                minimal=true
                shift
                ;;
            -d|--dev)
                dev=true
                shift
                ;;
            --python)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --env-name)
                CONDA_ENV_NAME="$2"
                shift 2
                ;;
            --skip-conda)
                skip_conda=true
                shift
                ;;
            --skip-aws)
                skip_aws=true
                shift
                ;;
            --skip-submodules)
                skip_submodules=true
                shift
                ;;
            --skip-system)
                skip_system=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Initialize log
    echo "MyoDolores Environment Setup - $(date)" > "$SETUP_LOG"
    log_message "INFO" "Starting setup with options: verbose=$verbose, force=$force, gpu=$gpu, minimal=$minimal, dev=$dev"
    
    print_header
    
    # Run setup steps
    check_requirements
    install_system_dependencies
    install_conda
    create_conda_environment
    install_python_packages
    install_aws_cli
    setup_project_structure
    install_framework_dependencies
    create_environment_file
    
    print_completion_summary
}

# Run main function with all arguments
main "$@"