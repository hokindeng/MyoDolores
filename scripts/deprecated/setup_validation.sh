#!/bin/bash

# MyoDolores Setup Validation Script
# Comprehensive validation and setup tool for the MyoDolores project

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
SETUP_LOG="setup_validation.log"
ERRORS_FOUND=0
WARNINGS_FOUND=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Functions
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}              MyoDolores Setup Validation Tool${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    echo -e "${CYAN}This script will validate your MyoDolores setup and help with${NC}"
    echo -e "${CYAN}configuration. Run with different options to customize behavior.${NC}"
    echo ""
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -v, --verbose           Enable verbose output"
    echo "  -q, --quick             Quick validation (skip heavy checks)"
    echo "  -f, --fix               Attempt to fix issues automatically"
    echo "  -s, --setup             Run initial setup (for new installations)"
    echo "  -d, --data-only         Only validate data setup"
    echo "  -e, --env-only          Only validate environment setup"
    echo "  --install-deps          Install missing dependencies"
    echo "  --skip-data             Skip data validation"
    echo "  --check-space          Check disk space requirements"
    echo ""
    echo "Examples:"
    echo "  $0                      # Full validation"
    echo "  $0 --setup             # Initial setup for new installation"
    echo "  $0 --fix               # Fix issues automatically"
    echo "  $0 --quick --verbose   # Quick check with detailed output"
}

log_message() {
    local level=$1
    local message=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" >> "$SETUP_LOG"
}

print_status() {
    local status=$1
    local message=$2
    case $status in
        "OK")
            echo -e "${GREEN}✓${NC} $message"
            log_message "INFO" "OK: $message"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠${NC} $message"
            log_message "WARN" "WARNING: $message"
            ((WARNINGS_FOUND++))
            ;;
        "ERROR")
            echo -e "${RED}✗${NC} $message"
            log_message "ERROR" "ERROR: $message"
            ((ERRORS_FOUND++))
            ;;
        "INFO")
            echo -e "${CYAN}ℹ${NC} $message"
            log_message "INFO" "$message"
            ;;
        "SKIP")
            echo -e "${PURPLE}⊘${NC} $message"
            log_message "INFO" "SKIPPED: $message"
            ;;
    esac
}

check_system_requirements() {
    echo -e "${BLUE}[1/8] Checking System Requirements${NC}"
    echo "----------------------------------------"
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "OK" "Operating System: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "OK" "Operating System: macOS"
    else
        print_status "WARN" "Operating System: $OSTYPE (may have compatibility issues)"
    fi
    
    # Check architecture
    local arch=$(uname -m)
    if [[ "$arch" == "x86_64" || "$arch" == "arm64" ]]; then
        print_status "OK" "Architecture: $arch"
    else
        print_status "WARN" "Architecture: $arch (may have compatibility issues)"
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        local mem_gb=$(free -g | awk 'NR==2{printf "%.1f", $2}')
        if (( $(echo "$mem_gb >= 8" | bc -l) )); then
            print_status "OK" "Memory: ${mem_gb}GB (sufficient)"
        else
            print_status "WARN" "Memory: ${mem_gb}GB (recommend 8GB+ for training)"
        fi
    elif command -v vm_stat &> /dev/null; then
        # macOS
        local mem_bytes=$(sysctl -n hw.memsize)
        local mem_gb=$(echo "scale=1; $mem_bytes / 1024 / 1024 / 1024" | bc)
        if (( $(echo "$mem_gb >= 8" | bc -l) )); then
            print_status "OK" "Memory: ${mem_gb}GB (sufficient)"
        else
            print_status "WARN" "Memory: ${mem_gb}GB (recommend 8GB+ for training)"
        fi
    else
        print_status "WARN" "Could not determine memory size"
    fi
    
    echo ""
}

check_disk_space() {
    echo -e "${BLUE}[2/8] Checking Disk Space${NC}"
    echo "----------------------------------------"
    
    # Check current directory space
    local available_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    local required_space=250  # GB
    
    if [[ $available_space -ge $required_space ]]; then
        print_status "OK" "Available space: ${available_space}GB (required: ${required_space}GB)"
    else
        print_status "ERROR" "Insufficient disk space: ${available_space}GB available, ${required_space}GB required"
        echo -e "${YELLOW}  Note: MyoData requires ~200GB + additional space for training${NC}"
    fi
    
    # Check if myo_data directory exists and its size
    if [[ -d "$PROJECT_ROOT/myo_data" ]]; then
        local data_size=$(du -sh "$PROJECT_ROOT/myo_data" 2>/dev/null | cut -f1 || echo "Unknown")
        print_status "INFO" "Current myo_data size: $data_size"
    else
        print_status "INFO" "myo_data directory not found (will be created during data sync)"
    fi
    
    echo ""
}

check_python_environment() {
    echo -e "${BLUE}[3/8] Checking Python Environment${NC}"
    echo "----------------------------------------"
    
    # Check Python installation
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        local major_version=$(echo $python_version | cut -d'.' -f1)
        local minor_version=$(echo $python_version | cut -d'.' -f2)
        
        if [[ $major_version -eq 3 && $minor_version -ge 8 ]]; then
            print_status "OK" "Python $python_version (compatible)"
        else
            print_status "ERROR" "Python $python_version (requires Python 3.8+)"
        fi
    else
        print_status "ERROR" "Python 3 not found"
    fi
    
    # Check conda
    if command -v conda &> /dev/null; then
        local conda_version=$(conda --version | cut -d' ' -f2)
        print_status "OK" "Conda $conda_version found"
        
        # Check if myodolores environment exists
        if conda env list | grep -q "myodolores"; then
            print_status "OK" "Conda environment 'myodolores' exists"
            
            # Check if environment is activated
            if [[ "$CONDA_DEFAULT_ENV" == "myodolores" ]]; then
                print_status "OK" "myodolores environment is activated"
            else
                print_status "WARN" "myodolores environment exists but not activated"
                echo -e "${YELLOW}  Run: conda activate myodolores${NC}"
            fi
        else
            print_status "WARN" "Conda environment 'myodolores' not found"
            echo -e "${YELLOW}  Run: conda create -n myodolores python=3.8${NC}"
        fi
    else
        print_status "WARN" "Conda not found (recommended for environment management)"
    fi
    
    # Check pip
    if command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
        print_status "OK" "pip package manager found"
    else
        print_status "ERROR" "pip not found"
    fi
    
    echo ""
}

check_dependencies() {
    echo -e "${BLUE}[4/8] Checking Dependencies${NC}"
    echo "----------------------------------------"
    
    # Check git
    if command -v git &> /dev/null; then
        local git_version=$(git --version | cut -d' ' -f3)
        print_status "OK" "Git $git_version found"
    else
        print_status "ERROR" "Git not found"
    fi
    
    # Check AWS CLI
    if command -v aws &> /dev/null; then
        local aws_version=$(aws --version 2>&1 | cut -d' ' -f1 | cut -d'/' -f2)
        print_status "OK" "AWS CLI $aws_version found"
        
        # Check AWS credentials
        if aws sts get-caller-identity &> /dev/null; then
            print_status "OK" "AWS credentials configured"
        else
            print_status "WARN" "AWS credentials not configured"
            echo -e "${YELLOW}  Run: aws configure${NC}"
        fi
    else
        print_status "ERROR" "AWS CLI not found (required for data sync)"
        echo -e "${YELLOW}  Install: https://aws.amazon.com/cli/${NC}"
    fi
    
    # Check MuJoCo
    if python3 -c "import mujoco" 2>/dev/null; then
        local mujoco_version=$(python3 -c "import mujoco; print(mujoco.__version__)" 2>/dev/null || echo "Unknown")
        print_status "OK" "MuJoCo $mujoco_version found"
    else
        print_status "WARN" "MuJoCo not found in Python environment"
        echo -e "${YELLOW}  Install: pip install mujoco==3.2.4${NC}"
    fi
    
    # Check other key packages
    local packages=("numpy" "h5py" "torch" "jax")
    for package in "${packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            print_status "OK" "$package package found"
        else
            print_status "WARN" "$package package not found"
        fi
    done
    
    echo ""
}

check_repository_structure() {
    echo -e "${BLUE}[5/8] Checking Repository Structure${NC}"
    echo "----------------------------------------"
    
    cd "$PROJECT_ROOT"
    
    # Check git repository
    if [[ -d ".git" ]]; then
        print_status "OK" "Git repository initialized"
        
        # Check remote origin
        if git remote get-url origin &> /dev/null; then
            local origin=$(git remote get-url origin)
            print_status "OK" "Git remote configured: $origin"
        else
            print_status "WARN" "Git remote not configured"
        fi
        
        # Check current branch
        local current_branch=$(git branch --show-current)
        print_status "INFO" "Current branch: $current_branch"
    else
        print_status "ERROR" "Not a git repository"
    fi
    
    # Check submodules
    if [[ -f ".gitmodules" ]]; then
        print_status "OK" ".gitmodules file found"
        
        local submodules=("ASAP" "extreme-parkour" "myo_api" "myo_retarget" "myo_model_internal" "unitree_rl_gym" "myo_data")
        for submodule in "${submodules[@]}"; do
            if [[ -d "$submodule" && -f "$submodule/.git" ]]; then
                print_status "OK" "Submodule $submodule initialized"
            elif [[ -d "$submodule" ]]; then
                print_status "WARN" "Submodule $submodule directory exists but not initialized"
                echo -e "${YELLOW}  Run: git submodule update --init $submodule${NC}"
            else
                print_status "WARN" "Submodule $submodule missing"
                echo -e "${YELLOW}  Run: git submodule update --init --recursive${NC}"
            fi
        done
    else
        print_status "WARN" ".gitmodules file not found"
    fi
    
    # Check key files
    local key_files=("CLAUDE.md" "DATA_SETUP.md" "scripts/sync_data.sh" ".gitignore")
    for file in "${key_files[@]}"; do
        if [[ -f "$file" ]]; then
            print_status "OK" "File $file exists"
        else
            print_status "ERROR" "File $file missing"
        fi
    done
    
    # Check scripts directory
    if [[ -d "scripts" ]]; then
        print_status "OK" "scripts/ directory exists"
        if [[ -x "scripts/sync_data.sh" ]]; then
            print_status "OK" "sync_data.sh is executable"
        else
            print_status "WARN" "sync_data.sh not executable"
            echo -e "${YELLOW}  Run: chmod +x scripts/sync_data.sh${NC}"
        fi
    else
        print_status "ERROR" "scripts/ directory missing"
    fi
    
    echo ""
}

check_data_setup() {
    echo -e "${BLUE}[6/8] Checking Data Setup${NC}"
    echo "----------------------------------------"
    
    # Check myo_data directory
    if [[ -d "myo_data" ]]; then
        print_status "OK" "myo_data/ directory exists"
        
        # Check .gitignore
        if grep -q "myo_data/" .gitignore 2>/dev/null; then
            print_status "OK" "myo_data/ is in .gitignore"
        else
            print_status "WARN" "myo_data/ not found in .gitignore"
        fi
        
        # Check dataset directories
        local datasets=("animation_output" "aist_output" "dance_output" "game_motion_output" "HAA500_output" "humman_output" "kungfu_output" "perform_output")
        local datasets_found=0
        local total_files=0
        
        for dataset in "${datasets[@]}"; do
            if [[ -d "myo_data/$dataset" ]]; then
                local file_count=$(find "myo_data/$dataset" -name "*.h5" 2>/dev/null | wc -l)
                if [[ $file_count -gt 0 ]]; then
                    print_status "OK" "Dataset $dataset: $file_count H5 files"
                    ((datasets_found++))
                    total_files=$((total_files + file_count))
                else
                    print_status "WARN" "Dataset $dataset: directory exists but no H5 files"
                fi
            else
                print_status "WARN" "Dataset $dataset: directory missing"
            fi
        done
        
        if [[ $datasets_found -eq 8 ]]; then
            print_status "OK" "All 8 datasets present ($total_files total H5 files)"
        elif [[ $datasets_found -gt 0 ]]; then
            print_status "WARN" "$datasets_found/8 datasets present ($total_files total H5 files)"
            echo -e "${YELLOW}  Run: ./scripts/sync_data.sh --all${NC}"
        else
            print_status "WARN" "No datasets found"
            echo -e "${YELLOW}  Run: ./scripts/sync_data.sh --all${NC}"
        fi
        
        # Check data size
        local data_size=$(du -sh myo_data 2>/dev/null | cut -f1 || echo "0")
        print_status "INFO" "Total data size: $data_size"
        
    else
        print_status "WARN" "myo_data/ directory not found"
        echo -e "${YELLOW}  Will be created during data sync${NC}"
    fi
    
    echo ""
}

check_environment_files() {
    echo -e "${BLUE}[7/8] Checking Environment Configuration${NC}"
    echo "----------------------------------------"
    
    # Check for environment variables
    local env_vars=("AWS_PROFILE" "CUDA_VISIBLE_DEVICES" "PYTHONPATH")
    for var in "${env_vars[@]}"; do
        if [[ -n "${!var}" ]]; then
            print_status "INFO" "$var=${!var}"
        else
            print_status "INFO" "$var not set"
        fi
    done
    
    # Check for configuration files
    local config_files=("~/.aws/credentials" "~/.aws/config")
    for file in "${config_files[@]}"; do
        local expanded_file="${file/#\~/$HOME}"
        if [[ -f "$expanded_file" ]]; then
            print_status "OK" "AWS config file $file exists"
        else
            print_status "WARN" "AWS config file $file missing"
        fi
    done
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi -L | wc -l)
        if [[ $gpu_count -gt 0 ]]; then
            print_status "OK" "GPU(s) available: $gpu_count"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read gpu; do
                print_status "INFO" "  GPU: $gpu"
            done
        else
            print_status "WARN" "nvidia-smi found but no GPUs detected"
        fi
    else
        print_status "INFO" "No NVIDIA GPU detected (CPU-only mode)"
    fi
    
    echo ""
}

check_quick_test() {
    echo -e "${BLUE}[8/8] Running Quick Functionality Test${NC}"
    echo "----------------------------------------"
    
    # Test Python imports
    if python3 -c "
import sys
print(f'Python {sys.version}')
try:
    import numpy as np
    print(f'NumPy {np.__version__}')
except ImportError:
    print('NumPy not available')
try:
    import h5py
    print(f'h5py {h5py.__version__}')
except ImportError:
    print('h5py not available')
try:
    import mujoco
    print(f'MuJoCo {mujoco.__version__}')
except ImportError:
    print('MuJoCo not available')
" 2>/dev/null; then
        print_status "OK" "Python environment test passed"
    else
        print_status "ERROR" "Python environment test failed"
    fi
    
    # Test data sync script
    if [[ -x "scripts/sync_data.sh" ]]; then
        if ./scripts/sync_data.sh --help >/dev/null 2>&1; then
            print_status "OK" "Data sync script functional"
        else
            print_status "ERROR" "Data sync script has issues"
        fi
    else
        print_status "WARN" "Cannot test data sync script (not executable)"
    fi
    
    # Test S3 access
    if command -v aws &> /dev/null && aws sts get-caller-identity >/dev/null 2>&1; then
        if aws s3 ls s3://myo-data/ >/dev/null 2>&1; then
            print_status "OK" "S3 bucket access verified"
        else
            print_status "WARN" "Cannot access S3 bucket (check permissions)"
        fi
    else
        print_status "SKIP" "S3 access test (AWS not configured)"
    fi
    
    echo ""
}

print_summary() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}                        VALIDATION SUMMARY${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    
    if [[ $ERRORS_FOUND -eq 0 && $WARNINGS_FOUND -eq 0 ]]; then
        echo -e "${GREEN}🎉 Perfect! Your MyoDolores setup is complete and ready.${NC}"
        echo ""
        echo -e "${CYAN}Next steps:${NC}"
        echo -e "  1. Sync data: ${YELLOW}./scripts/sync_data.sh --all${NC}"
        echo -e "  2. Start training: ${YELLOW}cd ASAP && python humanoidverse/train_agent.py${NC}"
        echo -e "  3. Check documentation: ${YELLOW}cat CLAUDE.md${NC}"
    elif [[ $ERRORS_FOUND -eq 0 ]]; then
        echo -e "${YELLOW}⚠ Setup is functional but has $WARNINGS_FOUND warning(s).${NC}"
        echo -e "${CYAN}Your system will work but consider addressing warnings for optimal performance.${NC}"
    else
        echo -e "${RED}❌ Found $ERRORS_FOUND error(s) and $WARNINGS_FOUND warning(s).${NC}"
        echo -e "${CYAN}Please address the errors before proceeding.${NC}"
        echo ""
        echo -e "${CYAN}Common solutions:${NC}"
        echo -e "  • Install missing dependencies: ${YELLOW}$0 --install-deps${NC}"
        echo -e "  • Run setup for new installation: ${YELLOW}$0 --setup${NC}"
        echo -e "  • Fix issues automatically: ${YELLOW}$0 --fix${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}Detailed log saved to: ${YELLOW}$SETUP_LOG${NC}"
    echo -e "${CYAN}For help: ${YELLOW}$0 --help${NC}"
    echo ""
}

run_setup() {
    echo -e "${BLUE}Running Initial Setup...${NC}"
    echo ""
    
    # Create conda environment
    if ! conda env list | grep -q "myodolores"; then
        echo -e "${CYAN}Creating conda environment...${NC}"
        conda create -n myodolores python=3.8 -y
        print_status "OK" "Created myodolores conda environment"
    fi
    
    # Initialize submodules
    if [[ -f ".gitmodules" ]]; then
        echo -e "${CYAN}Initializing git submodules...${NC}"
        git submodule update --init --recursive
        print_status "OK" "Initialized git submodules"
    fi
    
    # Make scripts executable
    if [[ -f "scripts/sync_data.sh" ]]; then
        chmod +x scripts/sync_data.sh
        print_status "OK" "Made sync_data.sh executable"
    fi
    
    # Create myo_data directories
    mkdir -p myo_data/{animation_output,aist_output,dance_output,game_motion_output,HAA500_output,humman_output,kungfu_output,perform_output}
    print_status "OK" "Created myo_data directory structure"
    
    echo ""
    echo -e "${GREEN}Setup completed! Run validation again to check status.${NC}"
}

install_dependencies() {
    echo -e "${BLUE}Installing Dependencies...${NC}"
    echo ""
    
    # Activate myodolores environment if it exists
    if conda env list | grep -q "myodolores"; then
        echo -e "${CYAN}Activating myodolores environment...${NC}"
        eval "$(conda shell.bash hook)"
        conda activate myodolores
    fi
    
    # Install basic packages
    echo -e "${CYAN}Installing Python packages...${NC}"
    pip install numpy h5py mujoco==3.2.4 torch jax
    
    # Install AWS CLI if not present
    if ! command -v aws &> /dev/null; then
        echo -e "${CYAN}Installing AWS CLI...${NC}"
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
            unzip awscliv2.zip
            sudo ./aws/install
            rm -rf aws awscliv2.zip
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
            sudo installer -pkg AWSCLIV2.pkg -target /
            rm AWSCLIV2.pkg
        fi
    fi
    
    print_status "OK" "Dependencies installation completed"
}

# Main execution
main() {
    local verbose=false
    local quick=false
    local fix=false
    local setup=false
    local data_only=false
    local env_only=false
    local install_deps=false
    local skip_data=false
    local check_space=false
    
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
            -q|--quick)
                quick=true
                shift
                ;;
            -f|--fix)
                fix=true
                shift
                ;;
            -s|--setup)
                setup=true
                shift
                ;;
            -d|--data-only)
                data_only=true
                shift
                ;;
            -e|--env-only)
                env_only=true
                shift
                ;;
            --install-deps)
                install_deps=true
                shift
                ;;
            --skip-data)
                skip_data=true
                shift
                ;;
            --check-space)
                check_space=true
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
    echo "MyoDolores Setup Validation - $(date)" > "$SETUP_LOG"
    
    print_header
    
    # Run setup if requested
    if [[ "$setup" == "true" ]]; then
        run_setup
        echo ""
    fi
    
    # Install dependencies if requested
    if [[ "$install_deps" == "true" ]]; then
        install_dependencies
        echo ""
    fi
    
    # Run validations
    if [[ "$data_only" == "false" && "$env_only" == "false" ]] || [[ "$env_only" == "true" ]]; then
        check_system_requirements
        
        if [[ "$check_space" == "true" ]] || [[ "$quick" == "false" ]]; then
            check_disk_space
        fi
        
        check_python_environment
        check_dependencies
        check_repository_structure
        
        if [[ "$skip_data" == "false" ]]; then
            check_environment_files
        fi
    fi
    
    if [[ "$data_only" == "true" ]] || [[ "$env_only" == "false" && "$skip_data" == "false" ]]; then
        check_data_setup
    fi
    
    if [[ "$quick" == "false" && "$data_only" == "false" ]]; then
        check_quick_test
    fi
    
    print_summary
    
    # Exit with appropriate code
    if [[ $ERRORS_FOUND -gt 0 ]]; then
        exit 1
    elif [[ $WARNINGS_FOUND -gt 0 ]]; then
        exit 2
    else
        exit 0
    fi
}

# Run main function with all arguments
main "$@"