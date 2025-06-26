#!/bin/bash
# Create Transfer Package for Personal Computer
# Creates a zip file with everything needed to run MyoDolores on personal computers

set -e

echo "Creating MyoDolores Transfer Package..."

# Package info
PACKAGE_NAME="MyoDolores_KeyboardControl_v1.0"
PACKAGE_DIR="$PACKAGE_NAME"
ARCHIVE_NAME="$PACKAGE_NAME.tar.gz"

# Clean previous package
rm -rf "$PACKAGE_DIR" "$ARCHIVE_NAME" 2>/dev/null || true

# Create package directory
mkdir -p "$PACKAGE_DIR"

echo "📁 Copying essential files..."

# Core scripts
cp train_keyboard_control.py "$PACKAGE_DIR/"
cp keyboard_demo.py "$PACKAGE_DIR/"
cp test_keyboard_demo.py "$PACKAGE_DIR/"

# Setup and validation
cp setup_myodolores.sh "$PACKAGE_DIR/"
cp validate_setup.py "$PACKAGE_DIR/"
cp requirements.txt "$PACKAGE_DIR/"

# Documentation
cp PERSONAL_SETUP.md "$PACKAGE_DIR/README.md"
cp CLAUDE.md "$PACKAGE_DIR/PROJECT_OVERVIEW.md" 

# Copy trained policy if it exists
if [ -f "keyboard_policy_final.pt" ]; then
    cp keyboard_policy_final.pt "$PACKAGE_DIR/"
    echo "✓ Including trained policy (1.2MB)"
else
    echo "! No trained policy found - will need to train on target machine"
fi

# Copy model files (essential)
echo "📂 Copying MyoSkeleton model files..."
mkdir -p "$PACKAGE_DIR/myo_model_internal"
cp -r myo_model_internal/myo_model "$PACKAGE_DIR/myo_model_internal/"

# Copy myo_api (essential)
echo "📂 Copying myo_api..."
cp -r myo_api "$PACKAGE_DIR/"

# Create simplified environment file
cat > "$PACKAGE_DIR/environment.yml" << 'EOF'
name: myodolores
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.8
  - numpy
  - scipy
  - matplotlib
  - h5py
  - pyyaml
  - pillow
  - pip
  - pip:
    - mujoco==3.2.4
    - torch
    - torchvision
    - torchaudio
    - termcolor
EOF

# Create quick install script
cat > "$PACKAGE_DIR/INSTALL.sh" << 'EOF'
#!/bin/bash
# Quick Install Script for MyoDolores

echo "MyoDolores Keyboard Control - Quick Install"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
echo "Creating conda environment..."
conda env create -f environment.yml

echo "Environment created! To activate:"
echo "  conda activate myodolores"
echo ""
echo "Then run:"
echo "  python validate_setup.py      # Check installation"
echo "  python train_keyboard_control.py  # Train policy (10-30 min)"
echo "  python test_keyboard_demo.py      # Test demo"
EOF

chmod +x "$PACKAGE_DIR/INSTALL.sh"

# Create Windows batch file
cat > "$PACKAGE_DIR/INSTALL.bat" << 'EOF'
@echo off
echo MyoDolores Keyboard Control - Windows Install
echo ==========================================

REM Check if conda is available
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Conda not found. Please install Miniconda first:
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Create environment
echo Creating conda environment...
conda env create -f environment.yml

echo.
echo Environment created! To activate:
echo   conda activate myodolores
echo.
echo Then run:
echo   python validate_setup.py
echo   python train_keyboard_control.py
echo   python test_keyboard_demo.py
pause
EOF

# Create startup scripts
cat > "$PACKAGE_DIR/activate_and_validate.sh" << 'EOF'
#!/bin/bash
# Activate environment and validate setup

source $(conda info --base)/etc/profile.d/conda.sh
conda activate myodolores
python validate_setup.py
EOF

chmod +x "$PACKAGE_DIR/activate_and_validate.sh"

cat > "$PACKAGE_DIR/train_and_test.sh" << 'EOF'
#!/bin/bash
# Train policy and run test

source $(conda info --base)/etc/profile.d/conda.sh
conda activate myodolores

echo "Training keyboard control policy..."
python train_keyboard_control.py

echo "Testing trained policy..."
python test_keyboard_demo.py
EOF

chmod +x "$PACKAGE_DIR/train_and_test.sh"

# Create package info file
cat > "$PACKAGE_DIR/PACKAGE_INFO.txt" << EOF
MyoDolores Keyboard Control Package
===================================

Package Version: 1.0
Created: $(date)
Created on: $(hostname)

Contents:
- Complete keyboard control implementation
- MyoSkeleton humanoid model (324 DOF)
- RL training pipeline with PPO
- Interactive demo with arrow key control
- Comprehensive setup and validation scripts

Hardware Requirements:
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- NVIDIA GPU recommended (CPU works but slower)
- 5GB disk space

Quick Start:
1. Extract this package
2. Run: ./INSTALL.sh (Linux/Mac) or INSTALL.bat (Windows)
3. Run: conda activate myodolores
4. Run: python validate_setup.py
5. Run: python train_keyboard_control.py
6. Run: python test_keyboard_demo.py

For detailed instructions, see README.md
EOF

# Create file listing
echo "📋 Creating file listing..."
find "$PACKAGE_DIR" -type f | sort > "$PACKAGE_DIR/FILE_LIST.txt"

# Calculate sizes
TOTAL_SIZE=$(du -sh "$PACKAGE_DIR" | cut -f1)
echo "📊 Package size: $TOTAL_SIZE"

# Create archive
echo "📦 Creating archive..."
tar -czf "$ARCHIVE_NAME" "$PACKAGE_DIR"

ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)

echo ""
echo "✅ Transfer package created successfully!"
echo "📦 Archive: $ARCHIVE_NAME ($ARCHIVE_SIZE)"
echo "📁 Directory: $PACKAGE_DIR ($TOTAL_SIZE)"
echo ""
echo "Package contents:"
echo "  • Complete MyoDolores implementation"
echo "  • MyoSkeleton humanoid model"
echo "  • RL training and demo scripts"
echo "  • Automated setup and validation"
echo "  • Cross-platform installation scripts"
if [ -f "keyboard_policy_final.pt" ]; then
    echo "  • Pre-trained policy (ready to use)"
else
    echo "  • Training required on target machine"
fi
echo ""
echo "To transfer to personal computer:"
echo "1. Copy $ARCHIVE_NAME to target machine"
echo "2. Extract: tar -xzf $ARCHIVE_NAME"
echo "3. Follow instructions in README.md"
echo ""
echo "📧 The package is ready for transfer!"