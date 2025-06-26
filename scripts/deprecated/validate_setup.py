#!/usr/bin/env python3
"""
MyoDolores Setup Validation Script
Checks if everything is properly installed and configured
"""

import sys
import os
import subprocess
from pathlib import Path

def colored_print(text, color='white'):
    """Print colored text"""
    colors = {
        'red': '\033[0;31m',
        'green': '\033[0;32m',
        'yellow': '\033[1;33m',
        'blue': '\033[0;34m',
        'white': '\033[0m'
    }
    print(f"{colors.get(color, colors['white'])}{text}\033[0m")

def check_requirement(name, check_func, required=True):
    """Check a requirement and print status"""
    try:
        result = check_func()
        if result:
            colored_print(f"✓ {name}", 'green')
            return True
        else:
            colored_print(f"✗ {name}", 'red')
            return False
    except Exception as e:
        colored_print(f"✗ {name}: {str(e)}", 'red')
        return False

def main():
    """Main validation function"""
    colored_print("MyoDolores Setup Validation", 'blue')
    colored_print("=" * 40, 'blue')
    
    all_passed = True
    
    # Python version
    def check_python():
        version = sys.version_info
        return version.major >= 3 and version.minor >= 8
    
    if not check_requirement(f"Python {sys.version.split()[0]}", check_python):
        colored_print("  Required: Python 3.8+", 'yellow')
        all_passed = False
    
    # Core packages
    packages = [
        ('numpy', lambda: __import__('numpy')),
        ('scipy', lambda: __import__('scipy')),
        ('matplotlib', lambda: __import__('matplotlib')),
        ('h5py', lambda: __import__('h5py')),
        ('torch', lambda: __import__('torch')),
        ('mujoco', lambda: __import__('mujoco')),
    ]
    
    colored_print("\n📦 Python Packages:", 'blue')
    for name, import_func in packages:
        if not check_requirement(name, import_func):
            all_passed = False
    
    # PyTorch CUDA
    def check_cuda():
        import torch
        return torch.cuda.is_available()
    
    colored_print("\n🎮 Hardware:", 'blue')
    if check_requirement("CUDA available", check_cuda, required=False):
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        colored_print(f"  GPU: {gpu_name}", 'green')
    else:
        colored_print("  Will use CPU training (slower)", 'yellow')
    
    # Project files
    files = [
        "train_keyboard_control.py",
        "keyboard_demo.py", 
        "test_keyboard_demo.py",
        "myo_model_internal/myo_model/myoskeleton/myoskeleton_with_motors.xml"
    ]
    
    colored_print("\n📁 Project Files:", 'blue')
    for file_path in files:
        def check_file():
            return Path(file_path).exists()
        
        if not check_requirement(file_path, check_file):
            all_passed = False
    
    # MuJoCo test
    def check_mujoco():
        import mujoco
        # Try to load a simple model
        xml = """
        <mujoco>
            <worldbody>
                <body>
                    <geom type="box" size="1 1 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        return True
    
    colored_print("\n🔧 MuJoCo Test:", 'blue')
    if not check_requirement("MuJoCo simulation", check_mujoco):
        all_passed = False
    
    # MyoSkeleton model test
    def check_myoskeleton():
        import mujoco
        import os
        
        # Change to model directory
        original_dir = os.getcwd()
        try:
            os.chdir("myo_model_internal/myo_model/")
            model = mujoco.MjModel.from_xml_path("myoskeleton/myoskeleton_with_motors.xml")
            data = mujoco.MjData(model)
            return model.nu > 0  # Has actuators
        finally:
            os.chdir(original_dir)
    
    colored_print("\n🦴 MyoSkeleton Model:", 'blue')
    if check_requirement("Model loading", check_myoskeleton):
        # Get model info
        try:
            import mujoco
            import os
            original_dir = os.getcwd()
            os.chdir("myo_model_internal/myo_model/")
            model = mujoco.MjModel.from_xml_path("myoskeleton/myoskeleton_with_motors.xml")
            os.chdir(original_dir)
            colored_print(f"  DOFs: {model.nq}, Actuators: {model.nu}", 'green')
        except:
            pass
    else:
        all_passed = False
    
    # Training readiness
    colored_print("\n🏋️ Training Readiness:", 'blue')
    
    def check_training_ready():
        # All core requirements met
        return all_passed
    
    training_ready = check_requirement("All requirements met", check_training_ready)
    
    # Policy file check
    def check_policy():
        return Path("keyboard_policy_final.pt").exists()
    
    if check_requirement("Trained policy available", check_policy, required=False):
        colored_print("  Ready to run demo!", 'green')
    else:
        colored_print("  Run training first: python train_keyboard_control.py", 'yellow')
    
    # Final summary
    colored_print("\n" + "=" * 40, 'blue')
    if all_passed:
        colored_print("🎉 All requirements satisfied!", 'green')
        colored_print("\nNext steps:", 'blue')
        colored_print("1. Train policy: python train_keyboard_control.py", 'white')
        colored_print("2. Test demo: python test_keyboard_demo.py", 'white')
        colored_print("3. Interactive: python keyboard_demo.py", 'white')
    else:
        colored_print("❌ Some requirements missing", 'red')
        colored_print("\nPlease install missing dependencies and try again", 'yellow')
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())