#!/usr/bin/env python3
"""
Test ASAP with MuJoCo Backend
Quick test to verify MuJoCo configuration works with ASAP
"""

import os
import sys
from pathlib import Path

# Add ASAP to path
sys.path.append('ASAP')

def test_asap_mujoco_config():
    """Test ASAP MuJoCo configuration"""
    print("🔧 Testing ASAP MuJoCo Configuration...")
    
    try:
        # Test basic imports
        from humanoidverse.simulator.mujoco.mujoco import MuJoCo
        print("✅ MuJoCo simulator import successful")
        
        # Test config loading
        from omegaconf import OmegaConf
        
        # Load MuJoCo config
        mujoco_config_path = "ASAP/humanoidverse/config/simulator/mujoco.yaml"
        if Path(mujoco_config_path).exists():
            config = OmegaConf.load(mujoco_config_path)
            print(f"✅ MuJoCo config loaded: {config.simulator._target_}")
        else:
            print(f"❌ MuJoCo config not found: {mujoco_config_path}")
            return False
        
        # Check if MyoSkeleton model exists
        from myo_model.utils.model_utils import get_model_xml_path
        model_path = get_model_xml_path('motors')
        if Path(model_path).exists():
            print(f"✅ MyoSkeleton model found: {model_path}")
        else:
            print(f"❌ MyoSkeleton model not found: {model_path}")
            return False
        
        # Test MuJoCo model loading
        import mujoco
        model = mujoco.MjModel.from_xml_path(model_path)
        print(f"✅ MuJoCo model loaded: {model.nq} DOFs, {model.nu} actuators")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_motion_data_access():
    """Test motion data access"""
    print("\n📊 Testing Motion Data Access...")
    
    try:
        import glob
        
        # Check motion data availability
        pattern = "myo_data/**/target_*.h5"
        motion_files = glob.glob(pattern, recursive=True)
        
        if motion_files:
            print(f"✅ Found {len(motion_files)} motion files")
            print(f"   Sample: {motion_files[0]}")
            return True
        else:
            print("❌ No motion files found")
            return False
            
    except Exception as e:
        print(f"❌ Motion data error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ASAP MuJoCo Integration Test")
    print("=" * 50)
    
    # Test ASAP config
    config_ok = test_asap_mujoco_config()
    
    # Test motion data
    data_ok = test_motion_data_access()
    
    print("\n📋 Test Results:")
    print("=" * 50)
    print(f"ASAP MuJoCo Config: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Motion Data Access: {'✅ PASS' if data_ok else '❌ FAIL'}")
    
    if config_ok and data_ok:
        print("\n🎉 All tests passed! Ready for ASAP training with MuJoCo")
        print("\n🚀 Next steps:")
        print("  python train_asap_motion_tracking.py --max-motions 100 --dry-run")
        print("  python train_asap_motion_tracking.py --max-motions 100")
        return True
    else:
        print("\n❌ Some tests failed. Check configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)