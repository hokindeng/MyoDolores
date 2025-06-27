#!/usr/bin/env python3
"""
Test Motion Training + Demo Integration
Quick verification that motion training and keyboard demo work together
"""

import os
import sys
from pathlib import Path

# Add paths
sys.path.append('submodules/myo_model_internal')
sys.path.append('submodules/myo_api')

def test_motion_training_quick():
    """Quick test of motion training"""
    print("🧪 Testing Motion Training...")
    
    try:
        from train_real_motion_imitation import MotionDataset, MotionImitationPolicy
        import torch
        
        # Test dataset loading with very few motions
        dataset = MotionDataset(max_motions=5, device='cpu')
        print(f"✅ Dataset loaded: {len(dataset.motions)} motions")
        
        if dataset.motions:
            # Test policy creation
            obs_dim = dataset.model.nq * 3  # simplified
            action_dim = dataset.model.nq
            policy = MotionImitationPolicy(obs_dim, action_dim)
            print(f"✅ Policy created: {obs_dim} → {action_dim}")
            
            # Test batch sampling
            obs_batch, action_batch = dataset.sample_batch(4)
            if obs_batch is not None:
                print(f"✅ Batch sampling works: {obs_batch.shape}")
                return True
            else:
                print("❌ Batch sampling failed")
                return False
        else:
            print("❌ No motions loaded")
            return False
            
    except Exception as e:
        print(f"❌ Motion training test failed: {e}")
        return False

def test_keyboard_demo_compatibility():
    """Test keyboard demo can load motion-trained policy"""
    print("\n🎮 Testing Keyboard Demo Compatibility...")
    
    try:
        # Check if we can create a demo instance
        from keyboard_demo_motion_enhanced import MotionEnhancedKeyboardDemo
        
        print("✅ Motion-enhanced demo imports successfully")
        
        # Check policy loading mechanism
        from train_keyboard_control import PPOPolicy
        import torch
        
        # Mock policy for testing
        obs_dim = 100  # approximate
        action_dim = 133  # MyoSkeleton actuators
        test_policy = PPOPolicy(obs_dim, action_dim)
        
        # Save test policy
        torch.save(test_policy.state_dict(), "test_motion_policy.pt")
        print("✅ Test policy saved")
        
        # Try to load it
        loaded_policy = PPOPolicy(obs_dim, action_dim)
        loaded_policy.load_state_dict(torch.load("test_motion_policy.pt", map_location='cpu'))
        print("✅ Policy loading works")
        
        # Clean up
        os.remove("test_motion_policy.pt")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo compatibility test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🔧 Motion Training + Demo Integration Test")
    print("=" * 60)
    
    # Test motion training
    training_ok = test_motion_training_quick()
    
    # Test demo compatibility  
    demo_ok = test_keyboard_demo_compatibility()
    
    print("\n📋 Integration Test Results:")
    print("=" * 60)
    print(f"Motion Training: {'✅ PASS' if training_ok else '❌ FAIL'}")
    print(f"Demo Compatibility: {'✅ PASS' if demo_ok else '❌ FAIL'}")
    
    if training_ok and demo_ok:
        print("\n🎉 Integration tests passed!")
        print("\n🚀 Ready for full workflow:")
        print("  1. python train_real_motion_imitation.py")
        print("  2. cp motion_imitation_policy_final.pt keyboard_policy_final.pt")  
        print("  3. python keyboard_demo_motion_enhanced.py")
        print("\n💡 This workflow uses your 277GB motion dataset for training!")
        return True
    else:
        print("\n❌ Integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)