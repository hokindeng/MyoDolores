#!/usr/bin/env python3
"""
Debug Motion Training Dimensions
Find exact observation dimensions for proper training setup
"""

import sys
sys.path.append('submodules/myo_model_internal')
sys.path.append('submodules/myo_api')

from train_real_motion_imitation import MotionDataset
import torch

def debug_dimensions():
    """Debug exact dimensions"""
    print("🔍 Debugging Motion Training Dimensions...")
    
    # Load small dataset
    dataset = MotionDataset(max_motions=1, device='cpu')
    
    if not dataset.motions:
        print("❌ No motions loaded")
        return
    
    print(f"Model nq (positions): {dataset.model.nq}")
    print(f"Model nv (velocities): {dataset.model.nv}") 
    print(f"Model nu (actuators): {dataset.model.nu}")
    
    motion = dataset.motions[0]
    print(f"Motion qpos shape: {motion['qpos'].shape}")
    print(f"Motion qvel shape: {motion['qvel'].shape}")
    
    # Test sample batch
    obs_batch, action_batch = dataset.sample_batch(1)
    if obs_batch is not None:
        print(f"Observation batch shape: {obs_batch.shape}")
        print(f"Action batch shape: {action_batch.shape}")
        
        # Calculate expected dimensions
        expected_obs = dataset.model.nq * 3  # current_pos + current_vel + target_pos
        actual_obs = obs_batch.shape[1]
        
        print(f"Expected obs dim: {expected_obs}")
        print(f"Actual obs dim: {actual_obs}")
        print(f"Difference: {expected_obs - actual_obs}")
        
        return actual_obs, dataset.model.nu
    else:
        print("❌ Failed to create batch")
        return None, None

if __name__ == "__main__":
    debug_dimensions()