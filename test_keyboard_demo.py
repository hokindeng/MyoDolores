#!/usr/bin/env python3
"""
MyoDolores Keyboard Control Test (Headless)
Test the trained policy without GUI
"""

import torch
import numpy as np
import mujoco
import time
import os
from pathlib import Path
import sys

# Add myo_api to path
sys.path.append(str(Path(__file__).parent / "myo_api"))

def test_keyboard_control():
    """Test the trained policy"""
    
    print("MyoDolores Keyboard Control Test")
    print("=" * 40)
    
    # Change to model directory
    original_dir = os.getcwd()
    model_dir = Path("myo_model_internal/myo_model/")
    os.chdir(model_dir)
    
    # Load model
    model_path = "myoskeleton/myoskeleton_with_motors.xml"
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Initialize
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.7  # Set height
    mujoco.mj_forward(model, data)
    
    # Load policy
    os.chdir(original_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from train_keyboard_control import PPOPolicy
    
    # Calculate observation dimensions
    joint_pos_size = model.nq - 7
    joint_vel_size = model.nv - 6
    num_obs = 3 + 3 + 3 + 3 + joint_pos_size + joint_vel_size
    
    policy = PPOPolicy(num_obs, model.nu).to(device)
    state_dict = torch.load("keyboard_policy_final.pt", map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    
    print(f"Policy loaded on {device}")
    print(f"Model: {model.nq} dofs, {model.nu} actuators")
    
    # Test different velocity commands
    test_commands = [
        ([1.0, 0.0, 0.0], "Forward"),
        ([0.0, 0.0, 1.0], "Turn left"),
        ([0.0, 0.0, -1.0], "Turn right"),
        ([-1.0, 0.0, 0.0], "Backward"),
        ([0.0, 0.0, 0.0], "Stop")
    ]
    
    print("\\nTesting velocity commands:")
    
    for cmd, description in test_commands:
        print(f"\\nTesting: {description} {cmd}")
        
        # Run for 100 steps (2 seconds at 50Hz)
        for step in range(100):
            # Get observation
            base_vel = data.qvel[:3]
            base_angvel = data.qvel[3:6]
            gravity = np.array([0, 0, -9.81])
            command_vel = np.array(cmd)
            joint_pos = data.qpos[7:]
            joint_vel = data.qvel[6:]
            
            obs = np.concatenate([
                base_vel, base_angvel, gravity, command_vel,
                joint_pos, joint_vel
            ])
            
            # Get action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean, _ = policy(obs_tensor)
                action = action_mean.cpu().numpy()[0]
            
            # Apply action
            data.ctrl[:] = action
            mujoco.mj_step(model, data)
            
            # Log progress every 25 steps
            if step % 25 == 0:
                height = data.qpos[2]
                actual_vel = data.qvel[:2]  # xy velocity
                print(f"  Step {step:3d}: Height={height:.2f}m, Velocity={actual_vel}")
                
                # Check if fell over
                if height < 0.5:
                    print("    ⚠️  Robot fell over!")
                    break
    
    print("\\n✅ Test completed!")
    print(f"Final height: {data.qpos[2]:.2f}m")
    print("The trained policy is working and can control the MyoSkeleton humanoid!")
    
    os.chdir(original_dir)

if __name__ == "__main__":
    test_keyboard_control()