#!/usr/bin/env python3
"""
MyoDolores Keyboard Control Demo
Real-time keyboard control of MyoSkeleton humanoid using trained RL policy
"""

import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import threading
from pathlib import Path
import sys

# Add myo_api to path
sys.path.append(str(Path(__file__).parent / "myo_api"))

class KeyboardControlDemo:
    """Real-time keyboard control demo"""
    
    def __init__(self, policy_path="keyboard_policy_final.pt"):
        # Store original directory
        self.original_dir = os.getcwd()
        
        # Use proper model path resolution
        sys.path.append('submodules/myo_model_internal')
        from myo_model.utils.model_utils import get_model_xml_path
        
        # Load MyoSkeleton model
        model_path = get_model_xml_path('motors')
        print(f"Loading model: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize model
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 1.7  # Set initial height
        mujoco.mj_forward(self.model, self.data)
        
        # Calculate observation dimensions (same as training)
        joint_pos_size = self.model.nq - 7  # Skip root position/orientation
        joint_vel_size = self.model.nv - 6  # Skip root velocity  
        self.num_obs = 3 + 3 + 3 + 3 + joint_pos_size + joint_vel_size
        
        print(f"Model: {self.model.nq} dofs, {self.model.nu} actuators, {self.num_obs} observations")
        
        # Load trained policy
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = self.load_policy(policy_path)
        
        # Control state
        self.velocity_commands = np.zeros(3)  # [vel_x, vel_y, ang_vel_z]
        self.running = True
        
        print(f"Demo ready! Device: {self.device}")
        print("Controls:")
        print("  Arrow Keys: UP/DOWN = forward/backward, LEFT/RIGHT = turn left/right")
        print("  SPACE = stop, ESC = quit")
        
    def load_policy(self, policy_path):
        """Load the trained policy"""
        from train_keyboard_control import PPOPolicy
        
        # Create policy with same architecture as training
        policy = PPOPolicy(self.num_obs, self.model.nu).to(self.device)
        
        if Path(policy_path).exists():
            state_dict = torch.load(policy_path, map_location=self.device)
            policy.load_state_dict(state_dict)
            print(f"Loaded policy from {policy_path}")
        else:
            print(f"Warning: Policy file {policy_path} not found, using random policy")
            
        policy.eval()
        return policy
        
    def get_observation(self):
        """Get current observation"""
        # Base velocity (3)
        base_vel = self.data.qvel[:3]
        
        # Base angular velocity (3)
        base_angvel = self.data.qvel[3:6]
        
        # Gravity vector in base frame (3)
        gravity = np.array([0, 0, -9.81])
        
        # Current velocity commands (3)
        command_vel = self.velocity_commands.copy()
        
        # Joint positions (skip root position/orientation)
        joint_pos = self.data.qpos[7:]
        
        # Joint velocities (skip root velocity)
        joint_vel = self.data.qvel[6:]
        
        # Combine observation
        obs = np.concatenate([
            base_vel, base_angvel, gravity, command_vel,
            joint_pos, joint_vel
        ])
        
        return obs
        
    def control_loop(self):
        """Main control loop running at 50Hz"""
        dt = 0.02  # 50Hz
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("Viewer launched! Use arrow keys for control...")
            
            while self.running and viewer.is_running():
                start_time = time.time()
                
                # Get observation
                obs = self.get_observation()
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # Get action from policy
                with torch.no_grad():
                    action_mean, _ = self.policy(obs_tensor)
                    action = action_mean.cpu().numpy()[0]
                
                # Apply action
                self.data.ctrl[:] = action
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update viewer
                viewer.sync()
                
                # Maintain 50Hz timing
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
    def keyboard_thread(self):
        """Keyboard input thread (simplified for demo)"""
        try:
            import keyboard
            print("Using keyboard library for input")
            
            while self.running:
                if keyboard.is_pressed('up'):
                    self.velocity_commands[0] = 1.0  # Forward
                elif keyboard.is_pressed('down'):
                    self.velocity_commands[0] = -1.0  # Backward
                else:
                    self.velocity_commands[0] = 0.0
                    
                if keyboard.is_pressed('left'):
                    self.velocity_commands[2] = 1.0  # Turn left
                elif keyboard.is_pressed('right'):
                    self.velocity_commands[2] = -1.0  # Turn right
                else:
                    self.velocity_commands[2] = 0.0
                    
                if keyboard.is_pressed('space'):
                    self.velocity_commands[:] = 0.0  # Stop
                    
                if keyboard.is_pressed('esc'):
                    self.running = False
                    break
                    
                time.sleep(0.05)  # 20Hz keyboard polling
                
        except ImportError:
            print("keyboard library not available, using manual control")
            print("Enter commands manually:")
            print("  'w' = forward, 's' = backward, 'a' = left, 'd' = right, 'q' = quit")
            
            while self.running:
                try:
                    cmd = input("Command (w/s/a/d/q): ").lower()
                    if cmd == 'w':
                        self.velocity_commands[0] = 1.0
                        print("Moving forward")
                    elif cmd == 's':
                        self.velocity_commands[0] = -1.0
                        print("Moving backward")
                    elif cmd == 'a':
                        self.velocity_commands[2] = 1.0
                        print("Turning left")
                    elif cmd == 'd':
                        self.velocity_commands[2] = -1.0
                        print("Turning right")
                    elif cmd == 'q':
                        self.running = False
                        break
                    else:
                        self.velocity_commands[:] = 0.0
                        print("Stopping")
                except KeyboardInterrupt:
                    self.running = False
                    break
    
    def run(self):
        """Run the demo"""
        try:
            # Start keyboard input thread
            keyboard_thread = threading.Thread(target=self.keyboard_thread, daemon=True)
            keyboard_thread.start()
            
            # Run main control loop
            self.control_loop()
            
        except KeyboardInterrupt:
            print("Demo interrupted")
        finally:
            self.running = False
            os.chdir(self.original_dir)
            print("Demo ended")


def main():
    """Main function"""
    print("MyoDolores Keyboard Control Demo")
    print("=" * 50)
    
    # Check for trained policy
    policy_file = "keyboard_policy_final.pt"
    if not Path(policy_file).exists():
        print(f"No trained policy found at {policy_file}")
        print("Please run train_keyboard_control.py first!")
        return
    
    # Create and run demo
    demo = KeyboardControlDemo(policy_file)
    demo.run()


if __name__ == "__main__":
    main()