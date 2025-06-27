#!/usr/bin/env python3
"""
MyoDolores Motion-Enhanced Keyboard Control Demo
Real-time keyboard control enhanced with actual motion data from 277GB dataset
Uses motion clips as reference for more natural movement
"""

import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import threading
import random
import glob
from pathlib import Path
import sys

# Add necessary paths
sys.path.append('submodules/myo_model_internal')
sys.path.append('submodules/myo_api')

from myo_model.utils.model_utils import get_model_xml_path
from myo_api.mj.motion.trajectory import mjTrajectory

class MotionEnhancedKeyboardDemo:
    """Motion-enhanced keyboard control demo using real motion data"""
    
    def __init__(self, policy_path="keyboard_policy_final.pt", max_motions=100):
        print("🎭 MyoDolores Motion-Enhanced Keyboard Demo")
        print("📊 Loading motion data from 277GB dataset...")
        
        # Load MyoSkeleton model
        model_path = get_model_xml_path('motors')
        print(f"Loading model: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize model
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 1.7  # Set initial height
        mujoco.mj_forward(self.model, self.data)
        
        print(f"Model: {self.model.nq} dofs, {self.model.nu} actuators")
        
        # Load motion data
        self.load_motion_data(max_motions)
        
        # Load trained policy
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = self.load_policy(policy_path)
        
        # Control state
        self.velocity_commands = np.zeros(3)  # [vel_x, vel_y, ang_vel_z]
        self.current_motion = None
        self.motion_frame = 0
        self.motion_mode = False  # Toggle between motion playback and keyboard control
        self.running = True
        
        print(f"Demo ready! Device: {self.device}")
        print("🎮 Enhanced Controls:")
        print("  Arrow Keys: UP/DOWN = forward/backward, LEFT/RIGHT = turn")
        print("  SPACE = stop, ESC = quit")
        print("  M = toggle motion mode (play recorded motions)")
        print("  N = next motion, P = previous motion")
        
    def load_motion_data(self, max_motions):
        """Load motion data for reference and playback"""
        print(f"🎭 Loading motion data (max {max_motions})...")
        
        # Find interesting motion files (prioritize locomotion and actions)
        motion_patterns = [
            "myo_data/**/target_*Walk*.h5",
            "myo_data/**/target_*Run*.h5", 
            "myo_data/**/target_*Jump*.h5",
            "myo_data/**/target_*Dance*.h5",
            "myo_data/**/target_*Move*.h5",
            "myo_data/**/target_*.h5"  # All others
        ]
        
        motion_files = []
        for pattern in motion_patterns:
            files = glob.glob(pattern, recursive=True)
            motion_files.extend(files)
            if len(motion_files) >= max_motions * 3:  # Get extras to filter
                break
        
        # Remove duplicates and limit
        motion_files = list(set(motion_files))
        random.shuffle(motion_files)
        motion_files = motion_files[:max_motions]
        
        print(f"📁 Loading {len(motion_files)} motion files...")
        
        self.motions = []
        for i, motion_file in enumerate(motion_files):
            try:
                # Extract motion name
                filename = os.path.basename(motion_file)
                motion_name = filename.replace("target_", "").replace(".h5", "")
                if motion_name.endswith(("_00", "_01", "_02", "_03", "_04")):
                    motion_name = motion_name[:-3]
                
                # Load trajectory
                trajectory = mjTrajectory(self.model, name=motion_name)
                trajectory.load(motion_file, motion_name)
                
                # Store motion data
                motion_data = {
                    'name': motion_name,
                    'trajectory': trajectory,
                    'duration': trajectory.time[-1],
                    'frames': trajectory.horizon,
                    'qpos': trajectory.qpos,
                    'qvel': trajectory.qvel,
                    'time': trajectory.time
                }
                
                self.motions.append(motion_data)
                
                if (i + 1) % 20 == 0:
                    print(f"  Loaded {i + 1}/{len(motion_files)} motions...")
                    
            except Exception as e:
                print(f"  ❌ Failed {motion_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(self.motions)} motions successfully")
        
        if self.motions:
            self.current_motion = self.motions[0]
            self.motion_frame = 0
            print(f"🎬 Current motion: {self.current_motion['name']}")
        
    def load_policy(self, policy_path):
        """Load the trained policy"""
        from train_keyboard_control import PPOPolicy
        
        # Calculate observation dimensions
        joint_pos_size = self.model.nq - 7
        joint_vel_size = self.model.nv - 6
        num_obs = 3 + 3 + 3 + 3 + joint_pos_size + joint_vel_size
        
        policy = PPOPolicy(num_obs, self.model.nu).to(self.device)
        
        if Path(policy_path).exists():
            state_dict = torch.load(policy_path, map_location=self.device)
            policy.load_state_dict(state_dict)
            print(f"✅ Loaded policy from {policy_path}")
        else:
            print(f"⚠️ Policy file {policy_path} not found, using random policy")
            
        policy.eval()
        return policy
    
    def get_observation(self):
        """Get current observation"""
        # Base velocity (3)
        base_vel = self.data.qvel[:3]
        
        # Base angular velocity (3)
        base_angvel = self.data.qvel[3:6]
        
        # Gravity vector (3)
        gravity = np.array([0, 0, -9.81])
        
        # Current velocity commands (3)
        command_vel = self.velocity_commands.copy()
        
        # Joint positions (skip root)
        joint_pos = self.data.qpos[7:]
        
        # Joint velocities (skip root)
        joint_vel = self.data.qvel[6:]
        
        # Combine observation
        obs = np.concatenate([
            base_vel, base_angvel, gravity, command_vel,
            joint_pos, joint_vel
        ])
        
        return obs
    
    def apply_motion_frame(self, motion, frame_idx):
        """Apply a specific frame from motion data"""
        if frame_idx >= motion['frames']:
            frame_idx = motion['frames'] - 1
            
        # Set position and velocity from motion data
        self.data.qpos[:] = motion['qpos'][frame_idx]
        self.data.qvel[:] = motion['qvel'][frame_idx]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def next_motion(self):
        """Switch to next motion"""
        if not self.motions:
            return
            
        current_idx = self.motions.index(self.current_motion)
        next_idx = (current_idx + 1) % len(self.motions)
        self.current_motion = self.motions[next_idx]
        self.motion_frame = 0
        print(f"🎬 Switched to: {self.current_motion['name']}")
    
    def prev_motion(self):
        """Switch to previous motion"""
        if not self.motions:
            return
            
        current_idx = self.motions.index(self.current_motion)
        prev_idx = (current_idx - 1) % len(self.motions)
        self.current_motion = self.motions[prev_idx]
        self.motion_frame = 0
        print(f"🎬 Switched to: {self.current_motion['name']}")
    
    def control_loop(self):
        """Main control loop at 50Hz"""
        dt = 0.02  # 50Hz
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("🎮 Motion-Enhanced Demo launched!")
            print(f"Mode: {'Motion Playback' if self.motion_mode else 'Keyboard Control'}")
            
            while self.running and viewer.is_running():
                start_time = time.time()
                
                if self.motion_mode and self.current_motion:
                    # Motion playback mode
                    self.apply_motion_frame(self.current_motion, self.motion_frame)
                    
                    # Advance frame
                    self.motion_frame += 1
                    if self.motion_frame >= self.current_motion['frames']:
                        self.motion_frame = 0  # Loop motion
                        
                else:
                    # Keyboard control mode
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
        """Keyboard input thread with enhanced controls"""
        try:
            import keyboard
            print("Using keyboard library for enhanced input")
            
            while self.running:
                # Motion mode toggle
                if keyboard.is_pressed('m'):
                    self.motion_mode = not self.motion_mode
                    mode_name = "Motion Playback" if self.motion_mode else "Keyboard Control"
                    print(f"🔄 Switched to: {mode_name}")
                    time.sleep(0.3)  # Prevent rapid toggling
                
                # Motion navigation
                if keyboard.is_pressed('n'):
                    self.next_motion()
                    time.sleep(0.3)
                elif keyboard.is_pressed('p'):
                    self.prev_motion()
                    time.sleep(0.3)
                
                # Regular keyboard controls (only in keyboard mode)
                if not self.motion_mode:
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
            print("keyboard library not available, using simplified manual control")
            while self.running:
                try:
                    print(f"\\nCurrent mode: {'Motion' if self.motion_mode else 'Control'}")
                    if self.current_motion:
                        print(f"Current motion: {self.current_motion['name']}")
                    cmd = input("Command (w/s/a/d/m/n/p/q): ").lower()
                    
                    if cmd == 'm':
                        self.motion_mode = not self.motion_mode
                        mode_name = "Motion Playback" if self.motion_mode else "Keyboard Control"
                        print(f"Switched to: {mode_name}")
                    elif cmd == 'n':
                        self.next_motion()
                    elif cmd == 'p':
                        self.prev_motion()
                    elif cmd == 'w':
                        self.velocity_commands[0] = 1.0
                    elif cmd == 's':
                        self.velocity_commands[0] = -1.0
                    elif cmd == 'a':
                        self.velocity_commands[2] = 1.0
                    elif cmd == 'd':
                        self.velocity_commands[2] = -1.0
                    elif cmd == 'q':
                        self.running = False
                        break
                    else:
                        self.velocity_commands[:] = 0.0
                        
                except KeyboardInterrupt:
                    self.running = False
                    break
    
    def run(self):
        """Run the enhanced demo"""
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
            print("Motion-enhanced demo ended")

def main():
    """Main function"""
    print("🎭 MyoDolores Motion-Enhanced Keyboard Demo")
    print("=" * 60)
    print("🎯 Features:")
    print("  • Real-time keyboard control")
    print("  • Motion data playback from 277GB dataset")
    print("  • Toggle between control and motion modes")
    print("  • Browse through different motion clips")
    print("=" * 60)
    
    # Create and run demo
    demo = MotionEnhancedKeyboardDemo(max_motions=50)
    demo.run()

if __name__ == "__main__":
    main()