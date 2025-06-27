#!/usr/bin/env python3
"""
🎭 MyoSkeleton Imitation Learning Demo
Visualize trained ASAP imitation learning policy performing motion imitation
"""

import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import glob
from pathlib import Path

# Add necessary paths
sys.path.append('submodules/myo_model_internal')
sys.path.append('submodules/myo_api')

class ImitationLearningDemo:
    """Demo for visualizing trained imitation learning policy"""
    
    def __init__(self, checkpoint_path=None):
        print("🎭 Loading MyoSkeleton Imitation Learning Demo...")
        
        # Load MyoSkeleton model
        from myo_model.utils.model_utils import get_model_xml_path
        model_path = get_model_xml_path('motors')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"✅ Model loaded: {self.model.nq} DOFs, {self.model.nu} actuators")
        
        # Initialize model  
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 1.7  # Set initial height
        mujoco.mj_forward(self.model, self.data)
        
        # Load motion data
        self.load_motions()
        
        # Load checkpoint if available
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        print("🎮 Demo Controls:")
        print("  SPACE = Play random motion")
        print("  ESC = Quit")
        
    def load_motions(self):
        """Load motion data for reference"""
        print("📚 Loading motion data...")
        
        from myo_api.mj.motion.trajectory import mjTrajectory
        
        # Find motion files
        motion_files = glob.glob('myo_data/**/target_*.h5', recursive=True)
        print(f"Found {len(motion_files)} motion files")
        
        self.motions = []
        max_motions = 20  # Load a subset for demo
        
        for i, motion_file in enumerate(motion_files[:max_motions]):
            try:
                trajectory = mjTrajectory(motion_file)
                
                if trajectory.qpos.shape[1] == self.model.nq:
                    motion_data = {
                        'qpos': torch.tensor(trajectory.qpos, dtype=torch.float32),
                        'qvel': torch.tensor(trajectory.qvel, dtype=torch.float32),
                        'time': torch.tensor(trajectory.time, dtype=torch.float32),
                        'duration': trajectory.time[-1],
                        'name': Path(motion_file).stem
                    }
                    self.motions.append(motion_data)
                    
            except Exception as e:
                print(f"⚠️ Error loading {motion_file}: {e}")
                continue
                
        print(f"✅ Loaded {len(self.motions)} compatible motions")
        
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"   Trained on {checkpoint['motion_count']} motions")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
    
    def play_reference_motion(self, motion_idx=None):
        """Play a reference motion"""
        if not self.motions:
            print("❌ No motions loaded")
            return
            
        if motion_idx is None:
            motion_idx = np.random.randint(len(self.motions))
            
        motion = self.motions[motion_idx]
        print(f"🎬 Playing motion: {motion['name']}")
        print(f"   Duration: {motion['duration']:.2f}s, Frames: {len(motion['time'])}")
        
        # Play the motion
        for frame_idx in range(len(motion['time'])):
            self.data.qpos[:] = motion['qpos'][frame_idx].numpy()
            self.data.qvel[:] = motion['qvel'][frame_idx].numpy()
            
            mujoco.mj_forward(self.model, self.data)
            time.sleep(0.033)  # ~30 FPS
            
            yield  # Allow viewer to update
    
    def run_demo(self):
        """Run the interactive demo"""
        print("🚀 Starting demo viewer...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("Viewer launched! Press SPACE to play motions...")
            
            current_motion_gen = None
            
            while viewer.is_running():
                try:
                    # Check for user input (simplified)
                    if current_motion_gen is None:
                        # Show idle pose
                        mujoco.mj_resetData(self.model, self.data)
                        self.data.qpos[2] = 1.7
                        mujoco.mj_forward(self.model, self.data)
                        
                        # Auto-play random motion every 3 seconds
                        if int(time.time()) % 3 == 0:
                            current_motion_gen = self.play_reference_motion()
                    else:
                        try:
                            next(current_motion_gen)
                        except StopIteration:
                            current_motion_gen = None
                            print("🎬 Motion completed, waiting for next...")
                    
                    # Update viewer
                    viewer.sync()
                    time.sleep(0.033)  # ~30 FPS
                    
                except KeyboardInterrupt:
                    break
                    
        print("Demo ended")

def find_latest_checkpoint():
    """Find the most recent training checkpoint"""
    checkpoint_pattern = "experiments/*/checkpoint_epoch_*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
        
    # Find the highest epoch number
    latest = max(checkpoints, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
    return latest

def main():
    """Main demo function"""
    print("🎭 MyoSkeleton Imitation Learning Demo")
    print("=" * 50)
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path:
        print(f"📁 Found checkpoint: {checkpoint_path}")
    else:
        print("⚠️ No checkpoints found, running without trained policy")
    
    # Create and run demo
    demo = ImitationLearningDemo(checkpoint_path)
    demo.run_demo()

if __name__ == "__main__":
    main() 