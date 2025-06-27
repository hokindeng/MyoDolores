#!/usr/bin/env python3
"""
Test Motion Data Loading with myo_api
Demonstrates how to properly load and use motion data from myo_data directory
"""

import sys
import glob
import numpy as np
import torch

# Add necessary paths
sys.path.append('submodules/myo_model_internal')
sys.path.append('submodules/myo_api')

from myo_model.utils.model_utils import get_model_xml_path
from myo_api.mj.motion.trajectory import mjTrajectory
import mujoco

class MotionDataLoader:
    """Loads and manages motion data for training"""
    
    def __init__(self, data_path="myo_data", max_motions=10):
        print("🎭 Initializing Motion Data Loader...")
        
        # Load MyoSkeleton model
        self.model_path = get_model_xml_path('motors')
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        print(f"🦴 MyoSkeleton: {self.model.nq} DOFs, {self.model.nu} actuators")
        
        # Find motion files
        self.motion_files = self._find_motion_files(data_path, max_motions)
        print(f"📁 Found {len(self.motion_files)} motion files")
        
        # Load motions
        self.motions = self._load_motions()
        print(f"✅ Loaded {len(self.motions)} motions successfully")
        
    def _find_motion_files(self, data_path, max_motions):
        """Find motion files to load"""
        pattern = f"{data_path}/**/target_*.h5"
        files = glob.glob(pattern, recursive=True)
        files.sort()
        
        if max_motions:
            files = files[:max_motions]
            
        return files
    
    def _load_motions(self):
        """Load motion trajectories"""
        motions = []
        
        for motion_file in self.motion_files:
            try:
                # Extract motion name
                filename = motion_file.split('/')[-1]
                motion_name = filename.replace("target_", "").replace(".h5", "")
                if motion_name.endswith(("_00", "_01", "_02")):
                    motion_name = motion_name[:-3]
                
                # Load trajectory
                trajectory = mjTrajectory(self.model, name=motion_name)
                trajectory.load(motion_file, motion_name)
                
                # Store motion data
                motion_data = {
                    'name': motion_name,
                    'file': motion_file,
                    'trajectory': trajectory,
                    'duration': trajectory.time[-1],
                    'frames': trajectory.horizon,
                    'fps': trajectory.horizon / trajectory.time[-1],
                    'qpos': torch.tensor(trajectory.qpos, dtype=torch.float32),
                    'qvel': torch.tensor(trajectory.qvel, dtype=torch.float32),
                    'time': torch.tensor(trajectory.time, dtype=torch.float32)
                }
                
                motions.append(motion_data)
                print(f"  ✅ {motion_name}: {motion_data['frames']} frames, {motion_data['duration']:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Failed {motion_file}: {e}")
                continue
        
        return motions
    
    def get_motion_statistics(self):
        """Analyze motion data statistics"""
        if not self.motions:
            return {}
            
        durations = [m['duration'] for m in self.motions]
        frame_counts = [m['frames'] for m in self.motions]
        fps_values = [m['fps'] for m in self.motions]
        
        # Concatenate all qpos data for analysis
        all_qpos = torch.cat([m['qpos'] for m in self.motions], dim=0)
        all_qvel = torch.cat([m['qvel'] for m in self.motions], dim=0)
        
        stats = {
            'num_motions': len(self.motions),
            'total_frames': all_qpos.shape[0],
            'total_duration': sum(durations),
            'avg_duration': np.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_fps': np.mean(fps_values),
            'qpos_mean': all_qpos.mean(dim=0),
            'qpos_std': all_qpos.std(dim=0),
            'qvel_mean': all_qvel.mean(dim=0),
            'qvel_std': all_qvel.std(dim=0),
            'position_range': (all_qpos.min().item(), all_qpos.max().item()),
            'velocity_range': (all_qvel.min().item(), all_qvel.max().item())
        }
        
        return stats
    
    def sample_random_frames(self, num_samples=5):
        """Sample random frames from loaded motions"""
        if not self.motions:
            return []
            
        samples = []
        for _ in range(num_samples):
            # Random motion
            motion = np.random.choice(self.motions)
            
            # Random frame
            frame_idx = np.random.randint(0, motion['frames'])
            
            sample = {
                'motion_name': motion['name'],
                'frame_idx': frame_idx,
                'time': motion['time'][frame_idx].item(),
                'qpos': motion['qpos'][frame_idx],
                'qvel': motion['qvel'][frame_idx]
            }
            samples.append(sample)
            
        return samples

def main():
    """Test the motion data loading interface"""
    print("🎭 MyoDolores Motion Data Loading Test")
    print("=" * 50)
    
    # Load motion data
    loader = MotionDataLoader(max_motions=20)
    
    if not loader.motions:
        print("❌ No motions loaded!")
        return
    
    # Show statistics
    print("\n📊 Motion Statistics:")
    stats = loader.get_motion_statistics()
    
    print(f"  Motions loaded: {stats['num_motions']}")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Total duration: {stats['total_duration']:.1f}s")
    print(f"  Average duration: {stats['avg_duration']:.2f}s")
    print(f"  Duration range: [{stats['min_duration']:.2f}s, {stats['max_duration']:.2f}s]")
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Position range: [{stats['position_range'][0]:.3f}, {stats['position_range'][1]:.3f}]")
    print(f"  Velocity range: [{stats['velocity_range'][0]:.3f}, {stats['velocity_range'][1]:.3f}]")
    
    # Sample some random frames
    print("\n🎲 Random Motion Samples:")
    samples = loader.sample_random_frames(5)
    
    for i, sample in enumerate(samples):
        print(f"  Sample {i+1}: {sample['motion_name']} @ frame {sample['frame_idx']} (t={sample['time']:.2f}s)")
        print(f"    QPos shape: {sample['qpos'].shape}, QVel shape: {sample['qvel'].shape}")
    
    print("\n🎉 Motion data loading test completed!")
    print("\nReady for integration with:")
    print("  • ASAP motion tracking training")
    print("  • Imitation learning algorithms")
    print("  • Real-time motion following")

if __name__ == "__main__":
    main()