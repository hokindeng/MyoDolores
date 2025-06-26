#!/usr/bin/env python3
"""
MyoSkeleton Integration Test
Tests MyoSkeleton model + motion data integration for imitation learning
"""

import os
import sys
import torch
import argparse
import glob
from pathlib import Path

# Add necessary paths
sys.path.append('submodules/myo_model_internal')
sys.path.append('submodules/myo_api')

# MyoSkeleton imports
from myo_api.mj.motion.trajectory import mjTrajectory
from myo_model.utils.model_utils import get_model_xml_path
import mujoco


class MyoSkeletonIntegrationTest:
    """
    Test MyoSkeleton integration for imitation learning readiness
    """
    
    def __init__(self, max_motions: int = 10):
        """Initialize the integration test"""
        self.max_motions = max_motions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize components
        self._setup_model()
        self._setup_motion_data()
    
    def _setup_model(self):
        """Setup MyoSkeleton model"""
        print("\n🦴 Setting up MyoSkeleton model...")
        
        try:
            model_path = get_model_xml_path('motors')
            self.model = mujoco.MjModel.from_xml_path(model_path)
            print(f"✅ Model loaded: {self.model.nq} DOFs, {self.model.nu} actuators")
            print(f"   Model path: {model_path}")
            
        except Exception as e:
            print(f"❌ Model setup failed: {e}")
            raise
    
    def _setup_motion_data(self):
        """Setup and load motion data"""
        print(f"\n🎭 Setting up motion data (max {self.max_motions} motions)...")
        
        # Find motion files
        motion_files = glob.glob('myo_data/**/target_*.h5', recursive=True)
        if self.max_motions:
            motion_files = motion_files[:self.max_motions]
        
        print(f"📁 Found {len(motion_files)} motion files")
        
        # Load motions
        self.motions = []
        for i, motion_file in enumerate(motion_files):
            try:
                # Extract motion name
                filename = os.path.basename(motion_file)
                motion_name = filename.replace('target_', '').replace('.h5', '')
                if motion_name.endswith(('_00', '_01', '_02', '_03', '_04')):
                    motion_name = motion_name[:-3]
                
                # Load trajectory
                traj_loader = mjTrajectory(self.model)
                traj_loader.load(motion_file, motion_name)
                
                # Convert to tensors
                motion_data = {
                    'name': motion_name,
                    'file': motion_file,
                    'time': torch.tensor(traj_loader.time, dtype=torch.float32, device=self.device),
                    'qpos': torch.tensor(traj_loader.qpos, dtype=torch.float32, device=self.device),
                    'qvel': torch.tensor(traj_loader.qvel, dtype=torch.float32, device=self.device),
                    'duration': traj_loader.time[-1],
                    'frames': len(traj_loader.time),
                    'fps': len(traj_loader.time) / traj_loader.time[-1]
                }
                
                self.motions.append(motion_data)
                print(f"  ✅ {motion_name}: {motion_data['frames']} frames, {motion_data['duration']:.2f}s, {motion_data['fps']:.1f} Hz")
                
            except Exception as e:
                print(f"  ❌ Failed {motion_file}: {e}")
        
        print(f"✅ Loaded {len(self.motions)} motions successfully")
    
    def test_motion_sampling(self, num_samples: int = 5):
        """Test motion sampling for imitation learning"""
        print(f"\n🎲 Testing motion sampling ({num_samples} samples)...")
        
        if not self.motions:
            print("❌ No motions available for sampling")
            return False
        
        try:
            # Sample random motions and times
            samples = []
            for i in range(num_samples):
                # Random motion
                motion_idx = torch.randint(0, len(self.motions), (1,)).item()
                motion = self.motions[motion_idx]
                
                # Random time within motion
                time = torch.rand(1).item() * motion['duration']
                
                # Find closest frame
                time_diff = torch.abs(motion['time'] - time)
                frame_idx = torch.argmin(time_diff).item()
                
                sample = {
                    'motion_idx': motion_idx,
                    'motion_name': motion['name'],
                    'time': time,
                    'frame_idx': frame_idx,
                    'qpos': motion['qpos'][frame_idx],
                    'qvel': motion['qvel'][frame_idx]
                }
                samples.append(sample)
                
                print(f"  Sample {i+1}: {sample['motion_name']} @ {sample['time']:.2f}s")
            
            # Test batch sampling
            motion_ids = torch.randint(0, len(self.motions), (num_samples,))
            times = torch.rand(num_samples) * 2.0  # Random times 0-2s
            
            batch_qpos = []
            batch_qvel = []
            
            for motion_id, time in zip(motion_ids, times):
                motion = self.motions[motion_id]
                # Clamp time to motion duration
                time = min(time, motion['duration'])
                
                time_diff = torch.abs(motion['time'] - time)
                frame_idx = torch.argmin(time_diff)
                
                batch_qpos.append(motion['qpos'][frame_idx])
                batch_qvel.append(motion['qvel'][frame_idx])
            
            batch_qpos = torch.stack(batch_qpos)
            batch_qvel = torch.stack(batch_qvel)
            
            print(f"✅ Batch sampling: QPos {batch_qpos.shape}, QVel {batch_qvel.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Motion sampling failed: {e}")
            return False
    
    def test_reward_computation(self):
        """Test basic reward computation for imitation learning"""
        print(f"\n🎯 Testing reward computation...")
        
        if not self.motions:
            print("❌ No motions available for reward testing")
            return False
        
        try:
            # Get reference and current states
            motion = self.motions[0]
            ref_qpos = motion['qpos'][10]  # Reference pose
            current_qpos = motion['qpos'][12] + torch.randn_like(motion['qpos'][12]) * 0.1  # Noisy current pose
            
            # Compute simple pose tracking reward
            pose_diff = torch.norm(ref_qpos - current_qpos)
            pose_reward = torch.exp(-2.0 * pose_diff)
            
            # Compute velocity tracking reward
            ref_qvel = motion['qvel'][10]
            current_qvel = motion['qvel'][12] + torch.randn_like(motion['qvel'][12]) * 0.1
            
            vel_diff = torch.norm(ref_qvel - current_qvel)
            vel_reward = torch.exp(-0.1 * vel_diff)
            
            # Combined reward
            total_reward = 0.8 * pose_reward + 0.2 * vel_reward
            
            print(f"  📊 Pose difference: {pose_diff:.4f}, Reward: {pose_reward:.4f}")
            print(f"  📊 Velocity difference: {vel_diff:.4f}, Reward: {vel_reward:.4f}")
            print(f"  📊 Total reward: {total_reward:.4f}")
            
            print("✅ Reward computation working")
            return True
            
        except Exception as e:
            print(f"❌ Reward computation failed: {e}")
            return False
    
    def test_motion_statistics(self):
        """Analyze motion data statistics"""
        print(f"\n📊 Analyzing motion statistics...")
        
        if not self.motions:
            print("❌ No motions available for analysis")
            return False
        
        try:
            # Collect all motion data
            all_qpos = torch.cat([m['qpos'] for m in self.motions], dim=0)
            all_qvel = torch.cat([m['qvel'] for m in self.motions], dim=0)
            
            # Compute statistics
            qpos_mean = all_qpos.mean(dim=0)
            qpos_std = all_qpos.std(dim=0)
            qvel_mean = all_qvel.mean(dim=0)
            qvel_std = all_qvel.std(dim=0)
            
            # Motion durations
            durations = [m['duration'] for m in self.motions]
            frame_counts = [m['frames'] for m in self.motions]
            
            print(f"  📈 Total frames: {all_qpos.shape[0]}")
            print(f"  📈 Position range: [{all_qpos.min():.3f}, {all_qpos.max():.3f}]")
            print(f"  📈 Velocity range: [{all_qvel.min():.3f}, {all_qvel.max():.3f}]")
            print(f"  📈 Duration range: [{min(durations):.2f}s, {max(durations):.2f}s]")
            print(f"  📈 Avg duration: {sum(durations)/len(durations):.2f}s")
            print(f"  📈 Frame range: [{min(frame_counts)}, {max(frame_counts)}]")
            
            print("✅ Motion statistics computed")
            return True
            
        except Exception as e:
            print(f"❌ Motion statistics failed: {e}")
            return False
    
    def run_full_test(self):
        """Run complete integration test"""
        print("🧪 MyoSkeleton Integration Test")
        print("=" * 50)
        
        results = {
            'model_setup': True,  # Already verified in __init__
            'motion_data': len(self.motions) > 0,
            'motion_sampling': self.test_motion_sampling(),
            'reward_computation': self.test_reward_computation(),
            'motion_statistics': self.test_motion_statistics()
        }
        
        print(f"\n📋 Test Results:")
        print(f"  ✅ Model Setup: {results['model_setup']}")
        print(f"  ✅ Motion Data: {results['motion_data']} ({len(self.motions)} motions)")
        print(f"  ✅ Motion Sampling: {results['motion_sampling']}")
        print(f"  ✅ Reward Computation: {results['reward_computation']}")
        print(f"  ✅ Motion Statistics: {results['motion_statistics']}")
        
        all_passed = all(results.values())
        print(f"\n🎉 Overall Result: {'PASS' if all_passed else 'FAIL'}")
        
        if all_passed:
            print("\n🚀 MyoSkeleton is ready for ASAP imitation learning!")
            print("\nNext steps:")
            print("  1. Increase max_motions to use full dataset")
            print("  2. Integrate with full ASAP framework")
            print("  3. Configure motion tracking environment")
            print("  4. Start training with AMP algorithm")
        
        return results


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="MyoSkeleton Integration Test")
    parser.add_argument("--max-motions", type=int, default=10,
                        help="Maximum number of motions to test")
    
    args = parser.parse_args()
    
    try:
        # Run integration test
        test = MyoSkeletonIntegrationTest(max_motions=args.max_motions)
        results = test.run_full_test()
        
        # Exit with appropriate code
        if all(results.values()):
            exit(0)
        else:
            exit(1)
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main() 