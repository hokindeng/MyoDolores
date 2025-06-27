#!/usr/bin/env python3
"""
Real Motion Imitation Training for MyoSkeleton
Uses the actual motion data from myo_data for imitation learning
Works without Isaac Gym - uses MuJoCo directly with your motion data
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import mujoco
import random
import time
from pathlib import Path
import glob
from datetime import datetime

# Add paths
sys.path.append('submodules/myo_model_internal')
sys.path.append('submodules/myo_api')

from myo_model.utils.model_utils import get_model_xml_path
from myo_api.mj.motion.trajectory import mjTrajectory

class MotionImitationPolicy(nn.Module):
    """Neural network policy for motion imitation"""
    
    def __init__(self, obs_dim, action_dim, hidden_size=512):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs):
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
    
    def get_action(self, obs):
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std.clamp(-20, 2))
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

class MotionDataset:
    """Dataset of motion sequences for imitation learning"""
    
    def __init__(self, data_path="myo_data", max_motions=1000, device='cpu'):
        self.device = device
        print(f"🎭 Loading motion dataset from {data_path}...")
        
        # Load MyoSkeleton model
        model_path = get_model_xml_path('motors')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        print(f"🦴 MyoSkeleton: {self.model.nq} DOFs, {self.model.nu} actuators")
        
        # Find motion files
        pattern = f"{data_path}/**/target_*.h5"
        motion_files = glob.glob(pattern, recursive=True)
        random.shuffle(motion_files)
        
        if max_motions:
            motion_files = motion_files[:max_motions]
        
        print(f"📁 Loading {len(motion_files)} motion files...")
        
        # Load motions
        self.motions = []
        self.total_frames = 0
        
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
                
                # Convert to tensors
                motion_data = {
                    'name': motion_name,
                    'qpos': torch.tensor(trajectory.qpos, dtype=torch.float32, device=device),
                    'qvel': torch.tensor(trajectory.qvel, dtype=torch.float32, device=device),
                    'time': torch.tensor(trajectory.time, dtype=torch.float32, device=device),
                    'duration': trajectory.time[-1],
                    'frames': trajectory.horizon
                }
                
                self.motions.append(motion_data)
                self.total_frames += trajectory.horizon
                
                if (i + 1) % 100 == 0:
                    print(f"  Loaded {i + 1}/{len(motion_files)} motions...")
                    
            except Exception as e:
                print(f"  ❌ Failed {motion_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(self.motions)} motions, {self.total_frames} total frames")
        
        # Compute motion statistics
        all_qpos = torch.cat([m['qpos'] for m in self.motions], dim=0)
        all_qvel = torch.cat([m['qvel'] for m in self.motions], dim=0)
        
        self.qpos_mean = all_qpos.mean(dim=0)
        self.qpos_std = all_qpos.std(dim=0) + 1e-8
        self.qvel_mean = all_qvel.mean(dim=0)
        self.qvel_std = all_qvel.std(dim=0) + 1e-8
        
        print(f"📊 Dataset statistics computed")
    
    def sample_batch(self, batch_size):
        """Sample a batch of motion states for training"""
        batch_current = []
        batch_target = []
        
        for _ in range(batch_size):
            # Sample random motion
            motion = random.choice(self.motions)
            
            # Sample random frame
            if motion['frames'] < 2:
                continue
                
            frame_idx = random.randint(0, motion['frames'] - 2)
            
            # Current state
            current_qpos = motion['qpos'][frame_idx]
            current_qvel = motion['qvel'][frame_idx]
            
            # Target state (next frame)
            target_qpos = motion['qpos'][frame_idx + 1]
            target_qvel = motion['qvel'][frame_idx + 1]
            
            # Normalize
            current_qpos_norm = (current_qpos - self.qpos_mean) / self.qpos_std
            current_qvel_norm = (current_qvel - self.qvel_mean) / self.qvel_std
            target_qpos_norm = (target_qpos - self.qpos_mean) / self.qpos_std
            
            # Create observation (state + target)
            obs = torch.cat([current_qpos_norm, current_qvel_norm, target_qpos_norm])
            
            # Target action (what we want the policy to output)
            action = target_qpos - current_qpos  # Position delta
            
            batch_current.append(obs)
            batch_target.append(action)
        
        if not batch_current:
            return None, None
            
        return torch.stack(batch_current), torch.stack(batch_target)

def train_motion_imitation():
    """Train motion imitation policy"""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_motions = 2000  # Use 2000 motions for good diversity
    batch_size = 256
    learning_rate = 3e-4
    num_epochs = 100
    save_interval = 25
    
    print("🚀 Motion Imitation Training")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Max motions: {max_motions}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print("=" * 50)
    
    # Load dataset
    dataset = MotionDataset(max_motions=max_motions, device=device)
    
    if not dataset.motions:
        print("❌ No motions loaded!")
        return
    
    # Create policy
    obs_dim = dataset.model.nq * 2 + dataset.model.nq  # current_pos + current_vel + target_pos
    action_dim = dataset.model.nq
    
    policy = MotionImitationPolicy(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    print(f"🧠 Policy: {obs_dim} obs → {action_dim} actions")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Multiple batches per epoch
        for batch_idx in range(50):  # 50 batches per epoch
            obs_batch, action_batch = dataset.sample_batch(batch_size)
            
            if obs_batch is None:
                continue
            
            # Forward pass
            predicted_actions, values = policy(obs_batch)
            
            # Loss (imitation loss)
            action_loss = nn.MSELoss()(predicted_actions, action_batch)
            
            # Backward pass
            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(action_loss.item())
        
        # Logging
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                
            print(f"Epoch {epoch + 1:3d}: Loss = {avg_loss:.6f} (Best: {best_loss:.6f})")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'dataset_stats': {
                        'qpos_mean': dataset.qpos_mean,
                        'qpos_std': dataset.qpos_std,
                        'qvel_mean': dataset.qvel_mean,
                        'qvel_std': dataset.qvel_std
                    }
                }
                
                torch.save(checkpoint, f'motion_imitation_policy_epoch_{epoch + 1}.pt')
                print(f"💾 Saved checkpoint at epoch {epoch + 1}")
    
    # Save final policy
    final_checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'dataset_stats': {
            'qpos_mean': dataset.qpos_mean,
            'qpos_std': dataset.qpos_std,
            'qvel_mean': dataset.qvel_mean,
            'qvel_std': dataset.qvel_std
        },
        'training_info': {
            'motions_used': len(dataset.motions),
            'total_frames': dataset.total_frames,
            'final_loss': best_loss
        }
    }
    
    torch.save(final_checkpoint, 'motion_imitation_policy_final.pt')
    print("✅ Final policy saved: motion_imitation_policy_final.pt")
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Trained on {len(dataset.motions)} motions ({dataset.total_frames} frames)")
    print(f"🏆 Best loss: {best_loss:.6f}")
    print("\n🎮 Ready for keyboard demo with real motion data!")

if __name__ == "__main__":
    train_motion_imitation()