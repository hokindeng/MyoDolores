#!/usr/bin/env python3
"""
MyoDolores Keyboard Control Training
Simple PPO training for velocity command following with MyoSkeleton
"""

import torch
import torch.nn as nn
import numpy as np
import mujoco
import time
import sys
import os
from pathlib import Path

# Add myo_api to path
sys.path.append(str(Path(__file__).parent / "myo_api"))
import myo_api as myo

class KeyboardControlEnv:
    """Simple MuJoCo environment for velocity control training"""
    
    def __init__(self, model_path, num_envs=256, device='cuda'):
        self.num_envs = num_envs
        self.device = device
        
        # Load MyoSkeleton model
        print(f"Loading model: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        
        # Action and observation dimensions
        self.num_actions = self.model.nu  # Number of actuators
        # Calculate actual observation size: vel(3) + angvel(3) + gravity(3) + commands(3) + joint_pos(nq-7) + joint_vel(nv-6)
        joint_pos_size = self.model.nq - 7  # Skip root position/orientation
        joint_vel_size = self.model.nv - 6  # Skip root velocity  
        self.num_obs = 3 + 3 + 3 + 3 + joint_pos_size + joint_vel_size
        
        print(f"Environment: {num_envs} envs, {self.num_actions} actions, {self.num_obs} observations")
        print(f"Model info: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
        print(f"Joint sizes: pos={joint_pos_size}, vel={joint_vel_size}")
        
        # Control parameters
        self.dt = self.model.opt.timestep
        self.max_episode_length = 1000
        self.episode_length = np.zeros(num_envs)
        
        # Rewards
        self.reward_scales = {
            'velocity_tracking': 2.0,
            'upright': 1.0,
            'joint_limits': -0.1,
            'energy': -0.001
        }
        
    def reset(self, env_ids=None):
        """Reset specified environments"""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
            
        for i in env_ids:
            mujoco.mj_resetData(self.model, self.data[i])
            # Set initial position (standing)
            self.data[i].qpos[2] = 1.7  # Height for MyoSkeleton
            mujoco.mj_forward(self.model, self.data[i])
            self.episode_length[i] = 0
            
        return self._get_observations(env_ids)
    
    def step(self, actions, commands):
        """Step the simulation forward"""
        actions = np.array(actions)
        commands = np.array(commands)  # [num_envs, 3] - [vel_x, vel_y, ang_vel_z]
        
        # Apply actions
        for i in range(self.num_envs):
            self.data[i].ctrl[:] = actions[i]
            mujoco.mj_step(self.model, self.data[i])
            self.episode_length[i] += 1
            
        # Get observations
        obs = self._get_observations()
        
        # Calculate rewards
        rewards = self._calculate_rewards(commands)
        
        # Check termination
        dones = self._check_termination()
        
        return obs, rewards, dones
    
    def _get_observations(self, env_ids=None):
        """Get observations for all environments"""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
            
        obs = np.zeros((len(env_ids), self.num_obs))
        
        for idx, i in enumerate(env_ids):
            data = self.data[i]
            
            # Base velocity (3)
            base_vel = data.qvel[:3]
            
            # Base angular velocity (3) 
            base_angvel = data.qvel[3:6]
            
            # Gravity vector in base frame (3)
            gravity = np.array([0, 0, -9.81])
            
            # Command velocities (filled by training loop) (3)
            command_vel = np.zeros(3)
            
            # Joint positions (nq)
            joint_pos = data.qpos[7:]  # Skip root position/orientation
            
            # Joint velocities (nv-6)
            joint_vel = data.qvel[6:]  # Skip root velocity
            
            # Combine observations
            obs[idx] = np.concatenate([
                base_vel, base_angvel, gravity, command_vel,
                joint_pos, joint_vel
            ])
            
        return obs
    
    def _calculate_rewards(self, commands):
        """Calculate rewards for velocity tracking"""
        rewards = np.zeros(self.num_envs)
        
        for i in range(self.num_envs):
            data = self.data[i]
            command = commands[i]
            
            # Velocity tracking reward
            actual_vel = data.qvel[:3]
            target_vel = command
            vel_error = np.linalg.norm(actual_vel[:2] - target_vel[:2])  # xy velocity
            vel_reward = np.exp(-2 * vel_error) * self.reward_scales['velocity_tracking']
            
            # Upright reward  
            upright = data.qpos[2]  # z position
            upright_reward = (upright - 1.0) * self.reward_scales['upright']
            
            # Energy penalty
            energy_penalty = np.sum(np.square(data.ctrl)) * self.reward_scales['energy']
            
            rewards[i] = vel_reward + upright_reward + energy_penalty
            
        return rewards
    
    def _check_termination(self):
        """Check if episodes should terminate"""
        dones = np.zeros(self.num_envs, dtype=bool)
        
        for i in range(self.num_envs):
            data = self.data[i]
            
            # Fallen over
            if data.qpos[2] < 0.8:  
                dones[i] = True
                
            # Episode timeout
            if self.episode_length[i] >= self.max_episode_length:
                dones[i] = True
                
        return dones


class PPOPolicy(nn.Module):
    """Simple PPO policy network"""
    
    def __init__(self, num_obs, num_actions, hidden_size=256):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(num_obs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_obs, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(num_actions))
        
    def forward(self, obs):
        """Forward pass returning action mean and value"""
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
    
    def get_action(self, obs):
        """Sample actions from policy"""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value


def train_keyboard_control():
    """Main training function"""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 256  # Reduced for Tesla T4
    learning_rate = 3e-4
    num_iterations = 1000
    
    print(f"Training on device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # Change to myo_model directory to resolve relative paths
    original_dir = os.getcwd()
    model_dir = Path("myo_model_internal/myo_model/")
    model_file = "myoskeleton/myoskeleton_with_motors.xml"
    
    if not (model_dir / model_file).exists():
        print(f"Error: Model not found at {model_dir / model_file}")
        return
    
    os.chdir(model_dir)
    print(f"Changed to model directory: {os.getcwd()}")
    model_path = model_file
        
    # Create environment
    env = KeyboardControlEnv(model_path, num_envs=num_envs, device=device)
    
    # Create policy
    policy = PPOPolicy(env.num_obs, env.num_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    print(f"Starting training: {num_iterations} iterations")
    
    # Training loop
    obs = env.reset()
    
    for iteration in range(num_iterations):
        # Generate random velocity commands (keyboard simulation)
        commands = np.random.uniform(-1, 1, (num_envs, 3))
        commands[:, 2] *= 0.5  # Reduce angular velocity range
        
        # Update observations with commands
        obs[:, 9:12] = commands  # Command indices in observation
        
        # Convert to torch
        obs_torch = torch.FloatTensor(obs).to(device)
        
        # Get actions
        with torch.no_grad():
            actions, log_probs, values = policy.get_action(obs_torch)
            
        # Step environment
        next_obs, rewards, dones = env.step(actions.cpu().numpy(), commands)
        
        # Reset done environments
        if np.any(dones):
            reset_ids = np.where(dones)[0]
            reset_obs = env.reset(reset_ids)
            next_obs[reset_ids] = reset_obs
            
        obs = next_obs
        
        # Simple logging
        if iteration % 50 == 0:
            avg_reward = np.mean(rewards)
            print(f"Iteration {iteration:4d}: Avg Reward = {avg_reward:8.3f}")
            
        # Save model periodically
        if iteration % 200 == 0 and iteration > 0:
            torch.save(policy.state_dict(), f'keyboard_policy_{iteration}.pt')
            print(f"Saved model at iteration {iteration}")
    
    # Final save
    torch.save(policy.state_dict(), 'keyboard_policy_final.pt')
    print("Training complete! Saved final model as 'keyboard_policy_final.pt'")
    
    # Restore original directory
    os.chdir(original_dir)


if __name__ == "__main__":
    train_keyboard_control()