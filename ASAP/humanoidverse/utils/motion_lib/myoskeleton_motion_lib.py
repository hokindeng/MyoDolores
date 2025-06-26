#!/usr/bin/env python3
"""
MyoSkeleton Motion Library for ASAP Imitation Learning
Integrates retargeted motion data with ASAP's motion tracking framework
"""

import os
import glob
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import random

# ASAP motion library imports
from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase

# MyoSkeleton imports - import without circular dependency
import sys
sys.path.append('../submodules/myo_model_internal')
sys.path.append('../submodules/myo_api')

from myo_api.mj.motion.trajectory import mjTrajectory
from myo_model.utils.model_utils import get_model_xml_path
import mujoco


class MyoSkeletonMotionLib(MotionLibBase):
    """
    MyoSkeleton-specific motion library that loads retargeted motion data
    from H5 files and integrates with ASAP's imitation learning framework.
    """
    
    def __init__(self, 
                 motion_lib_cfg,
                 num_envs,
                 device,
                 motion_data_path: str = "../../myo_data",
                 model_type: str = "motors",
                 motion_file_pattern: str = "**/target_*.h5",
                 max_motions: Optional[int] = None):
        """
        Initialize MyoSkeleton motion library.
        
        Args:
            motion_lib_cfg: ASAP motion library config
            num_envs: Number of environments
            device: torch device for tensors
            motion_data_path: Path to myo_data directory
            model_type: MyoSkeleton model type ("motors", "muscles", etc.)
            motion_file_pattern: Glob pattern for motion files
            max_motions: Maximum number of motions to load (None for all)
        """
        self.motion_data_path = motion_data_path
        self.model_type = model_type
        self.motion_file_pattern = motion_file_pattern
        self.max_motions = max_motions
        
        # Load MyoSkeleton model
        self._load_myoskeleton_model()
        
        # Load motion files
        self.motion_files = self._discover_motion_files()
        if max_motions:
            self.motion_files = self.motion_files[:max_motions]
            
        print(f"🎭 Found {len(self.motion_files)} motion files")
        
        # Load and process motions
        self.motions = self._load_motions()
        
        # Set up motion data for ASAP
        self._setup_motion_data()
        
        # Initialize parent MotionLibBase with processed motion data
        super().__init__(motion_lib_cfg, num_envs, device)
    
    def _load_myoskeleton_model(self):
        """Load MyoSkeleton MuJoCo model."""
        model_path = get_model_xml_path(self.model_type)
        self.myo_model = mujoco.MjModel.from_xml_path(model_path)
        print(f"🦴 MyoSkeleton loaded: {self.myo_model.nq} DOFs, {self.myo_model.nu} actuators")
    
    def _discover_motion_files(self) -> List[str]:
        """Discover all motion files matching the pattern."""
        pattern = os.path.join(self.motion_data_path, self.motion_file_pattern)
        motion_files = glob.glob(pattern, recursive=True)
        motion_files.sort()
        return motion_files
    
    def _load_motions(self) -> List[Dict]:
        """Load all motion trajectories using myo_api."""
        motions = []
        
        for i, motion_file in enumerate(self.motion_files):
            try:
                # Extract motion name from filename
                filename = os.path.basename(motion_file)
                motion_name = filename.replace("target_", "").replace(".h5", "")
                
                # Remove file suffix (e.g., "_00", "_01")
                if motion_name.endswith(("_00", "_01", "_02", "_03", "_04")):
                    motion_name = motion_name[:-3]
                elif motion_name[-3:].isdigit():
                    motion_name = motion_name[:-3]
                
                # Load trajectory
                traj_loader = mjTrajectory(self.myo_model)
                traj_loader.load(motion_file, motion_name)
                
                # Convert to torch tensors
                motion_data = {
                    'file_path': motion_file,
                    'motion_name': motion_name,
                    'time': torch.tensor(traj_loader.time, dtype=torch.float32, device=self._device),
                    'qpos': torch.tensor(traj_loader.qpos, dtype=torch.float32, device=self._device),
                    'qvel': torch.tensor(traj_loader.qvel, dtype=torch.float32, device=self._device),
                    'dt': torch.tensor(traj_loader.time[1] - traj_loader.time[0], dtype=torch.float32, device=self._device)
                }
                
                motions.append(motion_data)
                
                if (i + 1) % 50 == 0:
                    print(f"  Loaded {i + 1}/{len(self.motion_files)} motions...")
                    
            except Exception as e:
                print(f"❌ Failed to load {motion_file}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(motions)} motions")
        return motions
    
    def _setup_motion_data(self):
        """Set up motion data in ASAP-compatible format."""
        if not self.motions:
            raise ValueError("No motions loaded!")
        
        # Convert motions to ASAP format
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        
        # Process motion data for ASAP compatibility
        all_qpos = []
        all_qvel = []
        
        for motion in self.motions:
            num_frames = motion['qpos'].shape[0]
            duration = motion['time'][-1].item()
            fps = num_frames / duration
            dt = motion['dt'].item()
            
            self._motion_lengths.append(duration)
            self._motion_fps.append(fps)
            self._motion_dt.append(dt)
            self._motion_num_frames.append(num_frames)
            
            all_qpos.append(motion['qpos'])
            all_qvel.append(motion['qvel'])
        
        # Convert to tensors for ASAP compatibility
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float32, device=self._device)
        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float32, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float32, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)
        
        # Store processed data
        self._motion_qpos = torch.cat(all_qpos, dim=0)
        self._motion_qvel = torch.cat(all_qvel, dim=0)
        
        # Motion statistics for normalization
        self.qpos_mean = self._motion_qpos.mean(dim=0)
        self.qpos_std = self._motion_qpos.std(dim=0)
        self.qvel_mean = self._motion_qvel.mean(dim=0)
        self.qvel_std = self._motion_qvel.std(dim=0)
        
        # Set up motion boundaries for sampling
        self._motion_weights = self._motion_lengths / self._motion_lengths.sum()
        
        print(f"📊 Motion library statistics:")
        print(f"   Total frames: {self._motion_qpos.shape[0]}")
        print(f"   DOFs: {self._motion_qpos.shape[1]}")
        print(f"   Motion count: {len(self.motions)}")
        
        # Set up ASAP-compatible motion indexing
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        
        # Set number of motions for ASAP
        self._num_motions = len(self.motions)
    
    def sample_motion(self, num_envs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample random motions for imitation learning.
        
        Args:
            num_envs: Number of environments (motions to sample)
            
        Returns:
            Tuple of (qpos, qvel, motion_ids)
        """
        # Sample motions weighted by length
        motion_ids = torch.multinomial(self._motion_weights, num_envs, replacement=True)
        
        sampled_qpos = []
        sampled_qvel = []
        
        for motion_id in motion_ids:
            motion = self.motions[motion_id]
            # Sample random frame from motion
            frame_id = torch.randint(0, motion['qpos'].shape[0], (1,))
            sampled_qpos.append(motion['qpos'][frame_id])
            sampled_qvel.append(motion['qvel'][frame_id])
        
        return (torch.stack(sampled_qpos).squeeze(1),
                torch.stack(sampled_qvel).squeeze(1),
                motion_ids)
    
    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> Dict:
        """
        Get motion state at specific times for tracking - ASAP compatible interface.
        
        Args:
            motion_ids: Motion indices
            motion_times: Times within motions
            
        Returns:
            Dictionary with motion state data compatible with ASAP
        """
        batch_size = motion_ids.shape[0]
        
        # Prepare output tensors
        dof_pos = torch.zeros(batch_size, self.myo_model.nq, device=self._device)
        dof_vel = torch.zeros(batch_size, self.myo_model.nv, device=self._device)
        root_pos = torch.zeros(batch_size, 3, device=self._device)
        root_rot = torch.zeros(batch_size, 4, device=self._device)  # quaternion
        root_vel = torch.zeros(batch_size, 3, device=self._device)
        root_ang_vel = torch.zeros(batch_size, 3, device=self._device)
        
        for i, (motion_id, time) in enumerate(zip(motion_ids, motion_times)):
            motion = self.motions[motion_id]
            
            # Find closest time frame
            time_diff = torch.abs(motion['time'] - time)
            frame_id = torch.argmin(time_diff)
            
            # Get motion data at this frame
            qpos = motion['qpos'][frame_id]
            qvel = motion['qvel'][frame_id]
            
            dof_pos[i] = qpos
            dof_vel[i] = qvel  # Note: qvel may be 139 dims vs 140 qpos
            
            # Extract root state (assuming first 7 DOFs are root: pos(3) + quat(4))
            root_pos[i] = qpos[:3]
            root_rot[i] = qpos[3:7]  # quaternion [w, x, y, z]
            root_vel[i] = qvel[:3]
            root_ang_vel[i] = qvel[3:6]
        
        # Return ASAP-compatible state dictionary
        return {
            "root_pos": root_pos,
            "root_rot": root_rot, 
            "dof_pos": dof_pos,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel": dof_vel,
            "motion_aa": torch.zeros(batch_size, self.myo_model.nq * 3, device=self._device),  # placeholder
            "motion_bodies": torch.zeros(batch_size, 17, device=self._device),  # placeholder
        }
    
    def get_motion_length(self, motion_id: int) -> float:
        """Get duration of specific motion - ASAP compatible interface."""
        return self._motion_lengths[motion_id].item()
    
    def get_num_motions(self) -> int:
        """Get total number of loaded motions - ASAP compatible interface."""
        return self._num_motions
    
    def num_motions(self) -> int:
        """ASAP compatibility - alias for get_num_motions."""
        return self._num_motions
    
    def get_total_length(self) -> float:
        """ASAP compatibility - get total duration of all motions."""
        return self._motion_lengths.sum().item() 