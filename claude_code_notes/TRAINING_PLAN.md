# MyoDolores Keyboard Control Training Plan

## Project Overview
Goal: Create a keyboard-controlled demo for myo_model humanoid using up/down/left/right keys for locomotion control.

## Current Status - Environment Analysis Complete
✅ **myo_model_internal**: MyoSkeleton v1.0.0 - Full humanoid model with 324 joints, spine/neck/arms/legs
✅ **myo_api**: mjTrajectory system for loading H5 motion data, MuJoCo 3.2.4 integration
✅ **myo_data**: 8,000+ motion capture sequences (HAA500, AIST, dance, etc.) in H5 format
✅ **ASAP/HumanoidVerse**: RL framework with command-based control, 50Hz real-time loops
✅ **unitree_rl_gym**: Real robot deployment examples with velocity command interface

## Key Technical Insights
1. **Physics Reality**: Direct joint control won't work - need trained RL policy for balance/physics
2. **Command Architecture**: Both ASAP and unitree use velocity commands [vel_x, vel_y, ang_vel_z] as policy inputs
3. **Real-time Control**: 50Hz control loop with policy inference for responsive keyboard control
4. **Training Required**: No pre-trained policies exist for myo_model - must train from scratch

## Training Strategy - RL Policy for Velocity Command Following

### Environment Setup (Next Machine)
```bash
# Create conda environment
conda create -n myodolores python=3.8
conda activate myodolores

# Install dependencies
pip install mujoco==3.2.4
pip install -e ASAP/
pip install -e ASAP/isaac_utils/
pip install -e myo_api/
```

### Training Configuration
**Framework**: ASAP/HumanoidVerse with Isaac Gym
**Model**: MyoSkeleton from myo_model_internal/myo_model/myoskeleton/myoskeleton_with_motors.xml
**Task**: Locomotion with velocity command following
**Training time**: 8-10 hours on RTX 3090+ GPU

### Training Command
```bash
python ASAP/humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+robot=myoskeleton \
+rewards=loco/velocity_tracking \
+obs=loco/locomotion_obs_with_commands \
num_envs=4096 \
project_name=MyoDoloresDemo \
experiment_name=VelocityControl_Training
```

### Observation Space Design
```python
obs = [
    base_lin_vel[3],           # Current base velocity
    base_ang_vel[3],           # Current angular velocity  
    gravity_vector[3],         # Gravity direction in base frame
    command_vel[3],            # Target velocity commands [x, y, yaw]
    joint_positions[324],      # Joint position errors from reference
    joint_velocities[324],     # Joint velocities
    action_history[324],       # Previous actions
]
```

### Reward Design
```python
rewards = {
    'velocity_tracking': track_velocity_commands,  # Primary objective
    'upright_posture': maintain_balance,
    'joint_limits': avoid_joint_limits,
    'energy_penalty': minimize_torques,
    'stability': foot_contact_stability
}
```

### Keyboard Control Implementation (Post-Training)
```python
# Real-time control loop
while True:
    # 1. Get keyboard input
    vel_cmd = keyboard_to_velocity_commands(key_state)
    
    # 2. Update observation with commands
    obs[9:12] = vel_cmd  # Command portion of observation
    
    # 3. Policy inference
    actions = trained_policy(obs)
    
    # 4. Apply to simulation
    mj_data.ctrl[:] = actions
    mujoco.mj_step(mj_model, mj_data)
    
    time.sleep(0.02)  # 50Hz
```

### Key Files for Next Machine
- `/home/ubuntu/MyoDolores/myo_model_internal/myo_model/myoskeleton/myoskeleton_with_motors.xml` - Robot model
- `/home/ubuntu/MyoDolores/myo_data/HAA500_output/*/` - Motion reference data
- `/home/ubuntu/MyoDolores/ASAP/` - Training framework
- `/home/ubuntu/MyoDolores/myo_api/` - Model loading utilities

## Next Steps (GPU Machine Required)
1. Transfer codebase to GPU machine
2. Set up training environment 
3. Configure MyoSkeleton robot for ASAP training
4. Train velocity-command-following policy (8-10 hours)
5. Implement keyboard control demo with trained policy
6. Test and iterate on control responsiveness

## Hardware Requirements
- RTX 3090+ GPU (for 4096 parallel environments)
- 32GB+ RAM
- CUDA 11.8+ 
- Isaac Gym installation

## Expected Training Timeline
- Environment setup: 1-2 hours
- Training: 8-10 hours 
- Demo implementation: 2-3 hours
- **Total**: ~12-15 hours on GPU machine