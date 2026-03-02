# Tower of Hanoi - Reinforcement Learning

Three different ways to interact with the Tower of Hanoi puzzle!

## Files Overview

### 1. **main.py** - Training Script
Train reinforcement learning agents to solve the puzzle.

**Features:**
- Q-Learning (tabular, CPU-only)
- Deep Q-Learning (DQN with GPU acceleration)
- Saves trained agents for later use
- Command-line arguments or interactive mode

**Usage:**
```bash
# Interactive mode (original behavior)
python3 main.py --interactive

# Command-line mode (recommended for scripting/HPC)
python3 main.py --algorithm qlearning --disks 5 --episodes 10000
python3 main.py -a dqn -d 8 -e 20000

# Quick defaults (qlearning, 5 disks, 10000 episodes)
python3 main.py
```

**Arguments:**
- `--algorithm, -a`: Choose 'qlearning' or 'dqn' (default: qlearning)
- `--disks, -d`: Number of disks (default: 5)
- `--episodes, -e`: Training episodes (default: 10000 for qlearning, 20000 for dqn)
- `--interactive, -i`: Use interactive prompts instead of command-line args

**Outputs:**
- `hanoi_qlearn_Xdisks.pkl` - Q-Learning agent
- `hanoi_dqn_Xdisks.pth` - DQN agent

---

### 2. **play.py** - Interactive Game
Play the Tower of Hanoi yourself in the terminal!

**Usage:**
```bash
python3 play.py
```

**Controls:**
- Enter moves as two digits (e.g., `13` = move from peg 1 to peg 3)
- Pegs: 1 (left), 2 (middle), 3 (right)
- Type `r` to restart, `h` for help, `q` to quit

**Try to match the optimal solution!**

---

### 3. **solve.py** - Solution Demo
Load a trained agent and watch it solve the puzzle.

**Usage:**
```bash
python3 solve.py
```

**Features:**
- Lists available trained agents
- Choose Q-Learning or DQN agent
- Watch the solution step-by-step or see final result
- Shows efficiency vs optimal solution

---

### 4. **test_local.py** - Quick Local Test
Test training with 4 disks locally before HPC training.

**Usage:**
```bash
python3 test_local.py
```

Quick 10k episode training to verify everything works.

---

### 5. **train_hpc.py** - HPC Training Script
Optimized for HPC cluster training with command-line arguments.

**Usage:**
```bash
# Q-Learning (8 disks, 100k episodes)
python3 train_hpc.py --disks 8 --episodes 100000 --algorithm qlearning

# DQN with GPU (8 disks, 50k episodes)
python3 train_hpc.py --disks 8 --episodes 50000 --algorithm dqn
```

---

### 6. **submit_hpc.sh** - SLURM Batch Script
Submit training job to HPC cluster.

**Usage:**
```bash
sbatch submit_hpc.sh
```

Adjust SLURM parameters for your cluster.

---

## Requirements

```bash
# Basic (Q-Learning and play)
pip install numpy

# For DQN with GPU
pip install torch
```

## Workflow: Local Test → HPC Training

### Step 1: Install dependencies
```bash
pip install numpy
# Optional: pip install torch  # For DQN
```

### Step 2: Test locally with 4 disks
```bash
python3 test_local.py
```
This should complete in a few minutes and verify everything works.

### Step 3: Train on HPC with 8 disks

**Prepare for HPC:**
1. Copy entire `hanoi_tower` folder to your HPC cluster
2. Install dependencies on HPC: `pip install numpy`
3. Adjust `submit_hpc.sh` for your cluster's SLURM configuration

**Submit job:**
```bash
sbatch submit_hpc.sh
```

**Monitor:**
```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f hanoi_training_<jobid>.log
```

### Step 4: Test the trained model
After training completes on HPC, download the trained agent file and run:
```bash
python3 solve.py
```

---

## Algorithm Comparison

| Algorithm | Speed | Efficiency | GPU Support | Best For |
|-----------|-------|------------|-------------|----------|
| Q-Learning | Slower training | Good (70-90%) | No | 3-6 disks, reliable |
| DQN | Faster with GPU | Good (70-90%) | Yes | 6-8 disks, HPC |

**Recommended:**
- **Local (3-5 disks)**: Q-Learning (10-50k episodes)
- **HPC (6-8 disks)**: Q-Learning (100k episodes) or DQN (50k episodes with GPU)

---

## Optimal Solutions

| Disks | Optimal Moves | Training Time (estimate) |
|-------|---------------|--------------------------|
| 3 | 7 | ~2 min (5k episodes) |
| 4 | 15 | ~5 min (10k episodes) |
| 5 | 31 | ~20 min (50k episodes) |
| 6 | 63 | ~1 hour (100k episodes) |
| 7 | 127 | ~3 hours (100k episodes) |
| 8 | 255 | ~6-12 hours (100k+ episodes) |

Formula: 2^n - 1

---

## Reward System Explained

The reinforcement learning agent learns through a carefully designed reward system:

### **Penalties (Negative Rewards)**

1. **Invalid Move Penalty: -20**
   - Given when trying to move from an empty peg
   - Given when trying to place a larger disk on a smaller disk
   - **Purpose:** Teach the agent the basic rules quickly and strongly discourage violations

2. **Step Penalty: -15 per move**
   - Strong penalty for every valid move
   - **Purpose:** Aggressively encourage efficiency and shortest possible solutions

### **Rewards (Positive Rewards)**

3. **Progress Reward: +1.5 per disk on destination peg**
   - Calculated as: `1.5 × (number of disks on peg 3)`
   - Example: 3 disks on goal = +4.5 reward
   - **Purpose:** Guide the agent toward the goal state
   - **Note:** This partially offsets the step penalty, making progress rewarding

4. **Completion Reward: +100**
   - Large reward for successfully solving the puzzle
   - **Purpose:** Strong signal that the goal was reached

5. **Efficiency Bonus: +20 per move saved**
   - Calculated as: `20 × (optimal_moves - actual_moves)`
   - Example: Solve 5-disk (optimal=31) in 33 moves = +20×(-2) = -40 bonus
   - Example: Solve 5-disk in 31 moves (optimal) = +20×0 = 0 bonus (but still get +100)
   - **Purpose:** Incentivize finding shorter, more efficient solutions

### **Total Reward Examples**

**Scenario 1: Optimal Solution (5 disks)**
- 31 moves (optimal)
- Each move: -15 + progress rewards (varies by stage)
- Completion: +100
- Efficiency bonus: 0 (optimal)
- **Total: Net positive due to completion reward**

**Scenario 2: Inefficient Solution**
- 50 moves (5 disks, optimal=31)
- Each move: -15 + progress rewards
- Completion: +100
- Efficiency bonus: 20×(31-50) = -380
- **Total: Large negative (agent strongly learns to avoid this)**

**Scenario 3: Invalid Moves**
- Multiple invalid moves: -20 each
- **Total: Very negative (agent very quickly learns the rules)**

### **Why This Works**

1. **Very Fast Rule Learning:** Large penalty (-20) for invalid moves
2. **Goal-Oriented:** Progress rewards (+1.5×disks) guide toward solution
3. **Strongly Efficiency Driven:** Heavy penalty (-15) + bonus (+20 per saved move) aggressively pushes toward optimal paths
4. **Strong Success Signal:** Big reward (+100) for completion

This aggressive reward structure helps the agent learn efficiently:
1. First, learn the valid moves (avoid -20 penalties)
2. Then, learn to reach the goal (follow progress rewards to offset -15 step penalty)
3. Finally, optimize for shortest path (minimize moves to avoid penalties, maximize efficiency bonus)

---

## Troubleshooting

**numpy not found:**
```bash
pip install numpy
```

**PyTorch not found (for DQN):**
```bash
# CPU only
pip install torch

# With CUDA support (check PyTorch website for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Training takes too long:**
- Reduce episodes for testing
- Use smaller number of disks
- Use HPC cluster for 7-8 disks

---

Enjoy solving the Tower of Hanoi! 🗼
