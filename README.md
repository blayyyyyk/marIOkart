# MarI/O Kart
> Training an AI model to perform advanced speedrunner inputs of the popular racing title Mario Kart DS
  
**Table of Contents**
1. [Installation Guide](#installation-guide)
2. [Usage Guide](#usage-guide)
    1. [Dataset Collection](#dataset-collection)  
        1. [Recording .dsm](#recording-dsm)
        2. [Recording .sav](#recording-sav)
    2. [Training the Model](#training-the-model)
        1. [Supervised Learning](#supervised-learning)
        2. [Reinforcement Learning (experimental)](#reinforcement-learning)
    3. [Evaluating the Model](#evaluating-the-model)
    4. [Debugging Tools](#debugging-tools)
3. [Working With the MarioKart DS Memory API](#working-with-the-mariokart-ds-memory-api)
4. [Working With Gymnasium](#working-with-gymnasium)

## Installation Guide

1. Clone the repository

```
git clone https://github.com/blayyyyyk/marIOkart.git
```

2. Install the dependencies
> [!TIP]
> A virtual environment is recommended for installation

```
pip install -r requirements.txt
```

## Usage Guide

All tools for this projects can be run with the following command.

```bash
cd [name of repo directory]
python src/main.py
```

### Dataset Collection

For our supervised learning training pipeline, we required labelled training examples of speedrunner data. Right now, we train models to predict speedrunner inputs given a set of observation data.

**Movie Replays**  
Our training environment uses an Nintendo DS emulator called DeSmuME. Using DeSmuME's built-in movie replay feature, we can load a series of controller inputs into the start of the game, and the emulator will replay the game in real-time using frame-by-frame prerecorded inputs. During this replay, our script will collect and precompute our observation data to save to a file. Our model then must predict the player's controller input given this observation data.

#### Recording `.dsm`

`record.py` To record a dataset using a desmume compatible movie file.

```bash
usage: record.py [-h] [--dest DEST] [--num-proc NUM_PROC] [--process] [--scale SCALE] [--device DEVICE] [--verbose] source [source ...]

positional arguments:
  source                Movie files (.dsm) or Game saves (.sav) to collect observation datasets from. Can either be individual files, a list of files, or a directory of
                        acceptable files.

options:
  -h, --help            show this help message and exit
  --dest DEST, -o DEST  Where to output the datasets to
  --num-proc NUM_PROC   maximum number of subprocesses to spawn
  --process             when flag is enabled, will make a call to process() before collecting datasets
  --scale SCALE         specify the scale of the gtk window
  --device DEVICE       PyTorch device name (ex. 'cpu', 'mps', 'cuda')
  --verbose, -v         Enable verbose console logging for debugging
```

#### Recording `.sav`

`.sav` files are MarioKartDS-specific save files for the game. From save files, you can extract what are called "ghost inputs" from time trial races which essentially play the same role as `.dsm` files however ghost inputs are internal to the game. To record training datasets from `.sav` files, you must first extract all ghost inputs and convert to `.dsm` files with `process.py`  
`process.py` - Extract all `.dsm` files from a `.sav` file.

```bash
usage: process.py [-h] [--device DEVICE] [--verbose] source [source ...]

positional arguments:
  source           Game saves (.sav) to extract ghost inputs from. Can either be individual files, a list of files, or a directory of acceptable files.

options:
  -h, --help       show this help message and exit
  --device DEVICE  PyTorch device name (ex. 'cpu', 'mps', 'cuda')
  --verbose, -v    Enable verbose console logging for debugging
```

### Training the Model

We are experimenting with all types of learning methods. we are currently working on both supervised learning using speedrunner data and reinforcement learning which does not directly uses speedrunner data.

#### Supervised Learning

`train.py` - Runs the training pipeline on a specified model using supervised.

```bash
usage: train.py [-h] [--model-name MODEL_NAME] [--course-name COURSE_NAME] [--epochs EPOCHS] [--lr LR] [--batch-size BATCH_SIZE] [--split-ratio SPLIT_RATIO]
                [--seq-len SEQ_LEN] [--device DEVICE] [--verbose]
                source [source ...]

positional arguments:
  source                List of training data sources to use

options:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        The name of the model to record a checkpoint of after training
  --course-name COURSE_NAME, -c COURSE_NAME
  --device DEVICE       PyTorch device name (ex. 'cpu', 'mps', 'cuda')
  --verbose, -v         Enable verbose console logging for debugging

Training Hyperparameters:
  --epochs EPOCHS, -e EPOCHS
                        Number of training epochs.
  --lr LR               Learning rate
  --batch-size BATCH_SIZE
                        Batch size
  --split-ratio SPLIT_RATIO
                        Ratio of training data to split into train and test sets
  --seq-len SEQ_LEN     Size of the sequence length of the training samples
```

#### Reinforcement Learning

> [!CAUTION]
> This training pipeline is currently under development
> `train_rl.py`

```bash
usage: train_rl.py [-h] [--epochs EPOCHS]
                   [--course {f8c,yf,ccb,lm,dh,ds,wp,sr,dkp,ttc,mc,af,ws,pg,bc,rr,rmc1,rmmf,rpc,rlc1,rdp1,rfs,rbc2,rbp,rkb2,rcm,rlc2,rmb,rci2,rbb,rsg,ryc}] [--device DEVICE] [--verbose] [--scale SCALE]
                   movie_source [movie_source ...] {ppo,dqn,a2c}

positional arguments:
  movie_source          movie files to perform menu naviagtion before training
  {ppo,dqn,a2c}         training algorithm

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training epochs
  --course {f8c,yf,ccb,lm,dh,ds,wp,sr,dkp,ttc,mc,af,ws,pg,bc,rr,rmc1,rmmf,rpc,rlc1,rdp1,rfs,rbc2,rbp,rkb2,rcm,rlc2,rmb,rci2,rbb,rsg,ryc}
  --device DEVICE       PyTorch device name (ex. 'cpu', 'mps', 'cuda')
  --verbose, -v         Enable verbose console logging for debugging
  --scale SCALE         specify the scale of the gtk window
```

### Evaluating the Model

`eval.py` - Runs a model with inference only in a gym-based game environment with display. window. Requires `.dsm` files to control the initial menu navigation.

```bash
usage: eval.py [-h] [--scale SCALE] [--device DEVICE] [--verbose] model_name movie_source [movie_source ...]

positional arguments:
  model_name       name of model to evaluate
  movie_source     the file(s) or director(ies) containing movie replays to source menu controls during evaluation (accepted files: .dsm)

options:
  -h, --help       show this help message and exit
  --scale SCALE    specify the scale of the gtk window
  --device DEVICE  PyTorch device name (ex. 'cpu', 'mps', 'cuda')
  --verbose, -v    Enable verbose console logging for debugging
```

### Debugging Tools

The program also supports for user provided input in the game environment. This is useful if you're testing custom visual overlays for displaying in-game data.
`debug.py` - Debug mode for human-user input or movie replays (`.dsm`)

```bash
usage: debug.py [-h] [--scale SCALE] [--device DEVICE] [--verbose] mode movie-source [movie-source ...]

positional arguments:
  mode             debugging mode for debug tool. available modes (movie, play)
  movie-source     the file(s) or director(ies) containing movie replays to source menu controls during evaluation (accepted files: .dsm)

options:
  -h, --help       show this help message and exit
  --scale SCALE    specify the scale of the gtk window
  --device DEVICE  PyTorch device name (ex. 'cpu', 'mps', 'cuda')
  --verbose, -v    Enable verbose console logging for debugging
```
## Working With the MarioKart DS Memory API
Under the hood, we use a popular library called `py-desmume`. It is a python library for interfacing with the DeSmuME C API. We forked the python project and integrated custom functionality for accessing game attributes from memory. By optimizing the C/C++ to Python interoperability, this enables not only *100x* performance improvements in memory reads by bypassing traditional hooking overhead, but it also allows users to extract variables in an intuitive and object-oriented fashion. This high-speed data pipeline is essential for preventing bottlenecks during intensive model training.

With this API, you can easily access complex internal observation states directly from the emulator's RAM without needing to rely on heavier computer vision techniques or screen captures. [Learn more](https://github.com/blayyyyyk/py-desmume-mkds)

**Installation**
```bash
pip install py-desmume-mkds
```
**Example Usage**
```python
from desmume.emulator_mkds import MarioKart
import torch

emu = MarioKart()
emu.open('pathtorom.nds')

# Create the window for the emulator
window = emu.create_sdl_window()

# Run the emulation as fast as possible until quit
while not window.has_quit():
    window.process_input()   # Controls are the default DeSmuME controls, see below.
    emu.cycle()
    window.draw()

    # checks if a race has started
    if not emu.memory.race_ready: continue
    
    # access the player's current kart position
    kart_position: torch.Tensor = emu.memory.driver.position

    # access the player's current boost timer
    boost_timer: float = self.emu.memory.driver.boostTimer

    # access the player's current race_progress
    race_progress: float = emu.memory.race_status.driverStatus[0].raceProgress
```

## Working With Gymnasium
For those who are curious to start training their own RL agents in MarioKart DS, we have a standalone gymnasium environment. This environment abstracts the emulator mechanics into a standard RL problem, providing a structured observation space, discrete action space, and easily modifiable reward functions (such as optimizing for raceProgress).

The MarioKartDS-v0 environment handles the complexities of emulator synchronization and exposes essential spatial control parameters. By utilizing raycasting mechanics (ray_max_dist, ray_count), the environment gives the agent a simulated LiDAR-like view of the track to gauge distances to walls and hazards, bridging the gap between emulator internals and autonomous driving logic. [Learn more](https://github.com/blayyyyyk/gym-mkds)

**Installation**
```bash
pip install gym-mkds
```

**Example Usage**
```python
import gymnasium

env = gymnasium.make(
    "gym_mkds/MarioKartDS-v0",
    rom_path="pathtorom",
    ray_max_dist=3000,
    ray_count=20,
)
```

**Example Usage: Simple Human Input With Visual Overlays**
```python
import gymnasium
from functools import partial
from gym_mkds.wrappers import (
    EnvWindow,
    OverlayWrapper, 
    HumanInputWrapper, 
    compose_overlays, 
    collision_overlay, 
    sensor_overlay
)

env = gymnasium.make(
    "gym_mkds/MarioKartDS-v0",
    rom_path="pathtorom",
    ray_max_dist=3000,
    ray_count=20,
)

# make a composite overlay of different visuals
overlay_func = partial(compose_overlays, funcs=[collision_overlay, sensor_overlay])
env = OverlayWrapper(env, func=overlay_func)

# add a keyboard listener
env = HumanInputWrapper(env)

# create a gtk window to display the game
window = EnvWindow(env)
obs, info = env.reset()

# game loop
while window.is_alive:
    obs, reward, terminated, truncated, info = env.step(0)
    window.update()
```
