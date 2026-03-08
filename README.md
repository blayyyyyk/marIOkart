# MarI/O Kart
> Training an AI model to perform advanced speedrunner inputs of the popular racing title Mario Kart DS
  
**Table of Contents**
1. [Installation Guide](#installation-guide)
2. [Usage Guide](#usage-guide)
- 2.1 [Dataset Collection](#dataset-collection)
    - 2.1.1 [Recording .dsm](#recording-dsm)
    - 2.1.2 [Recording .sav](#recording-sav)
- 2.2 [Training the Model](#training-the-model)
    - 2.2.1 [Supervised Learning](#supervised-learning)
    - 2.2.2 [Reinforcement Learning (experimental)](#reinforcement-learning)
- 2.3 [Evaluating the Model](#evaluating-the-model)
- 2.4 [Debugging Tools](#debugging-tools)

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
