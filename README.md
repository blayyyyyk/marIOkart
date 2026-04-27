# MarI/O Kart
> Training an AI model to perform advanced speedrunner inputs of the popular racing title Mario Kart DS

MarI/O Kart is a comprehensive machine learning framework built in Python that trains autonomous driving agents to navigate Mario Kart DS. Utilizing PyTorch for model training, the project explores both supervised learning—predicting and cloning advanced speedrunner techniques from `.dsm` and `.sav` files—and experimental reinforcement learning. To support these high-performance models, the repository features a custom-optimized DeSmuME emulator API for lightning-fast memory access and a standalone Gymnasium environment, complete with GTK-based visual overlays for real-time observation rendering.

**Table of Contents**
1. [Installation Guide](#installation-guide)
2. [Usage Guide](#usage-guide)
3. [Working With the MarioKart DS Memory API](#working-with-the-mariokart-ds-memory-api)

## Installation Guide

1. Clone the repository

```
git clone https://github.com/blayyyyyk/marIOkart.git
```

2. Install the dependencies
> [!TIP]
> [uv]("https://docs.astral.sh/uv/getting-started/installation/") package manager is highly recommended for it's ease of use. But you can still install dependencies with `pip install -r requirements.txt`

```
cd marIOkart
uv sync
source .venv/bin/activate
```

## Usage Guide

### Training
```bash
python -m mariokart_ml train -c example_configs/train_rl.json
```
> [!TIP]
> You can add your own config json and/or override specific options with CLI flags

`--window` - Enables a visual window of the training

> [!CAUTION]
> This may reduce performance

`--scale [integer]` - Specify the scale of the visual window

`--num-procs [integer]` - Specify the number of parallel emulator instances to train in parallel.

`--save-model-path [.zip path]` - Specify the model weights to load before beginning training.

`--load-model-path [.zip path]` - Specify the output file name to save the trained model weights to after training.

`--env-name [environment name]` - The gymnasium environment to run the training in.

> [!TIP]
> Recommended environments: `mariokart_ml/TimeTrial-v1`, `mariokart_ml/TimeTrial-streamlit-v1`

#### Tensorboard Support

Training telemetry data is recorded by both Stable Baselines 3 and our own in-house data collection. Both can be viewed by starting the server with the specified `logdir`

**Stable Baselines**
```bash
tensorboard --logdir=tensorboard_logs/PPO_7
```

**Custom Telemetry**
```bash
tensorboard --logdir=tensorboard_logs/mkds_ppo_20260426_152202
```

Once the server is running, you can access tensorboard in your browser at http://localhost:[specified port number]

### Debugging

You can test the emulator with human input with the following command:

```bash
python -m mariokart_ml debug --play -c example_configs/debug.json
```

### Realtime Per-step metrics Dashboard

A web-based [streamlit](https://streamlit.io/) dashboard can be used to view realtime observation and reward distribution for each emulator instance. View the demo [here](https://streamable.com/02z96h)

To enable this dashboard, you must use the `mariokart_ml/TimeTrial-streamlit-v1`

**debugging**
```bash
python -m mariokart_ml debug --play -c example_configs/debug.json --env-name mariokart_ml/TimeTrial-streamlit-v1
```
**training**
```bash
python -m mariokart_ml train -c example_configs/train_rl.json --env-name mariokart_ml/TimeTrial-streamlit-v1
```

#### Starting the Streamlit Server

```bash
streamlit run src/mariokart_ml/streamlit/app.py -- --num-instances 16 --base-data-port 64000
```
> [!NOTE]
> `--num-instances` in almost all circumstances should be qual to `--num-procs`.

> [!WARNING]
> Make sure all streamlit tabs are closed before starting the streamlit server. If you are starting up a new training run, you DO NOT need to restart the streamlit server; your existing tab should pick up the data stream automatically when the new run starts up.

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
