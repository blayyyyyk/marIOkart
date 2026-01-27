from core.emulator import MarioKart
from src.core.memory import *
from src.utils.vector import get_mps_device
from mkdslib.stubgen_out import race_status_t
import ctypes, torch, os, warnings, sys
import numpy as np
from torch.utils.data import DataLoader
import json
from contextlib import nullcontext

warnings.filterwarnings("ignore")

KEYMASK_SIZE = 11 # number of bits from keymask included in labelled data



class RaceDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, sample_dim=3, target_dim=KEYMASK_SIZE):
        self.sample_mm = np.memmap(f"{folder_path}/samples.dat", dtype=np.float32, mode="r")
        self.target_mm = np.memmap(f"{folder_path}/targets.dat", dtype=np.bool, mode="r")
        self.length = len(self.sample_mm) // sample_dim
        self.sample_data = self.sample_mm.reshape(self.length, sample_dim)
        self.target_data = self.target_mm.reshape(self.length, target_dim)
        

    def __len__(self):
        return self.length - 1

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return torch.from_numpy(self.sample_data[idx]), torch.from_numpy(self.target_data[idx.start+1:idx.stop+1])
        
        return torch.from_numpy(self.sample_data[idx]), torch.from_numpy(self.target_data[idx+1])


def int_to_binary_array(num, width=KEYMASK_SIZE):
  """
  Converts a single integer to a binary numpy array.
  If width is provided, it pads with leading zeros.
  """
  if width:
    # Use numpy.binary_repr for fixed width (handles negative nums with two's complement)
    binary_str = np.binary_repr(num, width)
  else:
    # Use standard bin() and slice off the '0b' prefix
    binary_str = bin(num)[2:]

  # Convert each character in the string to an integer and create a numpy array
  binary_array = np.array([bit == "1" for bit in binary_str], dtype=np.bool)
  return binary_array


def create_dataset(dsm_path: str, output_dir_path: str, n_rays: int):
    device = torch.device("cpu")
    emu = MarioKart(
        n_rays=n_rays,
        max_dist=3000.0,
        device=device
    )
    emu.open('private/mariokart_ds.nds')
    emu.movie.play(dsm_path)
    emu.volume_set(0)
    window = emu.create_sdl_window()
    try:
        os.system("clear") # platform-specific
    except:
        pass

    # track the mean and std for feature normalization across batches/sequences
    running_mean = None
    running_M2: np.ndarray | None = None  # Sum of squares of differences
    n_count = 0

    print("Dataset recording started. Please exit the emulator window or CTRL+C once the replay has finished.")

    dataset_path = f"{output_dir_path}/{os.path.basename(dsm_path).split('.')[0]}"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    keys = np.array([0, 33, 289, 1, 257, 321, 801, 273, 17])
    obs_dim = 0

    upper_bound = torch.zeros((1,)).unsqueeze(0)
    
    with open(f"{dataset_path}/samples.dat", "wb") if args.record is else nullcontext() as sf, open(f"{dataset_path}/targets.dat", "wb") as tf:
        while not window.has_quit():
            window.process_input()   # Controls are the default DeSmuME controls, see below.
            window.draw()
            

            addr = emu.memory.unsigned.read_long(RACE_STATUS_PTR_ADDR)
            data = bytes(emu.memory.unsigned[addr:addr+ctypes.sizeof(race_status_t)])
            race_status = race_status_t.from_buffer_copy(data)
            

            if race_status.time.timeRunning != 1:
                emu.cycle()
                continue
                
            emu.cycle(grad=True)

            progress = race_status.driverStatus[0].raceProgress / 0x1000
            if progress > 1.0:
                window.destroy()
                break

            # Computation #
            
            input_vector = emu.memory.get_obs(n_rays, max_dist=3000.0, device=device)
            input_vector = torch.cat([
                input_vector,
                emu.grad
            ], dim=-1)
            arr = input_vector.detach().cpu().numpy()
            obs_dim = arr.shape[0]

            if running_mean is None:
                running_mean = np.zeros_like(arr, dtype=np.float64) # Use float64 for precision
            
            if running_M2 is None:
                running_M2 = np.zeros_like(arr, dtype=np.float64)

            n_count += 1
            delta = arr - running_mean
            running_mean += delta / n_count
            delta2 = arr - running_mean
            running_M2 += delta * delta2


            sf.write(arr.astype("float32").tobytes())

            # Inputs #
            keypad = emu.input.keypad_get()
            
            tf.write(np.array([keypad]).astype("int32").tobytes())

    assert running_M2 is not None
    if n_count > 1:
        variance = running_M2 / n_count
        std_dev = np.sqrt(variance).tolist()
    else:
        # Fallback if recording was empty
        std_dev = np.ones_like(running_mean).tolist() if running_mean is not None else []
    
    running_mean = running_mean.tolist() if running_mean is not None else []

    with open(f"{dataset_path}/metadata.json", "w") as f:
        data = {
            "mean": running_mean,
            "std": std_dev,
            "count": n_count,
            "obs_dim": obs_dim
        }
        json.dump(data, f)
        
    print(f"Dataset Saved Successfully at {dataset_path}!")
    print(keys, len(keys))
    
def record_main():
    create_dataset(
        dsm_path="./private/input_data/f8c_pikalex.dsm",
        output_dir_path="./private/training_data",
        n_rays=16
    )


if __name__ == "__main__":
    record_main()
    
    
