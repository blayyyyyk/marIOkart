from src.utils.desmume_ext import DeSmuME
from src.core.memory import *
from src.utils.vector import get_mps_device
from private.mkds import race_status_t
import ctypes, torch, os, warnings, sys
import numpy as np
from torch.utils.data import DataLoader

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


def create_dataset(dsm_path: str, output_dir_path: str):
    emu = DeSmuME()
    emu.open('private/mariokart_ds.nds')
    emu.movie.play(dsm_path)
    emu.volume_set(0)
    window = emu.create_sdl_window()
    device = get_mps_device()
    try:
        os.system("clear") # platform-specific
    except:
        pass

    print("Dataset recording started. Please exit the emulator window or CTRL+C once the replay has finished.")

    dataset_path = f"{output_dir_path}/{os.path.basename(dsm_path).split('.')[0]}"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    with open(f"{dataset_path}/samples.dat", "wb") as sf, open(f"{dataset_path}/targets.dat", "wb") as tf:
        while not window.has_quit():
            window.process_input()   # Controls are the default DeSmuME controls, see below.
            emu.cycle()
            window.draw()

            addr = emu.memory.unsigned.read_long(RACE_STATUS_PTR_ADDR)
            data = bytes(emu.memory.unsigned[addr:addr+ctypes.sizeof(race_status_t)])
            race_status = race_status_t.from_buffer_copy(data)

            if race_status.time.timeRunning != 1:
                continue

            # Computation #
            fwd: torch.Tensor = read_forward_distance_obstacle(emu, device=device, interval=(-0.1, 0.1), n_steps=24, sweep_plane="xz")
            left: torch.Tensor = read_left_distance_obstacle(emu, device=device, interval=(-0.1, 0.1), n_steps=24, sweep_plane="xz")
            right: torch.Tensor = read_right_distance_obstacle(emu, device=device, interval=(-0.1, 0.1), n_steps=24, sweep_plane="xz")
            input_vector = torch.cat([left, fwd, right], dim=-1)
            arr = input_vector.detach().cpu().numpy()
            sf.write(arr.astype("float32").tobytes())

            # Inputs #
            arr = int_to_binary_array(emu.input.keypad_get())
            tf.write(arr.astype("bool").tobytes())

    print(f"Dataset Saved Successfully at {dataset_path}!")
    
def record_main():
    create_dataset(
        dsm_path="./private/input_data/f8c_pikalex.dsm",
        output_dir_path="./private/training_data"
    )


if __name__ == "__main__":
    #record_main()
    device = get_mps_device()
    ds = RaceDataset("private/training_data/f8c_pikalex")
    print(ds[5:8])
    
    
