import os, math
from utils.init_emulator import init_desmume
from utils.keyboard_hack import init_keyhack
from utils.memory import read_cart_angle, read_cart_position
from utils.collision_data import read_collision_data
import torch
from utils.mps import get_mps_device

device = get_mps_device()

emu, window = init_desmume('mariokart_ds.nds')
init_keyhack(emu)

# Step 1: compute region offset
ptr_offset = 0

# Step 2: compute racer data pointer
ptr_racer_data = hex(0x0217ACF8 + ptr_offset)
addrs = {
    "ptrRacerData": 0x0217ACF8 + ptr_offset,
    "ptrPlayerInputs": 0x02175630 + ptr_offset,
    "ptrGhostInputs": 0x0217568C + ptr_offset,
    "ptrRaceTimers": 0x0217AA34 + ptr_offset,
    "ptrMissionInfo": 0x021A9B70 + ptr_offset,
    "ptrObjStuff": 0x0217B588 + ptr_offset,
    "racerCount": 0x0217ACF4 + ptr_offset,
    "ptrSomeRaceData": 0x021759A0 + ptr_offset,
    "ptrCheckNum": 0x021755FC + ptr_offset,
    "ptrCheckData": 0x02175600 + ptr_offset,
    "ptrScoreCounters": 0x0217ACFC + ptr_offset,
    "collisionData": 0x0217b5f4 + ptr_offset,
    "ptrCurrentCourse": 0x23cdcd8 + ptr_offset,
    "ptrCamera": 0x217AA4C + ptr_offset,
    "ptrVisibilityStuff": 0x217AE90 + ptr_offset,
    "cameraThing": 0x207AA24 + ptr_offset,
    "ptrBattleController": 0x0217b1dc + ptr_offset,
}

def collision_data_tensor(data, device=None):
    return torch.tensor(data, device=device)
    
def checkpoint_data_tensor(data, device=None):
    points_0 = []
    points_1 = []
    for entry in data.items():
        p0 = torch.tensor(entry["p0"], device=device)
        p1 = torch.tensor(entry["p1"], device=device)
        points_0.append(p0)
        points_1.append(p1)
        
    return torch.stack(points_0), torch.stack(points_1)
    
def find_nearest_distance(collision_data: torch.Tensor):
    print(collision_data)

collision_tensor = read_collision_data()
collision_tensor = collision_data_tensor(collision_tensor, device=device)
find_nearest_distance(collision_tensor)

def main():
    # Run the emulation as fast as possible until quit
    while not window.has_quit():
        window.process_input()   # Controls are the default DeSmuME controls, see below.
        emu.cycle()
        window.draw()
        
        #os.system('clear')
            
        x, y, z = read_cart_position(emu)
        rad = read_cart_angle(emu)

        
        
    
    
if __name__ == "__main__":
    main()