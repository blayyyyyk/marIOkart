from desmume.emulator import DeSmuME

def init_desmume(rom_path: str):
    # Initialize desmume
    emu = DeSmuME()
    emu.open(rom_path)
    emu.savestate.load(1)
    emu.volume_set(0) #turn down the fuckass volume
    
    # Create the window for the emulator
    window = emu.create_sdl_window()
    return emu, window