import random, colorsys
from multiprocessing.shared_memory import SharedMemory

def palette_gen(n):
    # Saturation: High (0.7 to 1.0) for vividness
    s = random.uniform(0.7, 1.0)
    # Lightness: High (0.5 to 0.7) to avoid darkness
    l = random.uniform(0.5, 0.7)
    
    return [colorsys.hls_to_rgb(h / n, l, s) for h in range(n)]

def hex_color_gen():
    # Hue: Random color (0.0 to 1.0)
    h = random.random()
    # Saturation: High (0.7 to 1.0) for vividness
    s = random.uniform(0.7, 1.0)
    # Lightness: High (0.5 to 0.7) to avoid darkness
    l = random.uniform(0.5, 0.7)
    
    # Convert to RGB (returns 0.0 to 1.0)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    
    # Return as hex string for your CSS
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
def shm_exists(name: str) -> bool:
    try:
        # Attempt to attach to an existing block
        shm = SharedMemory(name=name)
        shm.close()  # Always close the reference if you aren't using it
        return True
    except FileNotFoundError:
        return False
        
def attach_shm(name, size):
    if shm_exists(name):
        shm = SharedMemory(name=name, size=size)
    else:
        shm = SharedMemory(name=name, create=True, size=size)
    return shm