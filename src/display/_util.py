import random, colorsys

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