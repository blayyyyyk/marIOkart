import numpy as np
import gymnasium as gym
from typing import cast
from desmume.emulator import SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT_BOTH

import gi, cairo
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

def tile_images(images: np.ndarray) -> np.ndarray:
    """Reduces the image dimensions by tiling the images into a single image."""
    N, H, W, C = images.shape
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    # add blank tiles (if necessary)
    N_blank = cols * rows - N
    if N_blank > 0:
        blanks = np.zeros((N_blank, H, W, C), dtype=np.uint8)
        images = np.concatenate([images, blanks])

    out = (
        images.reshape(rows, cols, H, W, C)
        .swapaxes(1, 2)
        .reshape(rows * H, cols * W, C)
    )
    return out

class WindowBase(Gtk.Window):
    def __init__(self, env, scale: float = 1.0):
        super(WindowBase, self).__init__()
        self.set_title("MarioKart Async Training")
        self.set_default_size(SCREEN_WIDTH, SCREEN_HEIGHT_BOTH)
        self.size_set = None
        self.env = env
        self.is_alive = True
        self.scale = scale
        
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect("draw", self.on_draw)
        self.add(self.drawing_area)
        
        self.connect("destroy", self.on_destroy)
        # We don't set a default size yet, we'll let Cairo handle it
        self.show_all()
        
    def on_destroy(self, widget: Gtk.Widget):
        self.is_alive = False
        self.destroy()
        Gtk.main_quit()
        
    def update(self):
        # Queue a redraw for the main GTK window
        self.drawing_area.queue_draw()
        
        # Process GTK events
        while Gtk.events_pending():
            Gtk.main_iteration()
        
    def on_draw(self, widget: Gtk.Widget, ctx: cairo.Context) -> bool:
        ...
        
class EnvWindow(WindowBase):
    def __init__(self, env: gym.Env, scale: float = 1.0):
        super(EnvWindow, self).__init__(env, scale)
        
    def on_draw(self, widget: Gtk.Widget, ctx: cairo.Context) -> bool:
        arr = cast(np.ndarray, self.env.render())
        if not isinstance(arr, np.ndarray):
            return True
            
        width, height, _ = arr.shape
        new_arr = np.ones((width, height, 4), dtype=np.uint8)
        new_arr[:, :, :3] = arr
        half_height = height // 2
        top = np.ascontiguousarray(new_arr[:, :half_height, :])
        bottom = np.ascontiguousarray(new_arr[:, half_height:, :])
        
        upper_image = cairo.ImageSurface.create_for_data(
            top.data, cairo.FORMAT_RGB24, width, height // 2
        )
        
        lower_image = cairo.ImageSurface.create_for_data(
            bottom.data, cairo.FORMAT_RGB24, width, height // 2
        )
        
        ctx.scale(self.scale, self.scale)
        ctx.set_source_surface(upper_image, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_NEAREST)
        ctx.paint()
        
        ctx.scale(self.scale, self.scale)
        ctx.set_source_surface(lower_image, 0, SCREEN_HEIGHT)
        ctx.get_source().set_filter(cairo.FILTER_NEAREST)
        ctx.paint()
        
        return True
        
class VecEnvWindow(WindowBase):
    def __init__(self, vec_env: gym.vector.AsyncVectorEnv, scale: float = 1.0, colors: list[tuple[float, float, float]] | None = None):
        super(VecEnvWindow, self).__init__(env=vec_env, scale=scale)
        # Default colors if none provided (R, G, B normalized 0-1)
        self.colors = colors or [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 
            (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)
        ]
        self.frame_width = 4 # Border thickness in pixels

    def on_draw(self, widget: Gtk.Widget, ctx: cairo.Context) -> bool:
        images = self.env.render()
        if not isinstance(images, (np.ndarray, tuple)):
            return True
        if isinstance(images, tuple):
            images = np.stack(images)

        N, H, W, C = images.shape
        images = images[:, :H//2, :, :]
        H = images.shape[1]
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))

        # Update window size once based on the grid + frame spacing
        total_w = cols * (W + self.frame_width * 2)
        total_h = rows * (H + self.frame_width * 2)
        if not self.size_set:
            self.resize(int(total_w * self.scale), int(total_h * self.scale))
            self.size_set = True

        ctx.scale(self.scale, self.scale)

        for i in range(N):
            col = i % cols
            row = i // cols
            
            # Calculate top-left position for this tile's frame
            x = col * (W + self.frame_width * 2)
            y = row * (H + self.frame_width * 2)
            
            # 1. Draw the Frame
            color = self.colors[i % len(self.colors)]
            ctx.set_source_rgb(*color)
            ctx.rectangle(x, y, W + self.frame_width * 2, H + self.frame_width * 2)
            ctx.fill()

            # 2. Draw the Game Image
            # Pad image to 4 channels for Cairo compatibility
            img = images[i]
            rgba_buffer = np.zeros((H, W, 4), dtype=np.uint8)
            rgba_buffer[:, :, :3] = img
            stride = W * 4
            surface = cairo.ImageSurface.create_for_data(
                np.ascontiguousarray(rgba_buffer), cairo.FORMAT_RGB24, W, H, stride
            )
            
            ctx.set_source_surface(surface, x + self.frame_width, y + self.frame_width)
            ctx.get_source().set_filter(cairo.FILTER_NEAREST)
            ctx.paint()

            # 3. Draw the Label (Top Right of the specific tile)
            label_text = f"Agent {i}"
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(12)
            
            # Measure text to align it to the right
            extents = ctx.text_extents(label_text)
            text_x = x + (W + self.frame_width) - extents.width - 5
            text_y = y + self.frame_width + extents.height + 5
            
            # Draw a small background for text readability
            ctx.set_source_rgba(0, 0, 0, 0.6)
            ctx.rectangle(text_x - 2, text_y - extents.height - 2, extents.width + 4, extents.height + 4)
            ctx.fill()
            
            # Draw actual text in the frame color
            ctx.set_source_rgb(*color)
            ctx.move_to(text_x, text_y)
            ctx.show_text(label_text)

        return True