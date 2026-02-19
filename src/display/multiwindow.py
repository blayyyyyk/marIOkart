import gi

from src.display._util import palette_gen
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib
from src.display._impl import EmulatorContainerImpl
from typing import Literal, Optional
import math, cairo
from desmume.emulator import SCREEN_WIDTH, SCREEN_HEIGHT

class VirtualView(Gtk.Overlay, EmulatorContainerImpl):
    def __init__(
        self,
        width: int,
        height: int,
        orientation: Literal["horizontal", "vertical"],
        scale: int,
        id: int | None,
        label_color: tuple[float, float, float] = (0.2, 0.2, 0.2),
        label_text: Optional[str] = None
    ):
        Gtk.Overlay.__init__(self)
        self.setup_container(width, height, orientation, scale, id)
        self.id = id
        self.game_area = Gtk.DrawingArea()
        self.game_area.set_size_request(
            self.width * self.scale, self.height * self.scale
        )
        
        self.visible_width = self.total_width * self.scale
        self.visible_height = self.total_height * self.scale
        if self.visible_width > self.visible_height:
            self.visible_width /= 2
        else:
            self.visible_height /= 2
        
        self.game_area.connect("draw", self.on_draw_main)
        
        if label_text is None:
            self.label_text = f"Virtual Emulator Window {id}"
        else:
            self.label_text = label_text

        self.label = Gtk.Label(label=self.label_text)
        self.label.set_name("emulator-label")
        self.label.set_halign(Gtk.Align.START)
        self.label.set_valign(Gtk.Align.START)
        
        style_provider = Gtk.CssProvider()
        r, g, b = [int(c * 255) for c in label_color]
        print(label_color)
        css = f"""
        #emulator-label {{
            background-color: rgba({r}, {g}, {b}, 0.5);
            color: white;
            font-size: 10px;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 0 0 5px 0;
        }}
        """.encode('utf-8')
        style_provider.load_from_data(css)
        
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            style_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        
        self.label_color = label_color
        
        rgba = Gdk.RGBA(label_color[0], label_color[1], label_color[2], 1.0)
        self.label.override_background_color(Gtk.StateFlags.NORMAL, rgba)
        
        self.add_overlay(self.label)

        self.add(self.game_area)
        self.show_all()

    def on_draw_main(self, widget: Gtk.DrawingArea, ctx: cairo.Context, *args):
        assert self.shm_surface
        
        ctx.rectangle(0, 0, self.visible_width, self.visible_height)
        ctx.clip()
        
        ctx.scale(self.scale, self.scale)
        ctx.set_source_surface(self.shm_surface, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_NEAREST)
        ctx.paint()
        self.game_area.queue_draw()
        
        ctx.set_source_rgb(*self.label_color) # Use your tuple directly
        ctx.set_line_width(4.0) # Choose your thickness
        ctx.rectangle(0, 0, self.visible_width, self.visible_height)
        ctx.stroke()
        
        return True
        
class MultiEmulatorWindow(Gtk.Window):
    def __init__(
        self,
        width: int,
        height: int,
        orientation: Literal["horizontal", "vertical"],
        scale: int,
        n_windows: int,
    ):
        super().__init__()
        self.set_title(f"Multi-View ({n_windows})")

        self.grid = Gtk.Grid()
        # self.grid.set_column_homogeneous(True)
        # self.grid.set_row_homogeneous(True)
        n_cols = math.ceil(math.sqrt(n_windows))
        if n_cols > 0:
            n_rows = math.ceil(n_windows / n_cols)
        else:
            n_rows = 0  # Handle 0 windows edge case

        self.set_default_size(
            SCREEN_WIDTH * scale * n_cols // 2, SCREEN_HEIGHT * scale * n_rows
        )
        
        palette = palette_gen(n_windows)
        for id, label_color in enumerate(palette):
            # Calculate x (col) and y (row) index
            col = id % n_cols
            row = id // n_cols
            sub_window = VirtualView(
                width, 
                height, 
                orientation, 
                scale, 
                id=id,
                label_color=label_color,
                label_text=f"Instance {id}"
            )

            # Attach to grid: child, left, top, width, height
            self.grid.attach(sub_window, col, row, 1, 1)
            print(col, row)

        self.add(self.grid)
        self.connect("destroy", self.on_destroy)
        self.show_all()

    def on_destroy(self, widget: Gtk.Widget):
        for child in self.grid.get_children():
            child.destroy()

        Gtk.main_quit()