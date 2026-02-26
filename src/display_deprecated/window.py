from typing import (
    Any,
    Optional,
    Protocol,
    TypeVar,
    TypedDict,
    NotRequired,
    cast,
)
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process, Manager, Queue
from src.display.overlay import AsyncOverlay
from desmume.emulator import SCREEN_HEIGHT, SCREEN_PIXEL_SIZE, SCREEN_WIDTH
import cairo
from src.core.emulator import MarioKart
from src.display._util import palette_gen, attach_shm
import pynput, traceback, math, os
from nanoid import generate as _generate
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib
import torch

OVERLAY_REFRESH_RATE = 60
DEFAULT_WINDOW_NAME = "Window"
DEFAULT_WINDOW_STYLE = css = (
    f"""
#emulator-label {{
    background-color: rgba(255, 255, 255, 0.5);
    color: white;
    font-size: 10px;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 0 0 5px 0;
}}
""".encode(
        "utf-8"
    )
)
BACKEND_DEVICE = torch.device("cpu")

id = 0


def generate():
    return _generate(alphabet="0123456789abcdefghijklmnopqrst", size=20)


class Window:
    def __init__(
        self, width, height, native_scale: int = 1, uuid: Optional[str] = None
    ):
        self.n_bytes = width * height * (native_scale**2) * 4
        self.shm = attach_shm(name=(uuid or generate()), size=self.n_bytes)
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_size_request(width * native_scale, height * native_scale)

        self.width = width
        self.height = height
        self.drawing_area.connect("draw", self.on_main_draw)
        self.drawing_area.connect("destroy", self.on_main_destroy)

    @property
    def uuid(self):
        return self.shm.name

    def close(self):
        pass

    def on_main_draw(self, widget: Gtk.Widget, ctx: cairo.Context) -> bool: ...

    def _on_main_draw(self, widget: Gtk.Widget, ctx: cairo.Context) -> bool:
        result = self.on_main_draw(widget, ctx)
        
        return result

    def on_main_destroy(self, widget: Gtk.Widget): ...


class WindowBackend(Window, Gtk.OffscreenWindow):
    def __init__(
        self,
        emu,
        width,
        height,
        parent_uuid: str,
        uuid: Optional[str] = None,
        native_scale=1,
        screen_id: int = 0,
    ):
        Gtk.OffscreenWindow.__init__(self)
        Window.__init__(
            self, width=width, height=height, native_scale=native_scale, uuid=uuid
        )

        self.set_default_size(width, height)
        self.native_scale = native_scale
        self.parent_uuid = parent_uuid
        self._screen_id = screen_id
        self._emu = emu
        self.overlay_container = Gtk.Overlay()
        self.add(self.overlay_container)
        self.overlay_container.add(self.drawing_area)
        self.show_all()
        self.drawing_area.set_size_request(
            width * self.native_scale, height * self.native_scale
        )
        self.drawing_area.queue_resize()

    def on_main_destroy(self, widget: Gtk.Widget):
        self._on_main_destroy()

    def _on_main_destroy(self):
        for overlay in self.overlay_container.get_children():
            cast(AsyncOverlay, overlay).end()

        self.close()
        Gtk.main_quit()

    @property
    def layers(self):
        return [overlay for overlay in self.overlay_container.get_children() if isinstance(overlay, AsyncOverlay)]

    @property
    def running(self):
        return all([overlay._running for overlay in self.layers])

    def get_parent_status(self) -> bool:
        shm = SharedMemory(name=str(self.parent_uuid), size=1)
        assert shm.buf is not None
        return shm.buf[0] != 1

    def add_background_layer(self, overlay_cls: type[AsyncOverlay], torch_device):
        overlay = overlay_cls(
            emu=self._emu,
            width=self.width,
            height=self.height,
            scale=self.native_scale,
            device=torch_device,
            n_bytes=self.n_bytes,
            refresh_rate=OVERLAY_REFRESH_RATE,
            uuid=self.uuid,
        )
        overlay.set_can_focus(False)
        self.overlay_container.add_overlay(overlay)
        self.overlay_container.set_overlay_pass_through(overlay, True)

    def start_background_layers(self):
        for overlay in self.layers:
            overlay.show()
            overlay.start()

        self.show_all()

    def get_screen_id(self) -> int:
        return self._screen_id

    def on_main_draw(self, widget: Gtk.Widget, ctx: cairo.Context) -> bool:
        if not self.get_parent_status():
            return False

        if self.get_screen_id() == 0:
            game_buf = self._emu.display_buffer_as_rgbx()[: SCREEN_PIXEL_SIZE * 4]
        else:
            game_buf = self._emu.display_buffer_as_rgbx()[SCREEN_PIXEL_SIZE * 4 :]

        def_width, def_height = self.get_default_size()
        game_buf_surface = cairo.ImageSurface.create_for_data(
            game_buf,
            cairo.FORMAT_RGB24,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )

        assert self.shm.buf
        shm_surface = cairo.ImageSurface.create_for_data(
            self.shm.buf,
            cairo.FORMAT_RGB24,
            def_width * self.native_scale,
            def_height * self.native_scale,
        )
        shm_ctx = cairo.Context(shm_surface)
        shm_ctx.scale(self.native_scale, self.native_scale)
        shm_ctx.set_source_surface(game_buf_surface)
        shm_ctx.get_source().set_filter(cairo.FILTER_NEAREST)
        shm_ctx.paint()

        
        

        return True


class WindowFrontend(Window, Gtk.Overlay):
    def __init__(
        self, width, height, scale, native_scale, uuid: str, label: Optional[str] = None
    ):
        Gtk.Overlay.__init__(self)
        Window.__init__(
            self, width=width, height=height, native_scale=native_scale, uuid=uuid
        )
        self.drawing_area.set_size_request(width * scale, height * scale)
        self.scale = scale
        self.native_scale = native_scale
        self.label = Gtk.Label(label=(label or DEFAULT_WINDOW_NAME))
        self.label.set_name("emulator-label")
        self.label.set_halign(Gtk.Align.START)
        self.label.set_valign(Gtk.Align.START)
        style_provider = Gtk.CssProvider()
        style_provider.load_from_data(DEFAULT_WINDOW_STYLE)
        default = Gdk.Screen.get_default()
        assert default
        Gtk.StyleContext.add_provider_for_screen(
            default, style_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # add this method to match the backend method
        self.label_color = (0, 0, 0)
        self.get_default_size = lambda: (width, height)
        self.add_overlay(self.label)
        self.add(self.drawing_area)

    def set_label_color(self, color: tuple[float, float, float]):
        rgba = Gdk.RGBA(*color, 1.0)
        self.label_color = color
        self.label.override_background_color(Gtk.StateFlags.NORMAL, rgba)

    def on_main_draw(self, widget: Gtk.Widget, ctx: cairo.Context) -> bool:
        def_width, def_height = self.get_default_size()

        assert self.shm.buf
        shm_surface = cairo.ImageSurface.create_for_data(
            self.shm.buf,
            cairo.FORMAT_RGB24,
            def_width * self.native_scale,
            def_height * self.native_scale,
        )

        scale = self.scale / self.native_scale
        ctx.save()
        ctx.scale(scale, scale)
        ctx.set_source_surface(shm_surface, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_NEAREST)
        ctx.paint()
        ctx.restore()
        self.drawing_area.queue_draw()

        ctx.scale(self.scale, self.scale)
        ctx.set_source_rgb(*self.label_color)
        ctx.set_line_width(4.0)
        ctx.rectangle(0, 0, def_width, def_height)
        ctx.stroke()

        return True


class ProcessStatus(TypedDict):
    parent_uuid: str
    uuid: str
    status: str
    error_signature: NotRequired[str]
    error_traceback: NotRequired[str]


@dataclass
class EmulatorConfig:
    rom_path: Path
    record: Optional[Path]
    record_data: Optional[Path]
    fps: Optional[float]
    movie: Optional[Path]
    sram: Optional[Path]
    savestate: Optional[Path]
    force_overlay: bool
    verbose: Optional[bool]
    model: Optional[Path]


def configure_emulator(emu: MarioKart, config: EmulatorConfig):
    emu.open(str(config.rom_path))
    emu.volume_set(0)  # mute volume so I don't go INSANE

    # Emulator speed
    fps = 60.0  # default fps when not playing .dsm
    if config.fps:
        fps = float(config.fps)

    # DeSmuME movie replay
    # listener, input_state = None, None # realtime keyboard input unnecessary during movie playback
    if config.movie:
        print(config.movie)
        emu.movie.play(str(config.movie))

        # DeSmuME ROM Save file (critical for loading external .dsm files)
        if config.sram:
            emu.backup.import_file(str(config.sram))

    # DeSmuME movie record
    if config.record is not None:
        emu.movie.record(
            str(config.record), author_name="test", sram_save=str(config.sram)
        )

    # DeSmuME load from savestate
    if config.savestate:
        assert config.movie, "savestates cannot be loaded during movie replays"
        emu.savestate.load_file(str(config.savestate))

    # Record training data
    if config.record_data is not None:
        # speed up fps by default if recording training data from desmume movie replay
        if not (config.fps and config.movie):
            fps = 240.0

        # specify the directory that stores all training directories
        output_dir = (
            str(config.record_data) if str(config.record_data) != "." else os.getcwd()
        )
    else:
        output_dir = os.getcwd()

    assert fps > 0.0, "fps must be greater than 0"

    # Extra logic for recording training data, specifically for saving all .dat files
    if config.movie:
        output_base = os.path.basename(config.movie).split(".")[0]
    else:
        output_base = f"TRAINING_{datetime.today().strftime('%Y%m%d')}"

    dataset_path = None
    if config.record_data:
        dataset_path = f"{output_dir}/{output_base}"
        emu.record_dataset(dataset_path)

    if config.verbose:
        print(
            f"""
    DeSmuME Savestate:           {config.savestate if config.savestate else "None"}
    DeSmuME Movie (loading):     {config.movie if config.movie else "None"}
    DeSmuME Movie (saving):      {config.record if config.record else "None"}
    Training Data Directory:     {dataset_path if dataset_path else "None"}
    FPS:                         {fps:.1f}
        """
        )

    config.fps = fps


def get_error_signature(exception: Exception) -> str:
    # TracebackException captures the type, message, and stack frame info
    te = traceback.TracebackException.from_exception(exception)

    # Get the last frame of the traceback (where the error actually happened)
    last_frame = te.stack[-1]

    # Create a unique string: "ExceptionType: Message at filename.py:line_number"
    return f"{type(exception).__name__}: {str(exception)} at {last_frame.filename}:{last_frame.lineno}"


# async keyboard handler
def start_keyboard_listener():
    """Starts a non-blocking keyboard listener in a separate thread."""
    input_state = set()

    def on_press(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            input_state.add(name)

        except Exception:
            pass

    def on_release(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            input_state.discard(name)
        except Exception:
            pass

    listener = pynput.keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    return listener, input_state


T = TypeVar("T", bound=EmulatorConfig, covariant=True)


class EmulatorCallable[T](Protocol):
    def __call__(
        self, emu: MarioKart, config: T, input_state: Optional[set[str]]
    ) -> Any: ...


def instance_wrapper(
    func: EmulatorCallable[T],
    parent_uuid: str,
    proc_uuid: str,
    queue: Queue,
    overlays: list[type[AsyncOverlay]],
    native_scale,
    config: T,
):
    status_sent = False

    def put_status(label="completed", error=None):
        nonlocal status_sent
        if status_sent:
            return
        data: ProcessStatus = {
            "parent_uuid": parent_uuid,
            "uuid": proc_uuid,
            "status": label,
        }
        if error:
            signature = get_error_signature(e)
            full_tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            data["error_signature"] = signature
            data["error_traceback"] = full_tb

        queue.put(data)
        status_sent = True

    emu = MarioKart(has_grad=False, device=BACKEND_DEVICE)
    try:
        win = WindowBackend(
            emu,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            parent_uuid,
            uuid=proc_uuid,
            native_scale=native_scale,
        )
        win_b = WindowBackend(
            emu,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            parent_uuid,
            uuid=f"{proc_uuid}_B",
            native_scale=native_scale,
            screen_id=1
        )
        for o in overlays:
            win.add_background_layer(o, torch_device=BACKEND_DEVICE)

        def monitor_state():
            if not win.get_parent_status():
                return False

            if not win.running and emu.memory.race_ready:
                win.start_background_layers()
                return False

            return True
            
        def tick(input_state):
            GLib.idle_add(emu.cycle)
            win.drawing_area.queue_draw()
            win_b.drawing_area.queue_draw()
            return func(emu, config, input_state)

        # initialize emulator and keyboard inputs (keyboard is optional)
        configure_emulator(emu, config)
        assert config.fps
        listener, input_state = start_keyboard_listener()
        GLib.timeout_add(16, monitor_state)
        GLib.timeout_add(int(1000.0 // config.fps), tick, input_state)
        def destroy():
            win._on_main_destroy()
            put_status()
        win.connect("destroy", destroy)
        win.show_all()
        Gtk.main()
        listener.stop()
    except Exception as e:
        put_status(label="error", error=e)
        emu.close()
    finally:
        Gtk.main_quit()


class EmulatorBackend:
    def __init__(self, native_scale: int):
        self.uuid = generate()
        self.native_scale = native_scale
        self.stop_shm = SharedMemory(name=self.uuid, create=True, size=1)
        self.manager = Manager()
        self.processes: dict[str, Process] = {}
        self.status_queue: Queue[ProcessStatus] = Queue()
        assert self.stop_shm.buf is not None
        self.stop_shm.buf[0] = 0

    def add_instance(
        self,
        func: EmulatorCallable[T],
        config: T,
        overlays: list[type[AsyncOverlay]] = [],
    ):
        parent_uuid = self.uuid
        proc_uuid = generate()
        proc = Process(
            target=instance_wrapper,
            args=(
                func,
                parent_uuid,
                proc_uuid,
                self.status_queue,
                overlays,
                self.native_scale,
                config,
            ),
        )
        self.processes[proc_uuid] = proc

    @property
    def is_alive(self):
        return any([proc.is_alive() for proc in self.processes.values()])

    def start(self):
        for p in self.processes.values():
            p.start()

    def stop(self):
        assert self.stop_shm.buf
        self.stop_shm.buf[0] = 1
        


class EmulatorFrontend(Gtk.Window):
    def __init__(self, backend: EmulatorBackend, scale: int):
        super().__init__()
        self.set_title("Multiview")
        self.grid = Gtk.Grid()
        self.scale = scale
        self.native_scale = backend.native_scale
        self.n_windows = len(backend.processes)
        self.backend = backend
        self.errors = {}

        palette = palette_gen(self.n_windows)
        for i, uuid in enumerate(backend.processes.keys()):
            win = WindowFrontend(
                SCREEN_WIDTH,
                SCREEN_HEIGHT,
                scale=self.scale,
                native_scale=self.native_scale,
                uuid=uuid,
                label=f"Instance {i+1}",
            )
            win.set_label_color(palette[i])
            col, row = self.assign_grid_coords(i*2)
            self.grid.attach(win, col, row, 1, 1)
            
            win_b = WindowFrontend(
                SCREEN_WIDTH,
                SCREEN_HEIGHT,
                scale=self.scale,
                native_scale=self.native_scale,
                uuid=f"{uuid}_B",
                label=f"Instance {i+1} UI",
            )
            win_b.set_label_color(palette[i])
            col, row = self.assign_grid_coords(i*2+1)
            self.grid.attach(win_b, col, row, 1, 1)
        

        self.set_title(f"Multiview {len(self.grid.get_children())}")
        self.add(self.grid)
        self.connect("destroy", self._on_destroy)
        GLib.timeout_add(16, self.refresh_screens)
        self.show_all()

    def update_colors(self):
        palette = palette_gen(len(self.grid.get_children()))
        for w, c in zip(cast(list[WindowFrontend], self.grid.get_children()), palette):
            w.set_label_color(c)

    def assign_grid_coords(self, index: int) -> tuple[int, int]:
        n_cols = math.ceil(math.sqrt(self.n_windows * 2))
        col = index % n_cols
        row = index // n_cols
        return col, row

    def refresh_screens(self):
        prunable_proc_uuids = set()
        while not self.backend.status_queue.empty():
            data = self.backend.status_queue.get_nowait()

            if data["status"] == "error":
                assert "error_signature" in data
                assert "error_traceback" in data
                sig = data["error_signature"]

                if sig not in self.errors:
                    self.errors[sig] = {
                        "priority": len(self.errors),
                        "traceback": data["error_traceback"],
                        "uuids": [],
                    }

                self.errors[sig]["uuids"].append(data["uuid"])

            prunable_proc_uuids.add(data["uuid"])

        for uuid in prunable_proc_uuids:
            self.backend.processes[uuid].join()
            shm = SharedMemory(name=uuid)
            shm.close()
            shm.unlink()

        if not self.backend.is_alive:
            self.on_destroy()

        return True

    def start(self):
        try:
            self.show_all()
            Gtk.main()
        except KeyboardInterrupt:
            self.shutdown()

    def on_destroy(self):
        Gtk.main_quit()
        self.backend.stop()
        errors = list(self.errors.items())
        for sig, data in sorted(errors, key=lambda x: x[1]["priority"]):
            print(f"(Priority {data['priority']}) {sig}")
            print(f"Processes ({len(data['uuids'])}): {data['uuids']}")
            print(RuntimeError(data["traceback"]))

    def _on_destroy(self, widget: Gtk.Widget):
        self.on_destroy()

    # emergency shutdown (for exit key / sigint)
    def shutdown(self):
        for uuid, proc in self.backend.processes.items():
            if proc.is_alive():
                proc.join()

            try:
                shm = SharedMemory(name=uuid)
                shm.close()
                shm.unlink()
                shm = SharedMemory(name=f"{uuid}_B")
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                print(
                    "Warning: emergency release of process resources failed! Shared memory was likely leaked!\n\n"
                )
                raise e

        self.backend.stop()
