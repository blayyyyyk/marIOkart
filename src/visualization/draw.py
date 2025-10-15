from __future__ import annotations
import torch
from cairo import Context
import cairo
from typing import ParamSpec, TypeVar, Callable, Concatenate
import numpy as np
from queue import Queue, Empty
from threading import Lock

dot_radius = 5


draw_queue: Queue[Callable[[Context], None]] = Queue()


P = ParamSpec('P')
R = TypeVar('R')
def draw_stack_op(func: Callable[Concatenate[Context, P], None]) -> Callable[P, None]:
    global draw_queue
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        global draw_queue
        def sub_wrapper(ctx: Context):
            nonlocal args, kwargs
            func(ctx, *args, **kwargs)

        draw_queue.put(sub_wrapper)
    return wrapper

@draw_stack_op
def draw_points(ctx: Context, pts: np.ndarray, colors: np.ndarray, radius_scale: float | np.ndarray):
    if isinstance(radius_scale, float):
        radius_scale = radius_scale * np.array(1)

    if pts.ndim == 1:
        pts = pts[None, :]

    if colors.ndim == 1:
        colors = colors[None, :]

    assert colors.shape[0] == 1 or colors.shape[0] == pts.shape[0]
    if colors.shape[0] == 1:
        colors = colors.repeat(pts.shape[0], axis=0)

    for (x, y, z), (r, g, b) in zip(pts, colors):
        ctx.set_source_rgb(r, g, b)
        ctx.arc(x, y, radius_scale * z, 0, 2 * np.pi)
        ctx.fill()



@draw_stack_op
def draw_lines(ctx: Context, pts1: np.ndarray, pts2: np.ndarray, colors: np.ndarray, stroke_width_scale=1.0):
    if pts1.ndim == 1:
        pts1 = pts1[None, :]

    if pts2.ndim == 1:
        pts2 = pts2[None, :]

    assert pts2.shape[0] == pts1.shape[0], "All point arrays must have the same batch size"

    if colors.ndim == 1:
        colors = colors[None, :]

    assert colors.shape[0] == 1 or colors.shape[0] == pts1.shape[0]
    if colors.shape[0] == 1:
        colors = colors.repeat(pts1.shape[0], axis=0)

    for (p1, p2, (r, g, b)) in zip(pts1, pts2, colors):
        ctx.set_source_rgb(r, g, b)
        ctx.set_line_width(stroke_width_scale)
        ctx.move_to(*p1[:2])
        ctx.line_to(*p2[:2])
        ctx.stroke()

@draw_stack_op
def draw_triangles(
    ctx: Context,
    pts1: np.ndarray,
    pts2: np.ndarray,
    pts3: np.ndarray,
    colors: np.ndarray
):
    n = pts1.shape[0]
    assert pts2.shape[0] == n and pts3.shape[0] == n, "All point arrays must have the same batch size"

    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = colors[None, :]
    assert colors.shape[1] == 3, "colors must have 3 channels (RGB)"
    if colors.shape[0] == 1:
        colors = np.repeat(colors, n, axis=0)
    else:
        assert colors.shape[0] == n, "colors must be [1,3] or [N,3]"

    l1 = np.concatenate([pts2, pts3, pts1], axis=0)  # p2->p3, p3->p1, p1->p2
    l2 = np.concatenate([pts3, pts1, pts2], axis=0)
    c3 = np.tile(colors, (3, 1))

    draw_lines(l1, l2, c3)  # assumes draw_lines accepts NumPy arrays


def draw_text(ctx: Context, text: str, pos: tuple[float, float] = (0.0, 0.0), color: tuple[float, float, float] = (1.0, 0.5, 0.34), alpha: float = 1.0, font_size: float = 12, font_family: str = "Sans"):
    ctx.save()
    ctx.set_source_rgba(*color, alpha)
    ctx.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(font_size)
    ctx.move_to(*pos)
    ctx.show_text(text)
    ctx.restore()

def draw_paragraph(ctx: Context, text: str, pos: tuple[float, float] = (0.0, 0.0), color: tuple[float, float, float] = (1.0, 0.5, 0.34), alpha: float = 1.0, font_size: float = 12, vertical_spacing: float = 10, font_family: str = "Sans"):
    texts = text.split("\n")
    for i, t in enumerate(texts):
        draw_text(ctx, t, (pos[0], pos[1] + i * vertical_spacing), color, alpha, font_size, font_family)

def consume_draw_stack(ctx: Context, max_items: int | None = None) -> int:
    count = 0

    while True:
        if max_items is not None and count >= max_items:
            break
        try:
            fn = draw_queue.get_nowait()
        except Empty:
            break

        fn(ctx)
        count += 1

    return count
