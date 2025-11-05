"""Pluggable metric collectors and fitness scoring for MKDS training.

This module defines a small, picklable interface for **episode-level metrics**
that can be attached to the emulator loop without modifying the core trainer.
Each metric implements a three-phase lifecycle:

1) `reset()` — called once at the start of an episode to clear state.
2) `update(emu, device)` — called every frame to accumulate data.
3) `collect()` — called once at episode end to return scalar summaries.

Design notes:
    * **Picklability**: All metric implementations are top-level classes with
      minimal state (floats/ints/tensors) so they can be sent to worker
      processes via `multiprocessing` (spawn or fork).
    * **Independence**: Metrics do not modify emulator state or controls; they
      only *observe* via functions provided by `src.core.memory`.
    * **Units**:
        - Distances are in MKDS "world units" (derived from FX32).
        - Time `read_clock(emu)` is returned in **centiseconds** (10 ms units).
        - If you compute speed as `distance / centiseconds`, multiply by `100`
          to convert to per-second rates.

Helper functions `reset_all(...)` and `collect_all(...)` operate on a list
of `Metric` instances to simplify orchestration. A `FitnessScorer` protocol
is provided to decouple metric collection from scalar fitness scoring.

Typical usage:
    >>> metrics = [DistanceMetric(), OffroadMetric()]
    >>> reset_all(metrics)
    >>> while episode_running:
    ...     # run emulator step, inference, controls, etc.
    ...     for m in metrics:
    ...         m.update(emu, device=device)
    >>> summary = collect_all(metrics)  # {"distance": ..., "offroad_dist": ...}
    >>> fitness = default_fitness_scorer(summary)  # plug into selection
"""

from __future__ import annotations
from typing import Any, Protocol, Self
from abc import abstractmethod
from desmume.emulator import DeSmuME
from torch._prims_common import DeviceLikeType
from src.core.memory import *


class Metric:
    """Interface for any metric collector used during training.

    Each metric observes emulator state every frame and exposes a scalar
    summary at episode end. Implementations should avoid heavy allocations or
    device transfers in `update`, and should keep internal state simple so the
    object remains cheap to pickle across processes.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of an episode.

        Implementations must clear any rolling counters, cached positions, or
        flags so that a fresh episode starts from a known baseline. This method
        is called exactly once per episode, prior to the first `update`.
        """
        ...

    @abstractmethod
    def update(self, emu: DeSmuME, device) -> None:
        """Record or accumulate values for the current frame.

        This method is called once per emulator tick/frame. It should read any
        required game state via `src.core.memory` helpers and update internal
        counters accordingly.

        Args:
            emu: Live `DeSmuME` emulator instance for the current process.
            device: Optional Torch device hint if your reads/ops require a
                specific device. Many `src.core.memory` functions accept this.
        """
        ...

    @abstractmethod
    def collect(self) -> dict[str, float]:
        """Return scalar summary values at the end of the episode.

        Implementations should convert any tensor values to Python floats and
        return them under stable, unique keys. The convention in this module is
        to return exactly **one** key/value pair per metric.

        Returns:
            A dictionary mapping the metric's name to a scalar float value.
        """
        ...

class DistanceMetric(Metric):
    """Signed checkpoint-to-checkpoint progress along the course.

    This metric treats the checkpoint chain as a polyline and accumulates
    signed distance when crossing from one checkpoint segment to an adjacent
    one:

    * Moving forward (current ➔ next) **adds** the midpoint-to-midpoint
      segment length.
    * Moving backward (current ➔ previous) **subtracts** that length.

    This approach is:
        - **Robust to FPS**: updates only on checkpoint transitions, not
          per-frame arc length integration.
        - **Direction-aware**: penalizes reversing or losing progress.
        - **Course-agnostic**: relies only on NKM checkpoints you already load.

    Note:
        The accumulated value is a *coarse* approximation of path length since
        it uses straight-line distances between segment midpoints, not the
        exact racing line.
    """

    def __init__(self):
        """Initialize internal state."""
        self.prev_id: int | None = None
        self.curr_id: int | None = None
        self.next_id: int | None = None
        self.dist: float = 0

    def reset(self) -> None:
        """Clear progress counters and current checkpoint ID."""
        self.curr_id = None
        self.dist = 0

    def update(self, emu: DeSmuME, device: DeviceLikeType | None = None) -> None:
        """Accumulate signed progress on checkpoint transitions.

        Logic:
            1) On first call, caches `curr/prev/next` checkpoint IDs and returns.
            2) If the player remains on the same checkpoint, does nothing.
            3) If the checkpoint index changes:
                - Compute the midpoint of the *previous* segment and the
                  midpoint of the newly-entered segment.
                - Add their distance if we moved forward (curr ➔ next).
                - Subtract if we moved backward (curr ➔ prev).
                - Update the cached `curr/prev/next` IDs.

        Args:
            emu: Emulator instance.
            device: Optional Torch device used by memory readers.

        Raises:
            AssertionError: If midpoints could not be computed (should not
                happen under normal circumstances given valid NKM data).
        """
        current_checkpoint_id = read_current_checkpoint(emu)
        nkm = load_current_nkm(emu, device=device)
        checkpoint_count = nkm._CPOI.entry_count

        if self.curr_id is None:
            self.curr_id = current_checkpoint_id
            self.prev_id = read_previous_checkpoint(emu, checkpoint_count)
            self.next_id = read_next_checkpoint(emu, checkpoint_count)
            return

        if self.curr_id == current_checkpoint_id:
            return

        scale = 1
        midpoint_1 = read_current_checkpoint_position(emu, device=device).sum(dim=0) / 2
        midpoint_2 = None

        if current_checkpoint_id == self.next_id:
            midpoint_2 = read_next_checkpoint_position(emu, device=device).sum(dim=0) / 2
        elif current_checkpoint_id == self.prev_id:
            midpoint_2 = read_previous_checkpoint_position(emu, device=device).sum(dim=0) / 2
            scale = -1

        assert midpoint_1 is not None and midpoint_2 is not None, "Midpoints should be calculated"
        self.dist += scale * torch.norm(midpoint_1 - midpoint_2).item()

        self.curr_id = current_checkpoint_id
        self.prev_id = read_previous_checkpoint(emu, checkpoint_count)
        self.next_id = read_next_checkpoint(emu, checkpoint_count)

    def collect(self) -> dict[str, float]:
        """Return the signed checkpoint progress accumulated this episode.

        Returns:
            A dict with a single key:
                - `"distance"`: Signed scalar world-units progressed, positive
                  when moving forward through checkpoints, negative when
                  moving backward.
        """
        return {
            'distance': self.dist
        }


class SpeedMetric(Metric):
    """Average episode speed computed from a `DistanceMetric`.

    This metric divides the total signed distance reported by a companion
    `DistanceMetric` by the elapsed clock time between the **first observed
    checkpoint** and the episode end.

    Important:
        * `read_clock(emu)` returns **centiseconds**. As implemented, the
          resulting `speed` is in **world-units per centisecond**. Multiply by
          `100.0` to obtain world-units per second.
        * This is an episode-level average; it does not reflect instantaneous
          speed or time spent stationary before the first checkpoint crossing.
    """

    def __init__(self, distance_metric: DistanceMetric):
        """Create a `SpeedMetric`.

        Args:
            distance_metric: A live `DistanceMetric` instance whose `dist`
                field will be read during `update`/`collect`. This is injected
                so the trainer can share one distance computation across
                multiple metrics.
        """
        self.distance_metric = distance_metric
        self.speed: float = 0.0
        self.start_time = 0

    def reset(self):
        """Clear the time baseline and current speed."""
        self.start_time = 0

    def update(self, emu: DeSmuME, device: DeviceLikeType | None = None) -> None:
        """Update the average speed from current distance and clock.

        Behavior:
            * Before any checkpoint is observed by `DistanceMetric`, captures
              a `start_time` and waits.
            * Afterward, divides the current `distance_metric.dist` by
              `(read_clock(emu) - start_time)` (centiseconds).

        Args:
            emu: Emulator instance.
            device: Unused; present for interface parity.
        """
        if self.distance_metric.curr_id is None:
            self.start_time = read_clock(emu)
            return

        end_time = read_clock(emu)
        if end_time - self.start_time == 0:
            return

        self.speed = self.distance_metric.dist / (end_time - self.start_time)

    def collect(self) -> dict[str, float]:
        """Return the episode-average speed.

        Returns:
            A dict with a single key:
                - `"speed"`: Average speed in world-units per **centisecond**.
                  Multiply by `100` to convert to world-units per second.
        """
        return {
            'speed': self.speed
        }


class OffroadMetric(Metric):
    """Approximate distance traveled while the kart is offroad.

    This metric toggles an internal `is_offroad` flag based on the ground
    prism's collision type under the player (using `read_touching_prism_type`
    with an attribute mask matching known offroad IDs {2, 3, 5}).

    When the kart transitions:
        * **Onroad ➔ Offroad**: caches the current position.
        * **Offroad ➔ Onroad**: adds the straight-line displacement between
          the cached entry position and the current position to `offroad_dist`.

    Caveat:
        The accumulated distance is a straight-line approximation between
        entry and exit points; it does **not** integrate arc length over time
        while offroad. For tight zig-zags offroad, it will **underestimate**
        true path length.
    """

    def __init__(self):
        """Initialize internal counters."""
        self.offroad_dist = 0.0

    def reset(self) -> None:
        """Clear cached position, offroad distance, and offroad state."""
        self.prev_position = None
        self.offroad_dist = 0.0
        self.is_offroad = False

    def update(self, emu: DeSmuME, device: DeviceLikeType | None = None):
        """Update offroad tracking based on current surface attribute.

        Args:
            emu: Emulator instance.
            device: Torch device forwarded to memory helpers.

        Behavior:
            * If this is the first frame, caches position and returns.
            * Determines if the current surface is offroad via an attribute
              mask: `(attr == 3) | (attr == 2) | (attr == 5)`.
            * If the offroad state is unchanged, does nothing.
            * On **entering** offroad, caches the current position.
            * On **exiting** offroad, adds the straight-line distance between
              the cached entry position and the current position to
              `offroad_dist`, then updates `is_offroad`.
        """
        position = read_position(emu, device=device)
        if self.prev_position is None:
            self.prev_position = position
            return

        attr_mask = lambda x: ((x == 3) | (x == 2) | (x == 5))
        current_is_offroad = read_touching_prism_type(emu, attr_mask, device=device)

        if current_is_offroad == self.is_offroad:
            return

        if current_is_offroad:
            self.prev_position = position
        elif not current_is_offroad:
            self.offroad_dist += torch.norm(position - self.prev_position).item()

        self.is_offroad = current_is_offroad

    def collect(self) -> dict[str, float]:
        """Return the approximate offroad distance traveled.

        Returns:
            A dict with a single key:
                - `"offroad_dist"`: Approximate straight-line distance covered
                  while offroad during the episode, in world units.
        """
        return {
            'offroad_dist': self.offroad_dist
        }


def collect_all(metrics: list[Metric]):
    """Collect scalar summaries from a list of metrics.

    This is a convenience that merges per-metric outputs into a single flat
    dictionary. It assumes each metric returns exactly one key/value pair and
    uses the **first** item of the returned dict.

    Args:
        metrics: List of initialized metric instances that have already been
            `reset()` and `update(...)`-ed for the episode.

    Returns:
        A dictionary mapping metric names to scalar floats, e.g.
        `{"distance": 123.4, "offroad_dist": 5.6}`.

    Example:
        >>> collect_all([DistanceMetric(), OffroadMetric()])
        {'distance': 0.0, 'offroad_dist': 0.0}
    """
    out: dict[str, float] = {}
    for metric in metrics:
        d = metric.collect()
        k, v = list(d.items())[0]
        out[k] = v

    return out


def reset_all(metrics: list[Metric]):
    """Reset all metrics in-place.

    Calls `reset()` on each metric in order. Useful at the start of every
    episode to ensure clean state.

    Args:
        metrics: List of metric instances to be reset.
    """
    for metric in metrics:
        metric.reset()


class FitnessScorer(Protocol):
    """Callable protocol for converting metric dicts into scalar fitness.

    A `FitnessScorer` consumes the merged output of `collect_all(...)` and
    returns a single float suitable for selection and ranking in an
    evolutionary or RL loop.

    Example:
        >>> def scorer(m): return m['distance'] - 10.0*m.get('offroad_dist', 0.0)
        >>> isinstance(scorer, FitnessScorer)
        True
    """

    def __call__(self, metrics: dict[str, float]) -> float:
        """Compute a scalar fitness from metric summaries.

        Args:
            metrics: Flat dictionary of scalar metrics (e.g., the result of
                `collect_all(...)`).

        Returns:
            A single float representing the individual's fitness.
        """
        ...


def default_fitness_scorer(metrics: dict[str, float]) -> float:
    """Return a distance-only fitness score.

    This default scorer simply returns the `"distance"` metric. It is intended
    as a minimal baseline and may raise a `KeyError` if the `"distance"` key
    is absent.

    Args:
        metrics: Flat dictionary of scalar metrics.

    Returns:
        The value associated with the `"distance"` key.

    Raises:
        KeyError: If `"distance"` is not present in `metrics`.
    """
    return metrics['distance']
