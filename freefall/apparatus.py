"""The drop apparatus: a height knob, a push button and a timer.

Everything that knows about the physics lives here. The tools, the loop and
the final check use only the Apparatus interface, so the simulation can be
swapped for a real rig (a release mechanism on a height stage, a strike
plate and a timer) without touching anything else:

    set_height(h)  move the release point, block until there, return where it is
    get_height()   where the release point is now
    drop()         release the ball and return the timer's reading, seconds
    close()        park the rig

The simulation: t = sqrt(2 h / g), with g = 9.8 m/s^2, plus Gaussian noise
on every timing.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ApparatusError(RuntimeError):
    """A request the apparatus cannot carry out. The message is shown to the model."""


class Apparatus(ABC):
    min_height: float = 0.1     # meters
    max_height: float = 2.0
    units: str = "s"

    @abstractmethod
    def set_height(self, h: float) -> float:
        """Move the release point to h meters; return where it actually is."""

    @abstractmethod
    def get_height(self) -> float:
        """The current release height, meters."""

    @abstractmethod
    def drop(self) -> float:
        """Release the ball; return the timer's reading in seconds."""

    def close(self) -> None:
        """Park the rig. No-op by default."""

    def check_height(self, h: float) -> float:
        try:
            h = float(h)
        except (TypeError, ValueError):
            raise ApparatusError(f"The knob needs a number, not {h!r}.") from None
        if not math.isfinite(h):
            raise ApparatusError("The knob needs a finite number.")
        if h < self.min_height:
            raise ApparatusError(f"The knob cannot go below {self.min_height:g}.")
        if h > self.max_height:
            raise ApparatusError(f"The knob cannot go above {self.max_height:g}.")
        return h


@dataclass
class DropConfig:
    """How the simulated rig behaves.

    g               acceleration of the fall, m/s^2
    noise_s         standard deviation of each timing, seconds
    timer_offset_s  a constant added to every timing (release latency, a slow
                    strike plate); 0 = a perfect timer
    start_height    where the knob sits before the model first turns it
    """

    g: float = 9.8
    noise_s: float = 0.002
    timer_offset_s: float = 0.0
    start_height: float = 0.1


class SimulatedDropApparatus(Apparatus):
    def __init__(self, config: DropConfig | None = None, seed: int | None = None):
        self.config = config or DropConfig()
        self._rng = random.Random(seed)
        self._h = self.check_height(self.config.start_height)

    def set_height(self, h: float) -> float:
        self._h = self.check_height(h)
        return self._h

    def get_height(self) -> float:
        return self._h

    def true_time(self, h: float | None = None) -> float:
        """The noiseless fall time (simulation only)."""
        h = self._h if h is None else h
        return math.sqrt(2.0 * h / self.config.g) + self.config.timer_offset_s

    def drop(self) -> float:
        return self.true_time() + self._rng.gauss(0.0, self.config.noise_s)


class HardwareApparatus(Apparatus):
    """The real rig. Fill these in for the height stage, release and timer."""

    def set_height(self, h: float) -> float:
        raise NotImplementedError("HardwareApparatus.set_height: drive the height stage.")

    def get_height(self) -> float:
        raise NotImplementedError("HardwareApparatus.get_height: read the height stage.")

    def drop(self) -> float:
        raise NotImplementedError("HardwareApparatus.drop: release the ball, read the timer.")
