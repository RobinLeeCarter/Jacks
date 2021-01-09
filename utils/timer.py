from dataclasses import dataclass
import time
from typing import List


@dataclass
class Lap:
    lap_start: float
    lap_end: float
    lap_name: str

    @property
    def lap_time(self) -> float:
        return self.lap_end - self.lap_start

    def print(self):
        print(f"{self.lap_name}:\t{self.lap_time:.3f}")


class Timer:
    def __init__(self):
        self.start_time: float = time.perf_counter()
        self.lap_start: float = self.start_time
        self.end_time: float = 0
        self.laps: List[Lap] = []
        self.running: bool = False

    @property
    def total(self) -> float:
        if self.end_time != 0:
            return self.end_time - self.start_time
        else:
            return 0.0

    def start(self):
        self.running = True
        self.laps.clear()
        self.start_time = time.perf_counter()
        self.lap_start = self.start_time

    def lap(self, name: str = "", show: bool = False) -> float:
        lap_end = time.perf_counter()
        lap = Lap(self.lap_start, lap_end, name)
        self.laps.append(lap)
        if show:
            lap.print()

        self.lap_start = lap_end
        return lap.lap_time

    def stop(self, name: str = "", show=True) -> float:
        self.end_time = time.perf_counter()
        self.running = False
        if show:
            print("")
            for lap in self.laps:
                lap.print()
            self.print(name)
        return self.total

    def show_cumulative_percentage(self):
        total: float = self.total
        cumulative: float = 0.0
        for lap in self.laps:
            cumulative += lap.lap_time
            cumulative_percentage = 100 * (cumulative / total)
            print(f"{lap.lap_name}:\t{cumulative_percentage:.0f}%")

    def print(self, name: str):
        if name:
            print(f"{name} :\t{self.total:.3f}")
        else:
            print(f"Total time:\t{self.total:.3f}")
