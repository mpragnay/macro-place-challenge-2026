"""
Initial Positions Visualizer

Returns macros at their initial positions (no movement).
Use this to visualize the starting state of a benchmark.

Usage:
    uv run evaluate submissions/examples/initial_positions.py --vis
    uv run evaluate submissions/examples/initial_positions.py -b ibm03 --vis
"""

import torch
from macro_place.benchmark import Benchmark


class InitialPositions:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return benchmark.macro_positions.clone()
