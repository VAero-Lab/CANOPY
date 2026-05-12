"""
Fractal Wing — tree-branch wing structure generator.

Public API
----------
Geometry:
    NACAFourDigit, Wing

Data Structures:
    Seg, SubParams, BranchDef, Station, TrunkSpec

Generator:
    TreeGenerator, ConvWB

Utilities:
    distribute_stations, Viz

Helpers:
    make_mixed_stations, make_diagonal_only_stations,
    make_zoned_stations, make_graded_stations
"""

from .structures import Seg, SubParams, BranchDef, Station, TrunkSpec
from .generator import TreeGenerator, ConvWB
from .distributions import distribute_stations
from .visualization import Viz
from .helpers import (
    make_mixed_stations,
    make_diagonal_only_stations,
    make_zoned_stations,
    make_graded_stations,
)
from .aeroshape_adapter import AeroWingAdapter
from .cad_export import build_brep_webs, export_hollow_skin
from .meshing import GmshMesher
from .fem_solver import build_ccx_deck
from .run_utils import run_ccx

__all__ = [
    'AeroWingAdapter',
    'Seg',
    'SubParams',
    'BranchDef',
    'Station',
    'TrunkSpec',
    'TreeGenerator',
    'ConvWB',
    'Viz',
    'make_diagonal_only_stations',
    'make_mixed_stations',
    'make_zoned_stations',
    'make_graded_stations',
    'build_brep_webs',
    'export_hollow_skin',
    'GmshMesher',
    'build_ccx_deck',
    'run_ccx',
]

__version__ = "0.1.0"
