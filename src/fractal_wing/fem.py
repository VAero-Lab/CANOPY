"""
FEM Structural Integration Module.

Provides functions to extract a clean structural topology (Nodes, Elements, Properties)
from the fractal wing segment geometry, suitable for Finite Element Method (FEM) 
shell modeling.
"""

from __future__ import annotations

import json
import numpy as np
from typing import List, Dict, Any

from .structures import Seg


def build_structural_graph(segs: List[Seg], tol: float = 1e-6) -> Dict[str, Any]:
    """
    Extract a mathematical graph (Nodes, Edges, Properties) from a list of segments.
    
    Coincident segment endpoints are merged into unique Nodes within the specified
    tolerance, ensuring there are no gaps in the resulting connectivity matrix.

    Parameters
    ----------
    segs : list of Seg
        The generated fractal segments.
    tol : float
        Tolerance for merging nodes. Default is 1e-6 meters (1 um).

    Returns
    -------
    dict
        A dictionary containing:
        - 'nodes': list of {"id": int, "x": float, "y": float}
        - 'elements': list of {"id": int, "n1": int, "n2": int}
        - 'properties': list of {"element_id": int, "thickness": float, "level": int}
    """
    nodes_list = []
    node_coords = []
    
    elements = []
    properties = []
    
    def get_node_id(p: np.ndarray) -> int:
        # Simple distance check for merging nodes
        for idx, coord in enumerate(node_coords):
            if np.linalg.norm(coord - p) < tol:
                return idx
        
        # If not found, create new node
        new_id = len(node_coords)
        node_coords.append(p)
        nodes_list.append({"id": new_id, "x": float(p[0]), "y": float(p[1])})
        return new_id

    for i, seg in enumerate(segs):
        n1 = get_node_id(seg.p0)
        n2 = get_node_id(seg.p1)
        
        # It's possible (though unlikely unless length < tol) that a segment collapses
        if n1 != n2:
            elements.append({"id": i, "n1": n1, "n2": n2})
            properties.append({
                "element_id": i,
                "thickness": float(seg.thick),
                "level": int(seg.level)
            })

    return {
        "nodes": nodes_list,
        "elements": elements,
        "properties": properties
    }


def export_to_json(graph: Dict[str, Any], filepath: str):
    """
    Export the structural graph to a JSON file.

    Parameters
    ----------
    graph : dict
        The output from `build_structural_graph()`.
    filepath : str
        The path to save the JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(graph, f, indent=2)

