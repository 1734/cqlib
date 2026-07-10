# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Reusable compiler transforms.

Transforms expose focused compiler stages for callers that need more control
than the end-to-end :func:`cqlib.compile.compile` workflow. Each transform
returns a result object instead of mutating the input circuit.

Example::

    from cqlib.circuit import Circuit
    from cqlib.compile.transform import canonicalize_circuit

    circuit = Circuit(1)
    circuit.h(0)

    result = canonicalize_circuit(circuit)
    assert result.circuit is not circuit
"""

from . import decompose as decompose
from . import layout as layout
from . import routing as routing
from . import resynthesis as resynthesis
from .canonicalize import CanonicalizeConfig as CanonicalizeConfig
from .canonicalize import CanonicalizeResult as CanonicalizeResult
from .canonicalize import Canonicalizer as Canonicalizer
from .canonicalize import canonicalize_circuit as canonicalize_circuit
from .layout import LayoutDiagnostics as LayoutDiagnostics
from .layout import LayoutObjective as LayoutObjective
from .layout import LayoutResult as LayoutResult
from .layout import LayoutScore as LayoutScore
from .layout import Vf2EdgeRequirement as Vf2EdgeRequirement
from .layout import Vf2LayoutConfig as Vf2LayoutConfig
from .layout import greedy_layout as greedy_layout
from .layout import sabre_layout as sabre_layout
from .layout import trivial_layout as trivial_layout
from .layout import vf2_perfect_layout as vf2_perfect_layout
from .routing import RoutedCircuit as RoutedCircuit
from .routing import SabreRouteResult as SabreRouteResult
from .routing import route_sabre as route_sabre
from .routing import route_with_layout as route_with_layout
from .routing_basis import LowerToRoutingBasis as LowerToRoutingBasis
from .routing_basis import lower_to_routing_basis as lower_to_routing_basis
from .rewrite import KnowledgeRewriteResult as KnowledgeRewriteResult
from .rewrite import KnowledgeRewriteStats as KnowledgeRewriteStats
from .rewrite import KnowledgeRewriter as KnowledgeRewriter
from .rewrite import RewriteConfig as RewriteConfig
from .rewrite import RewriteMode as RewriteMode
from .rewrite import rewrite_circuit as rewrite_circuit
from .result import TransformResult as TransformResult

__all__ = [
    "decompose",
    "layout",
    "routing",
    "resynthesis",
    "TransformResult",
    "CanonicalizeConfig",
    "Canonicalizer",
    "CanonicalizeResult",
    "canonicalize_circuit",
    "LayoutObjective",
    "LayoutScore",
    "LayoutDiagnostics",
    "LayoutResult",
    "Vf2EdgeRequirement",
    "Vf2LayoutConfig",
    "trivial_layout",
    "greedy_layout",
    "vf2_perfect_layout",
    "sabre_layout",
    "RoutedCircuit",
    "SabreRouteResult",
    "route_with_layout",
    "route_sabre",
    "LowerToRoutingBasis",
    "lower_to_routing_basis",
    "RewriteMode",
    "RewriteConfig",
    "KnowledgeRewriter",
    "KnowledgeRewriteStats",
    "KnowledgeRewriteResult",
    "rewrite_circuit",
]
