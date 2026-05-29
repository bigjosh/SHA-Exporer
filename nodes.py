"""Shared library for the SHA256 circuit tooling.

Format (see plan.md §7): plain text, one node per line, comma-separated.
Blank lines and lines starting with '#' are ignored. Everything else is strict:
malformed lines and duplicate node IDs raise and crash (favor simple code over
defensive checks).

Node types:
    C,<id>,<0|1>            constant
    N,<id>,<in0>,<in1>      NAND of two inputs
    O,<id>,<in>             output (copy of its single input)

A referenced-but-undefined ID is a *free input* with unknown value X (e.g. the
512 MESSAGE-* bits in the base graph).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


@dataclass(frozen=True)
class Node:
    type: str  # 'C' | 'N' | 'O'
    id: str
    # exactly one of the following is meaningful per type:
    value: int | None = None      # C: 0 or 1
    inputs: tuple[str, str] | None = None  # N: (in0, in1)
    input: str | None = None      # O: single input id


class DuplicateNodeError(KeyError):
    """Raised when a node ID is defined more than once."""


def _check_id(nid: str) -> str:
    if not _ID_RE.match(nid):
        raise ValueError(f"illegal node id: {nid!r}")
    return nid


def parse_line(line: str) -> Node:
    """Parse one non-blank, non-comment line into a Node. Strict."""
    parts = [p.strip() for p in line.strip().split(",")]
    kind = parts[0]
    if kind == "C":
        _, nid, val = parts  # len mismatch -> ValueError (intentional crash)
        v = int(val)
        if v not in (0, 1):
            raise ValueError(f"constant value must be 0/1: {line!r}")
        return Node("C", _check_id(nid), value=v)
    if kind == "N":
        _, nid, a, b = parts
        return Node("N", _check_id(nid), inputs=(_check_id(a), _check_id(b)))
    if kind == "O":
        _, nid, a = parts
        return Node("O", _check_id(nid), input=_check_id(a))
    raise ValueError(f"unknown node type {kind!r} in line: {line!r}")


class Graph:
    """A circuit: a mapping of defined node IDs to Nodes."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}

    # --- construction -----------------------------------------------------
    def add(self, node: Node) -> None:
        if node.id in self.nodes:
            raise DuplicateNodeError(node.id)
        self.nodes[node.id] = node

    # --- queries ----------------------------------------------------------
    def fanins(self, node: Node) -> tuple[str, ...]:
        if node.type == "N":
            return node.inputs  # type: ignore[return-value]
        if node.type == "O":
            return (node.input,)  # type: ignore[return-value]
        return ()

    def referenced_ids(self) -> set[str]:
        refs: set[str] = set()
        for n in self.nodes.values():
            refs.update(self.fanins(n))
        return refs

    def free_inputs(self) -> set[str]:
        """Referenced IDs that are not defined by any node (unknown inputs)."""
        return {r for r in self.referenced_ids() if r not in self.nodes}

    def outputs(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.type == "O"]

    def topo_order(self) -> list[str]:
        """Topological order of *defined* nodes (free inputs are implicit sources).

        Iterative DFS post-order. Raises on a cycle (the circuit must be acyclic).
        """
        WHITE, GREY, BLACK = 0, 1, 2
        color: dict[str, int] = {}
        order: list[str] = []
        for start in self.nodes:
            if color.get(start, WHITE) != WHITE:
                continue
            stack = [(start, False)]
            while stack:
                nid, processed = stack.pop()
                if processed:
                    color[nid] = BLACK
                    order.append(nid)
                    continue
                if color.get(nid, WHITE) == BLACK:
                    continue
                if color.get(nid, WHITE) == GREY:
                    raise ValueError(f"cycle detected at node {nid!r}")
                color[nid] = GREY
                stack.append((nid, True))
                for f in self.fanins(self.nodes[nid]):
                    if f in self.nodes and color.get(f, WHITE) != BLACK:
                        stack.append((f, False))
        return order

    # --- serialization ----------------------------------------------------
    def serialize(self) -> str:
        out: list[str] = []
        for nid, n in self.nodes.items():
            if n.type == "C":
                out.append(f"C,{nid},{n.value}")
            elif n.type == "N":
                out.append(f"N,{nid},{n.inputs[0]},{n.inputs[1]}")
            else:
                out.append(f"O,{nid},{n.input}")
        return "\n".join(out) + "\n"

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(self.serialize())


def parse_lines(lines) -> Graph:
    g = Graph()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        g.add(parse_line(line))
    return g


def parse(path: str) -> Graph:
    with open(path, encoding="utf-8") as fh:
        return parse_lines(fh)


def _nand(a: int | None, b: int | None) -> int | None:
    """3-valued NAND: any 0 input -> 1; both 1 -> 0; else X (None)."""
    if a == 0 or b == 0:
        return 1
    if a == 1 and b == 1:
        return 0
    return None


def evaluate(graph: Graph, pi_values: dict[str, int | None] | None = None
             ) -> dict[str, int | None]:
    """Evaluate every defined node (3-valued). Free inputs come from pi_values
    (default X/None). Returns {node_id: 0|1|None} for all defined nodes and for
    any free inputs that were supplied a value."""
    pi_values = pi_values or {}
    vals: dict[str, int | None] = {}

    def val(nid: str) -> int | None:
        if nid in graph.nodes:
            return vals[nid]
        return pi_values.get(nid)  # free input

    for nid in graph.topo_order():
        n = graph.nodes[nid]
        if n.type == "C":
            vals[nid] = n.value
        elif n.type == "N":
            vals[nid] = _nand(val(n.inputs[0]), val(n.inputs[1]))
        else:  # O
            vals[nid] = val(n.input)
    return vals


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Self-test: round-trip, free inputs, duplicate detection, evaluation.
    sample = """
    # AND(x, y) built from NAND, exposed as output OUT
    N,t,x,y
    N,andxy,t,t
    O,OUT,andxy
    """.strip().splitlines()

    g = parse_lines(sample)
    assert set(g.nodes) == {"t", "andxy", "OUT"}, g.nodes
    assert g.free_inputs() == {"x", "y"}, g.free_inputs()
    assert [o.id for o in g.outputs()] == ["OUT"]
    # round-trip
    g2 = parse_lines(g.serialize().splitlines())
    assert g2.serialize() == g.serialize()
    # topo order respects dependencies
    order = g.topo_order()
    assert order.index("t") < order.index("andxy") < order.index("OUT")

    # AND truth table via 3-valued evaluate
    for x in (0, 1):
        for y in (0, 1):
            out = evaluate(g, {"x": x, "y": y})["OUT"]
            assert out == (x & y), (x, y, out)
    # X propagation: AND(X, 0) = 0, AND(X, 1) = X
    assert evaluate(g, {"x": None, "y": 0})["OUT"] == 0
    assert evaluate(g, {"x": None, "y": 1})["OUT"] is None

    # duplicate definition crashes
    try:
        parse_lines(["C,dup,0", "C,dup,1"])
        raise SystemExit("FAIL: duplicate id not detected")
    except DuplicateNodeError:
        pass

    # cycle detection
    try:
        parse_lines(["N,a,b,b", "N,b,a,a"]).topo_order()
        raise SystemExit("FAIL: cycle not detected")
    except ValueError:
        pass

    print("nodes.py self-test OK")
