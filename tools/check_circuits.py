#!/usr/bin/env python3

import pyzx as zx
import numpy as np
import glob
import sys

def reconstruct_fidelity(prefix):
    states = np.load(prefix + ".cliffs.npy")
    cols = []
    for i in range(states.shape[1]):
        circ = zx.Circuit.from_qasm_file(prefix + f"_{i}.cliffs.qasm")
        g = circ.to_graph()
        g.apply_state("0" * circ.qubits)
        cols.append(g.to_matrix())
    circs = np.hstack(cols)
    target = np.sum(states, axis=1)[:, None]
    q, _ = np.linalg.qr(circs)
    q = np.matrix(q)
    f = np.linalg.norm(q@(q.H@target))
    return f

if len(sys.argv) > 2 or len(sys.argv) == 2 and sys.argv[1] in ('--help', '-h'):
    print("Usage: check_circuits.py FOLDER\nChecks that all `.cliffs.npy` files in the given folder have valid circuits for each term.\nDefaults to the current working directory, and recurses to any subfolders.")
else:
    folder = sys.argv[1] if len(sys.argv) == 2 else '.'
    passed = 0
    failed = 0
    for file in glob.glob(f"{folder}/**/*.cliffs.npy", recursive=True):
        prefix = file.removesuffix(".cliffs.npy")
        f = reconstruct_fidelity(prefix)
        if np.isclose(f, 1):
            print(f"PASS: {prefix} => fidelity = {f:0.5f}")
            passed += 1
        else:
            print(f"FAIL: {prefix} => fidelity = {f:0.5f}")
            failed += 1
    print(f"{passed} passed, {failed} failed")