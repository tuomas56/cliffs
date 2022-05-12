import numpy as np
import pyzx as zx

n = int(input("Number of qubits: "))
circ = zx.Circuit(n)
s = input(">> ")
while s != '':
    name, *qubits = s.strip().split()
    circ.add_gate(name, *map(int, qubits))
    s = input(">> ")
circ = circ.to_graph()
circ.apply_state("0"*n)
np.save(input("Output filename: "), circ.to_matrix())
