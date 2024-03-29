# Cliffs

Cliffs is a tool for finding Clifford decompositions of quantum states. It uses the random walk method detailed in the paper ["Trading Classical and Quantum Computational Resources"](https://doi.org/10.1103/PhysRevX.6.021043) by Bravyi, Smith, and Smolin. Note that this program will attempt to search over the real Cliffords when the input state is real, since it can be proven that the optimal decomposition of a real state has all real Cliffords. Since this is a smaller search space, it is recommended that you should transform your state to a real one first if possible. E.g the decomposition of $\ket{T}^{\otimes 6}$ into seven terms requires 100x more annealing steps to find than the same for $\ket{H}^{\otimes 6}$ (which is the same state up to local Cliffords, but is real-valued).

### Usage

See the `--help` flag for an overview of all the options. The easiest possible usage is like
```
cliffs <CHI> <TARGET>
```
where `CHI` is the number of terms in the decomposition and `TARGET` is the `.npy` file containing the target state. If you have a large state and cliffs isn't finding any decompositions, then you can try increasing the number of annealing steps by using the `-s` option. If you want to find decompositions of a state tensored with itself, use the `-t` option - note that cliffs will flatten your input array into a vector *before* it takes the tensor product! This may result in an unexpected ordering of qubits.

### File Formats

Cliffs takes its input as a matrix of complex values in the NumPy ".npy" file format. These can be produced in Python with the [`numpy.save`](https://numpy.org/doc/stable/reference/generated/numpy.save.html) function. For each term of any decompositions found, a ".npy" file is produced containing the statevector, as well as a ".qasm" containing its corresponding Clifford state preparation circuit and a ".tikz" containing its affine-plus-phases normal form ZX-diagram in a TikZit/PyZX compatible format. These can be loaded into Python using the [`numpy.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html), [`pyzx.Circuit.from_qasm`](https://pyzx.readthedocs.io/en/latest/api.html#pyzx.circuit.Circuit.from_qasm), and [`pyzx.Graph.from_tikz`](https://pyzx.readthedocs.io/en/latest/api.html#pyzx.graph.base.BaseGraph.from_tikz) functions respectively.

### Caveats

Cliffs is currently under development, and some parts of the code are buggy. In particular, fast QR updates are currently not used, due to unsolved issues. Additionally, the `qasm` and `tikz` outputs may have issues, although this is much improved from earlier versions - it's now much more likely to be correct, but YMMV.

That being said, Cliffs has been used to generate some useful new results (decompositions of ZH zero-labelled H-box / the ZX triangle operator), as well as reproducing results such as the H/T magic state decompositions from Bravyi, Smith, and Smolin, and the cat-state decompositions from Qassim, Pashayan, and Gosset. All these are given in the `data` directory.

### Building & Installation

#### Dependencies
* A recent version of the Rust compiler.
* A copy of [OpenBLAS](https://www.openblas.net) installed on your system.

#### For MacOS (untested) / Linux
* Your package manager should have a copy of OpenBLAS. For instance on Ubuntu/Debian/etc. `libopenblas-dev` is the relevant package.
* Run `cargo install`.

#### For Windows (possible, but painful)
* I recommend using the `nightly-x86_64-pc-windows-gnu` Rust toolchain, not the MSVC version.
* Binary releases of OpenBLAS are provided [here](https://github.com/xianyi/OpenBLAS).
* Very tentatively run `cargo install`.
* If you can get it to build on MSVC, please let me know :)
* If all else fails, it will definitely build in WSL2.

Note that OpenBLAS is not linked statically, so the binary will need a copy of that installed for it to run. If you want a portable version, you can try changing `openblas-system` to `openblas-static` in `Cargo.toml`, but YMMV.

### Roadmap

* ~~Improve the GSLC generation~~
* Enable fast QR updating for improved performance
* Add an iterative deepening search and pruning method
* Include a dedicated Clifford testing mode 
