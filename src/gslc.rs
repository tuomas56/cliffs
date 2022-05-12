use std::collections::HashMap;
use serde_json::json;

/// A structure that represents an (unnormalized) Clifford state
/// in the form of a graph state with local Cliffords.
/// This can be updated to add CZ, H and S gates in O(n^2) time each.
/// The algorithms are based roughly on the rewrite rules presented in:
/// "Kissinger A, van de Wetering J. Picturing Quantum Software. 2022. (Unpublished)""
/// No attempt is made to calculate scalar factors! This is _only_ maintaining
/// the structure of the graph.
#[derive(Clone, Debug)]
pub struct GSLC {
    n: usize,
    adjacency: Vec<Vec<bool>>,
    phases: Vec<usize>,
    hadamards: Vec<bool>
}

impl GSLC {
    pub fn new(n: usize) -> GSLC {
        GSLC {
            n, 
            adjacency: vec![vec![false; n]; n],
            phases: vec![0; n],
            hadamards: vec![false; n]
        }
    }

    fn get_phase(&self, i: usize) -> usize {
        self.phases[i]
    }

    fn add_phase_to_qubit(&mut self, i: usize, k: usize) {
        self.phases[i] += k;
        self.phases[i] %= 4;
    }

    // Add phase k to all the neighbors of i
    fn add_phase_to_neighbours(&mut self, i: usize, k: usize) {
        for j in 0..self.n {
            if self.has_edge(i, j) && i != j {
                self.add_phase_to_qubit(j, k);
            }
        }
    }

    fn has_hadamard(&self, i: usize) -> bool {
        self.hadamards[i]
    }

    fn toggle_hadamard(&mut self, i: usize) {
        self.hadamards[i] = !self.hadamards[i]
    }

    fn has_edge(&self, i: usize, j: usize) -> bool {
        self.adjacency[i][j]
    }

    // Toggle a Hadamard edge between qubits.
    // Self-edges are never added, instead a pi-phase is added.
    fn toggle_edge(&mut self, i: usize, j: usize) {
        if i != j {
            self.adjacency[i][j] = !self.adjacency[i][j];
            self.adjacency[j][i] = !self.adjacency[j][i];
        } else {
            self.add_phase_to_qubit(i, 2);
        }
    }
    
    // Add an edge to j from all the neighbors of i
    fn add_edge_to_neighbours(&mut self, i: usize, j: usize) {
        for k in 0..self.n {
            if self.has_edge(i, k) {
                self.toggle_edge(k, j);
            }
        }
    }

    // Complement all the edges connecting nodes in the neighborhood of i
    fn local_complementation(&mut self, i: usize) {
        for j in 0..self.n {
            if self.has_edge(i, j) {
                for k in 0..j {
                    if self.has_edge(i, k) {
                        self.toggle_edge(j, k)
                    }
                }
            }
        }
    }

    // Complement all the edges connecting nodes in the
    // neighbourhood of i to ones in the neighbourhood of j
    fn connect_neighbours(&mut self, i: usize, j: usize) {
        for k in 0..self.n {
            if self.has_edge(i, k) {
                for l in 0..self.n {
                    if self.has_edge(j, l) {
                        self.toggle_edge(k, l);
                    }
                }
            }
        }
    }

    fn swap_qubits(&mut self, i: usize, j: usize) {
        self.adjacency.swap(i, j);
        for k in 0..self.n {
            self.adjacency[k].swap(i, j);
        }
    }

    /// Apply a Hadamard gate to qubit q
    pub fn apply_h(&mut self, q: usize) {
        // println!("toggling hadamard");
        self.toggle_hadamard(q);
    }

    /// Apply a Z[pi] gate to qubit q
    pub fn apply_z(&mut self, q: usize) {
        if self.has_hadamard(q) {
            // With a Hadamard, we can push through
            // and then do a pi-copy to add phases to the neighbours.
            self.add_phase_to_neighbours(q, 2);
            // If the phase on q was +-pi/2 then we need to flip it.
            if self.get_phase(q) & 2 == 1 {
                self.add_phase_to_qubit(q, 2);
            }
        } else {
            self.add_phase_to_qubit(q, 2);
        }
    }

    /// Apply a Z[pi/2] gate to qubit q
    pub fn apply_s(&mut self, q: usize) {
        if self.has_hadamard(q) {
            // With a Hadamard, we can push it through and then
            // a combination of phases and local complementation
            // will absorb it.
            let (kn, kq, hq) = match self.get_phase(q) {
                0 => (1, 0, true),
                1 => (3, 3, false),
                2 => (3, 0, true),
                3 => (1, 3, false),
                _ => unreachable!()
            };

            // println!("applying local complementation");
            self.local_complementation(q);
            self.add_phase_to_qubit(q, kq);
            self.add_phase_to_neighbours(q, kn);
            if !hq {
                self.toggle_hadamard(q);
            }
        } else {
            // With no Hadamard this just updates the phase.
            // println!("applying phase");
            self.add_phase_to_qubit(q, 1);
        }
    }

    /// Apply a CZ gate between qubits i and j
    pub fn apply_cz(&mut self, i: usize, j: usize) {
        // If the local clifford is of the form H + Z[(-1)^k pi/2]
        // then we can do a local complementation to get rid of the Hadamard.
        if self.has_hadamard(i) && self.get_phase(i) % 2 == 1 {
            // println!("simplifying hadamard + pi/2");
            self.local_complementation(i);
            self.add_phase_to_neighbours(i, 4 - self.get_phase(i));
            self.add_phase_to_qubit(i, 2);
            self.toggle_hadamard(i);
        }

        // Same on the other qubit.
        if self.has_hadamard(j) && self.get_phase(j) % 2 == 1 {
            // println!("simplifying hadamard + pi/2");
            self.local_complementation(j);
            self.add_phase_to_neighbours(j, 4 - self.get_phase(j));
            self.add_phase_to_qubit(j, 2);
            self.toggle_hadamard(j);
        }

        if !self.has_hadamard(i) && !self.has_hadamard(j) {
            // With no Hadamards, we can just add an edge.
            // println!("adding edge from CZ");
            self.toggle_edge(i, j);
        } else if !self.has_hadamard(i) {
            // If there is one Hadamard, we have to do one pivot.
            // println!("doing half-pivot");
            self.add_edge_to_neighbours(j, i);
            self.add_phase_to_qubit(i, self.get_phase(j));

            if self.has_edge(i, j) {
                self.add_phase_to_qubit(i, 2);
            }
        } else if !self.has_hadamard(j) {
            // println!("doing half-pivot");
            self.add_edge_to_neighbours(i, j);
            self.add_phase_to_qubit(j, self.get_phase(i));

            if self.has_edge(i, j) {
                self.add_phase_to_qubit(j, 2);
            }
        } else {
            // If there are two Hadamards, we need to do
            // two pivots, which results in a complete bipartite graph
            // between their neighbourhoods and possibly a swap
            // if they were already connected.
            // println!("doing full pivot");
            if self.has_edge(i, j) {
                // println!("pivoting with swap");
                self.swap_qubits(i, j);
                self.toggle_edge(i, j);
                self.toggle_hadamard(i);
                self.toggle_hadamard(j);
            }
            
            self.add_phase_to_neighbours(i, self.get_phase(j));
            self.add_phase_to_neighbours(j, self.get_phase(i));
            self.connect_neighbours(i, j);
        }
    }

    /// Convert this GSLC to the Quantomatic/PyZX .qgraph format.
    pub fn to_qgraph(&self) -> String {
        // Note we assemble the qubits top to bottom, but the y coordinate
        // is increasing bottom to top, so all the y coordinates are flipped.
        // This is important since PyZX orders the qubits by location not name.
        let mut boundaries = HashMap::new();
        for i in 0..self.n {
            boundaries.insert(format!("b{}", i), json!({
                "annotation": {
                    "boundary": true,
                    "coord": [1, self.n-i],
                    "input": false,
                    "output": true
                }
            }));
        }

        let mut nodes = HashMap::new();
        for i in 0..self.n {
            let phase = format!("{}\\pi/2", self.get_phase(i));
            nodes.insert(format!("v{}", i), json!({
                "annotation": {
                    "coord": [0, self.n-i]
                },
                "data": {
                    "type": "Z",
                    "value": phase
                }
            }));
        }

        let mut node = self.n;
        let mut edge = 0;
        let mut edges = HashMap::new();

        for i in 0..self.n {
            if !self.has_hadamard(i) {
                let start = format!("b{}", i);
                let end = format!("v{}", i);
                edges.insert(format!("e{}", edge), json!({
                    "src": start,
                    "tgt": end
                }));
                edge += 1;
            } else {
                nodes.insert(format!("v{}", node), json!({
                    "annotation": {
                        "coord": [-1, self.n-i]
                    },
                    "data": {
                        "type": "hadamard",
                        "is_edge": "true"
                    }
                }));
                let start = format!("v{}", i);
                let middle = format!("v{}", node);
                let end = format!("b{}", i);
                edges.insert(format!("e{}", edge), json!({
                    "src": start,
                    "tgt": middle
                }));
                edges.insert(format!("e{}", edge+1), json!({
                    "tgt": middle,
                    "src": end
                }));
                node += 1;
                edge += 2;
            }
        }

        for i in 0..self.n {
            for j in 0..i {
                if self.has_edge(i, j) {
                    nodes.insert(format!("v{}", node), json!({
                        "annotation": {
                            "coord": [0, self.n-i]
                        },
                        "data": {
                            "type": "hadamard",
                            "is_edge": "true"
                        }
                    }));
                    let start = format!("v{}", i);
                    let middle = format!("v{}", node);
                    let end = format!("v{}", j);
                    edges.insert(format!("e{}", edge), json!({
                        "src": start,
                        "tgt": middle
                    }));
                    edges.insert(format!("e{}", edge+1), json!({
                        "src": middle,
                        "tgt": end
                    }));
                    node += 1;
                    edge += 2;
                }
            }
        }

        // We do not include a scalar since we didn't compute the correct one anyways :)
        json!({
            "wire_vertices": boundaries,
            "node_vertices": nodes,
            "undir_edges": edges,
            "scalar": "{\"power2\": 0, \"phase\": \"0\"}"
        }).to_string()
    }
}
