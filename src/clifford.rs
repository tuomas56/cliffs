use ndarray_linalg::{c64, Scalar, Norm};
use ndarray as nd;
use zx::{vec_graph::GraphLike, scalar::Zero};

/// A data structure for solving systems of linear equations mod four.
struct ModSolver {
    rows: usize,
    cols: usize,
    m: Vec<u8>,
    r: Vec<u8>,
    v: Vec<u8>
}

impl ModSolver {
    fn new(rows: usize, cols: usize, m: Vec<u8>, v: Vec<u8>) -> Self {
        // r is a matrix storing the column operations
        let mut r = vec![0; cols * cols];
        for i in 0..cols {
            r[i * cols + i] = 1;
        }
        Self {
            rows, cols, m, v, r
        }
    }

    // Apply a row swap operation to m and the corresponding action to v
    fn row_swap(&mut self, i: usize, j: usize) {
        for k in 0..self.cols {
            self.m.swap(i * self.cols + k, j * self.cols + k);
        }
        self.v.swap(i, j);
    }

    // Apply a column swap operation and the corresponding action to r
    fn col_swap(&mut self, i: usize, j: usize) {
        for k in 0..self.rows {
            self.m.swap(k * self.cols + i, k * self.cols + j);
        }

        for k in 0..self.cols {
            self.r.swap(k * self.cols + i, k * self.cols + j);
        }
    }

    // Apply a 2x2 update to two rows of the matrix and v, we must have
    // ac - bd = 1 to maintain unimodularity of r and correctness of v.
    fn row_op(&mut self, i: usize, j: usize, a: u8, b: u8, c: u8, d: u8) {
        for k in 0..self.cols {
            let x = (a * self.m[i * self.cols + k] + b * self.m[j * self.cols + k]) % 4;
            let y = (c * self.m[i * self.cols + k] + d * self.m[j * self.cols + k]) % 4;
            self.m[i * self.cols + k] = x;
            self.m[j * self.cols + k] = y;
        }

        let x = (a * self.v[i] + b * self.v[j]) % 4;
        let y = (c * self.v[i] + d * self.v[j]) % 4;
        self.v[i] = x;
        self.v[j] = y;
    }

    // The same thing but for columns, applying the update to m and v.
    fn col_op(&mut self, i: usize, j: usize, a: u8, b: u8, c: u8, d: u8) {
        for k in 0..self.rows {
            let x = (a * self.m[k * self.cols + i] + b * self.m[k * self.cols + j]) % 4;
            let y = (c * self.m[k * self.cols + i] + d * self.m[k * self.cols + j]) % 4;
            self.m[k * self.cols + i] = x;
            self.m[k * self.cols + j] = y;
        }

        for k in 0..self.cols {
            let x = (a * self.r[k * self.cols + i] + b * self.r[k * self.cols + j]) % 4;
            let y = (c * self.r[k * self.cols + i] + d * self.r[k * self.cols + j]) % 4;
            self.r[k * self.cols + i] = x;
            self.r[k * self.cols + j] = y;
        }
    }

    // Get the bezout coefficients and gcd for two integers
    // since we are working mod four always, this is hardcoded.
    fn egcd(a: u8, b: u8) -> (u8, u8, u8) {
        match (a, b) {
            (0, x) => (x, 0, 1),
            (x, 0) => (x, 1, 0),
            (1, _) => (1, 1, 0),
            (_, 1) => (1, 0, 1),
            (2, 2) => (2, 1, 0),
            (2, 3) => (1, 1, 1),
            (3, 2) => (1, 1, 1),
            (3, 3) => (3, 1, 0),
            _ => unreachable!()
        }
    }

    // Reduce the matrix to smith normal form
    fn snf(&mut self) {
        // Position along the diagonal
        let mut t = 0;
        // Place to start looking for next pivot column
        let mut scol = 0;

        while t < self.rows {
            // Find a pivot column (i.e one that is non-zero)
            let Some(jt) = (scol..self.cols)
                .find(|&i| (0..self.rows)
                    .any(|j| self.m[j * self.cols + i] != 0)) 
                else { break };

            // Find the smallest non-zero value in the column and
            // move it to the diagonal to become the pivot.
            let k = (0..self.rows)
                .filter(|&i| self.m[i * self.cols + jt] != 0)
                .min_by_key(|&i| self.m[i * self.cols + jt])
                .unwrap();
            if t != k {
                self.row_swap(t, k);
            }

            // While the row and column are not cleared off-diagonal:
            while (jt+1..self.cols).any(|i| self.m[t * self.cols + i] != 0) 
                || (t+1..self.rows).any(|i| self.m[i * self.cols + jt] != 0) {
                // While there is an element of the column that isn't a multiple of this one:
                let mut pivot = self.m[t * self.cols + jt];
                while let Some(k) = (0..self.rows).find(|&k| self.m[k * self.cols + jt] % pivot != 0) {
                    let (beta, sigma, tau) = Self::egcd(pivot, self.m[k * self.cols + jt]);
                    // Do row operations to replace the pivot with the gcd of the pivot and the other element
                    // which we can do with only integer row operations by Bezout's theorem.
                    self.row_op(t, k, sigma % 4, tau % 4, 4 - self.m[k * self.cols + jt] / beta, pivot / beta);
                    pivot = self.m[t * self.cols + jt];
                }
                
                // Now this pivot is a divisor of every element in the column,
                // so we can zero it out by subtracting only integer multiple of the pivot row
                for k in t+1..self.rows {
                    if self.m[k * self.cols + jt] != 0 {
                        self.row_op(k, t, 1, 4 - self.m[k * self.cols + jt] / pivot, 0, 1);
                    }
                }

                // Do the same thing for rows
                while let Some(k) = (0..self.cols).find(|&k| self.m[t * self.cols + k] % pivot != 0) {
                    let (beta, sigma, tau) = Self::egcd(pivot, self.m[t * self.cols + k]);
                    self.col_op(jt, k, sigma % 4, tau % 4, 4 - self.m[t * self.cols + k] / beta, pivot / beta);
                    pivot = self.m[t * self.cols + jt];
                }
                
                for k in t+1..self.cols {
                    if self.m[t * self.cols + k] != 0 {
                        self.col_op(k, jt, 1, 4 - self.m[t * self.cols + k] / pivot, 0, 1);
                    }
                }
            }

            t += 1;
            scol = jt + 1;
        }

        // Now the matrix is empty except for the pivots,
        // so move all empty columns to the end to make it diagonal
        while let Some((i, j)) = (0..self.rows)
            .filter_map(|i| (0..self.cols)
                .filter(|&j| j != i)
                .find(|&j| self.m[i * self.cols + j] != 0)
                .map(|j| (i, j)))
            .next() {
            self.col_swap(i, j);
        }

        // Now use row and column operations to fix the divisibility
        // while maintaining diagonality - make it so each element on 
        // the diagonal divides the next.
        while let Some(i) = (1..self.rows.min(self.cols))
            .filter(|&i| self.m[(i - 1) * self.cols + i - 1] != 0)
            .find(|&i| self.m[i * self.cols + i] % self.m[(i - 1) * self.cols + i - 1] != 0) {
            self.col_op(i - 1, i, 1, 1, 0, 1);
            let (beta, sigma, tau) = Self::egcd(self.m[(i - 1) * self.cols + i - 1], self.m[i * self.cols + i]);
            self.row_op(i - 1, i, sigma, tau, 4 - self.m[i * self.cols + i] / beta, self.m[(i - 1) * self.cols + i - 1] / beta);
            self.col_op(i, i - 1, 1, 4 - self.m[(i - 1) * self.cols + i] / self.m[(i - 1) * self.cols + (i - 1)], 0, 1);
        }
    }

    /// Solve this system and return one answer.
    /// If you need more solutions, lattice methods are required.
    fn solve(&mut self) -> Vec<u8> {
        // First put the matrix in smith normal form
        self.snf();
        // Now self.v is updated to L*v where D = LAR is the SNF of the matrix
        // so if x is such that Ax = v, we have LAx = Lv (L is unimodular so invertible)
        // and hence if Dy = Lv, then Ry = R(R^-1 A^+ L^-1)Lv = A^+v.
        
        // Now, solve for y:
        for i in 0..self.rows.min(self.cols) {
            if self.m[i * self.cols + i] != 0 {
                // We produce a modular inverse using the extended euclidean algorithm.
                // If no inverse exists, no solutions exist, due to properties of SNF.
                let (g, _, _) = Self::egcd(self.v[i], self.m[i * self.cols + i]);
                self.v[i] /= g;
                match self.m[i * self.cols + i] / g {
                    1 => (),
                    3 => self.v[i] = 4 - self.v[i],
                    0 | 2 => panic!("no solutions!"),
                    _ => unreachable!()
                }
            } else if self.v[i] != 0 {
                panic!("no solutions!");
            }
        }
        
        // Given y, we just do matrix multiplication to find x = Ry
        let mut x = vec![0; self.cols];
        for i in 0..self.cols {
            x[i] = ((0..self.cols.min(self.rows))
                .map(|j| (self.r[i * self.cols + j] * self.v[j]) as usize)
                .sum::<usize>() % 4) as u8;
        }

        x
    }
}

// Find a basis for the nullspace of a matrix over GF(2)
fn find_nullspace(m: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    // First, put the matrix into RREF
    let mut m = zx::linalg::Mat2::new(m);
    m.gauss(true);

    // Identify the pivot columns, these are bound variables
    let mut pivots = Vec::new();
    let mut free = Vec::new();
    let mut row = 0;
    let mut col = 0;
    while row < m.num_rows() && col < m.num_cols() {
        if m[(row, col)] != 0 {
            pivots.push((row, col));
            row += 1;
            col += 1;
        } else {
            free.push(col);
            col += 1;
        }
    }    

    // The nullspace is spanned by the free variables,
    // so for each vector in a basis for the free variables
    // (we take the standard basis), solve for the bound variables
    // to obtain a vector in the nullspace.
    let mut rows = Vec::new();
    for f in free {
        let mut row = vec![0; m.num_cols()];
        row[f] = 1;
        for &(r, c) in &pivots {
            row[c] = m[(r, f)];
        }
        rows.push(row);
    }
    
    // These vectors are linearly independent by construction
    // so they form a basis for the nullspace
    rows
}

// Find the affine part of the AP normal form for a state phi:
//   phi = sum(x; Ax = b) { i^f(x) * x }
// eps is the precision for determining which elements are non-zero
fn find_affine(phi: &nd::Array1<c64>, eps: f64) -> (Vec<Vec<u8>>, Vec<u8>) {
    let n = phi.shape()[0].ilog2() as usize;
    // Make a matrix of all the binary expansions of non-zero indices in phi,
    // these vectors satisfy Ax = b by construction
    let mut bms = Vec::new();
    for i in 0..phi.shape()[0] {
        if phi[i].abs() > eps {
            let mut row = vec![0; n];
            for j in 0..n {
                row[j] = ((i & (1 << (n - 1 - j))) != 0) as u8;
            }
            bms.push(row);
        }
    }

    // Find one vector x' in bms and subtract it from all of them
    // Since Ax' = b, we have A(x - x') = b - b = 0, so the row space
    // of this matrix is the nullspace of A.
    let sol = bms[0].clone();
    for i in 0..bms.len() {
        for j in 0..n {
            bms[i][j] ^= sol[j];
        }
    }

    // The nullspace of the nullspace is the row space, so
    // find a basis for the nullspace of bms to find a basis for 
    // the row space of A
    let a = find_nullspace(bms);
    // Assume that A is given by this basis (which we can do WLOG)
    // then b = Ax' by construction, so we just substitute in x'.
    let mut b = vec![0; a.len()];
    for i in 0..b.len() {
        b[i] = (0..n)
            .map(|j| a[i][j] & sol[j])
            .fold(0, |x, y| x ^ y);
    }

    (a, b)
}

// Find the phases part of the AP normal form for a state phi:
//   phi = sum(x; Ax = b) { i^f(x) * x }
// eps is the precision for determining the phases of elements.
fn find_phases(phi: &nd::Array1<c64>, eps: f64) -> (Vec<u8>, Vec<Vec<u8>>, u8) {
    let n = phi.shape()[0].ilog2() as usize;
    // Construct a matrix of binary expansions of non-zero indices
    let mut bms = Vec::new();
    // Save for each non-zero index the corresponding phase
    let mut v = Vec::new();
    // Global phase to divide out
    let mut gphase = c64::zero();
    for i in 0..phi.shape()[0] {
        if phi[i].abs() > eps {
            // Global phase is first non-zero element
            if gphase.is_zero() {
                gphase = phi[i];
            }

            // Construct binary expansion
            let mut row = vec![0; n];
            for j in 0..n {
                row[j] = ((i & (1 << (n - 1 - j))) != 0) as u8;
            }
            bms.push(row);

            // Determine the phase of the element, modulo gphase
            if (phi[i] / gphase - c64::new(1.0, 0.0)).abs() < eps {
                v.push(0);
            } else if (phi[i] / gphase - c64::new(0.0, 1.0)).abs() < eps {
                v.push(1);
            } else if (phi[i] / gphase - c64::new(-1.0, 0.0)).abs() < eps {
                v.push(2);
            } else if (phi[i] / gphase - c64::new(0.0, -1.0)).abs() < eps {
                v.push(3);
            } else {
                // If its not 1, -1, i or -i then this is not clifford...
                panic!("uh: {}", phi[i]);
            }
        }
    }

    // Note that f(x) has the form f(x) = Lx + x^TQx + c, so 
    // construct a multivariate Vandermonde matrix M to solve for L, Q, and c
    let mut m = Vec::new();
    for i in 0..bms.len() {
        // First n columns are linear terms
        for j in 0..n {
            m.push(bms[i][j]);
        }

        // Then O(n^2) columns for the quadratic terms
        for j in 0..n {
            for k in 0..j {
                m.push(2 * bms[i][j] * bms[i][k]);
            }
        }

        // Then one column at the end for linear terms
        m.push(1);
    }

    // Then solve Mx = v where v is the phases from before.
    // Note we solve mod four since we only care about congruence mod 4,
    // as i^4 = 1, and we are working with log_i(phi[k]).
    let mut system = ModSolver::new(bms.len(), n + n * (n - 1) / 2 + 1, m, v);
    let x = system.solve();

    // Recover L, Q, and c from the vector x:
    let mut idx = 0;
    let mut l = vec![0; n];
    let mut q = vec![vec![0; n]; n];
    for i in 0..n {
        l[i] = x[idx];
        idx += 1;
    }
    for j in 0..n {
        for k in 0..j {
            q[j][k] = x[idx] % 2;
            q[k][j] = x[idx] % 2;
            idx += 1;
        }
    }
    let c = x[idx];

    (l, q, c)
}

// Find the AP normal form for a Clifford state phi:
//   phi = sum(x; Ax = b) { i^f(x) * x }
// eps is the precision for determining the phases and amplitude of elements.
// f(x) is given as f(x) = L^Tx + x^TQx + c for vector L, symmetric matrix Q.
fn find_apnf(phi: &nd::ArrayView1<c64>, eps: f64) -> (Vec<Vec<u8>>, Vec<u8>, Vec<u8>, Vec<Vec<u8>>, u8) {
    // First normalize the state so it has the expected amplitudes.
    let mut phi = phi.to_owned();
    let s1 = phi.iter()
        .filter(|x| x.abs() > eps)
        .count();
    let s1 = (s1 as f64).sqrt();
    let s = s1 / phi.norm();
    phi *= c64::from(s);
    
    let (a, b) = find_affine(&phi, eps);
    let (l, q, c) = find_phases(&phi, eps);

    (a, b, l, q, c)
}

// Convert an AP normal form description of a Clifford state into a ZX-diagram
fn apnf_to_zx(a: Vec<Vec<u8>>, b: Vec<u8>, l: Vec<u8>, q: Vec<Vec<u8>>, c: u8) -> zx::vec_graph::Graph {
    let mut g = zx::vec_graph::Graph::new();
    let verts = (0..l.len())
        .map(|i| {
            // Every qubit gets a Z-spider and boundary vertex,
            // the phase of the Z-spider is given by L
            let z = g.add_vertex_with_data(zx::graph::VData {
                ty: zx::graph::VType::Z,
                phase: (l[i] as isize, 2).into(),
                qubit: i as i32,
                row: 2
            });

            let b = g.add_vertex_with_data(zx::graph::VData {
                ty: zx::graph::VType::B,
                phase: 0.into(),
                qubit: i as i32,
                row: 3
            });

            g.add_edge_smart(z, b, zx::graph::EType::N);
            g.outputs_mut().push(b);

            z
        })
        .collect::<Vec<_>>();

    // Add H-edges between Z-spiders according to Q
    for i in 0..l.len() {
        for j in 0..i {
            if q[i][j] != 0 {
                g.add_edge_smart(verts[i], verts[j], zx::graph::EType::H);
            }
        }
    }
    
    // c is the global phase so add it to the scalar
    g.scalar_mut().mul_phase((c as isize, 2).into());

    // For the affine part, construct an X-spider for each row of A
    for i in 0..b.len() {
        let x = g.add_vertex_with_data(zx::graph::VData {
            ty: zx::graph::VType::X,
            phase: (b[i] as isize, 1).into(),
            qubit: i as i32,
            row: 1
        });

        for j in 0..verts.len() {
            if a[i][j] != 0 {
                // The corresponding row of A determines what Z-spiders
                // the X-spider is connected to.
                g.add_edge_smart(x, verts[j], zx::graph::EType::N);
            }
        }
    }

    g
}

// After simplification, a ZX-diagram may not be in strictly GSLC form
// (as it may be simpler), so we will need to unfuse identity Z-spiders 
fn unfuse_identity(g: &mut zx::vec_graph::Graph, u: usize, v: usize) {
    let z = g.add_vertex_with_data(zx::graph::VData {
        ty: zx::graph::VType::Z,
        phase: 0.into(),
        qubit: g.qubit(v),
        row: g.row(v)
    });

    let (a, b) = match g.edge_type(u, v) {
        zx::graph::EType::H => (zx::graph::EType::H, zx::graph::EType::N),
        zx::graph::EType::N => (zx::graph::EType::H, zx::graph::EType::H)
    };
    g.add_edge_with_type(u, z, a);
    g.add_edge_with_type(z, v, b);
    g.remove_edge(u, v);
}

// Make a graph-like ZX-diagram a formal GSLC by unfusing spiders wherever necessary
fn make_gslc(g: &mut zx::vec_graph::Graph) {
    // We may have an output connected directly to another output,
    // so generate a circuit for the Bell state in this case.
    for i in 0..g.outputs().len() {
        let v = g.outputs()[i];
        let u = g.neighbor_at(v, 0);
        if g.vertex_type(u) == zx::graph::VType::B {
            unfuse_identity(g, u, v);
        }
    }

    // We may have several outputs connected to the same spider
    // (e.g a GHZ state), so unfuse these to give every output its own Z-spider
    'outer: loop {
        for i in 0..g.outputs().len() {
            let v = g.outputs()[i];
            let c = g.neighbor_at(v, 0);
            for j in 0..i {
                let u = g.outputs()[j];
                if i != j && g.neighbor_at(u, 0) == c {
                    // If we find two outputs connected to the same spider
                    // unfuse, taking care that the edge connecting the spiders
                    // is a Hadamard edge to maintain GSLC form.
                    unfuse_identity(g, c, u);
                    // Now try again
                    continue 'outer;
                }
            }
        }
        // If we didn't find anything we're done
        break;
    }
}

// Give a circuit that prepares a GSLC
fn gslc_to_circuit(g: &zx::vec_graph::Graph) -> zx::circuit::Circuit {
    let mut circ = zx::circuit::Circuit::new(g.outputs().len());
    // Find the Z-spider associated with each output
    let verts = g.outputs()
        .iter()
        .map(|&out| (g.neighbor_at(out, 0), out))
        .filter(|&(i, _)| g.vertex_type(i) == zx::graph::VType::Z)
        .collect::<Vec<_>>();

    for (i, &(v, _)) in verts.iter().enumerate() {
        // Add Hadamards to transform |0> to |+>
        circ.add_gate("h", vec![i]);
        // Add the required phases for each spider
        if !g.phase(v).is_zero() {
            circ.add_gate_with_phase("rz", vec![i], g.phase(v));
        }
    }

    // Add CZ gates to connect the qubits as specified
    for (i, &(v, _)) in verts.iter().enumerate() {
        for (j, &(u, _)) in verts[..i].iter().enumerate() {
            if g.connected(u, v) {
                circ.add_gate("cz", vec![i, j]);
            }
        }
    }

    // The only local cliffords are hadamards, so add these wherever needed
    for (i, &(v, b)) in verts.iter().enumerate() {
        if let zx::graph::EType::H = g.edge_type(v, b) {
            circ.add_gate("h", vec![i]);
        }
    }

    circ
}

/// Generate TikZ code that represents the given ZX-diagram
pub fn graph_to_tikz(g: &zx::vec_graph::Graph) -> String {
    let mut verts = Vec::new();
    // We assemble in two layers, first vertices then edges.
    for v in g.vertices() {
        let p = g.phase(v);
        let ty = g.vertex_type(v);

        // Each vertex has a style,
        let style = match ty {
            zx::graph::VType::B => "none",
            zx::graph::VType::H => unimplemented!(),
            zx::graph::VType::Z => if !p.is_integer() {
                "Z phase dot"
            } else {
                "Z dot"
            },
            zx::graph::VType::X => if !p.is_integer() {
                "X phase dot"
            } else {
                "X dot"
            }
        }.to_string();

        // .. and a phase:
        let phase = if p.is_zero() {
            String::new()
        } else {
            let &n = p.numer();
            let &d = p.denom();
            match (n, d) {
                (1, 1) => format!("$\\pi$"),
                (1, _) => format!("$\\frac{{\\pi}}{{{}}}$", d),
                (_, 1) => format!("${}\\pi$", n),
                (_, _) => format!("$\\frac{{{}\\pi}}{{{}}}$", n, d)
            }
        };  

        // The x and y positions are given by the row and qubit:
        let x = g.row(v);
        let y = -g.qubit(v);   
        // Now every vertex becomes a TikZ node
        let s = format!("        \\node [style={}] ({}) at ({}, {}) {{{}}};", style, v, x, y, phase);
        verts.push(s);
    }

    let mut edges = Vec::new();
    for (v, w, ty) in g.edges() {
        // Every edge is a TikZ draw command
        let mut s = String::from("        \\draw ");
        // Add a style for H-edges:
        if ty == zx::graph::EType::H {
            s.push_str("[style=hadamard edge] ");
        }
        s.push_str(&format!("({}) to ({});", v, w));
        edges.push(s);
    }

    // Assemble it all in the two layers, as TikZit expects:
    let verts = verts.join("\n");
    let edges = edges.join("\n");
    format!("\\begin{{tikzpicture}}
    \\begin{{pgfonlayer}}{{nodelayer}}
{}
    \\end{{pgfonlayer}}
    \\begin{{pgfonlayer}}{{edgelayer}}
{}
    \\end{{pgfonlayer}}
\\end{{tikzpicture}}", verts, edges)
}

/// Given a Clifford state as a statevector, find the corresponding AP normal form
/// ZX-diagram and a Clifford circuit that prepares this state. eps is the precision
/// with which we identify values in the vector.
pub fn find_clifford(phi: &nd::ArrayView1<c64>, eps: f64) -> (zx::vec_graph::Graph, zx::circuit::Circuit) {
    let (a, b, l, q, c) = find_apnf(phi, eps);
    let g = apnf_to_zx(a, b, l, q, c);
    // APNF to GSLC is as easy as simplifying fully then some unfusion:
    let mut gslc = g.clone();
    zx::simplify::clifford_simp(&mut gslc);
    make_gslc(&mut gslc);
    let circ = gslc_to_circuit(&gslc);
    (g, circ)
}
