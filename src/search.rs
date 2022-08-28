use ndarray_linalg::{c64, Norm};
use ndarray as nd;
use rand::Rng;
use crate::{gslc, qr};

/// A geometric series specified by the start, end and number of steps.
#[derive(Clone, Copy)]
pub struct GeometricSequence {
    current: f64,
    factor: f64,
    end: f64
}

impl GeometricSequence {
    pub fn new(start: f64, end: f64, steps: usize) -> Self {
        GeometricSequence {
            current: start, end,
            factor: (end / start).powf(1.0 / (steps + 1) as f64)
        }
    }
}

impl Iterator for GeometricSequence {
    type Item = f64;
    
    fn next(&mut self) -> Option<f64> {
        if self.current <= self.end {
            let value = self.current;
            self.current *= self.factor;
            Some(value)
        } else {
            None
        }
    }
}

// A struct to generate random Pauli strings of size n.
struct PauliGenerator {
    n: usize,
    only_reals: bool,
    temp: nd::Array1<c64>,
    string: Vec<usize>
}

impl PauliGenerator {
    fn new(n: usize, only_reals: bool) -> Self {
        PauliGenerator { 
            n, only_reals,
            temp: nd::Array1::zeros(1 << n),
            string: Vec::with_capacity(n)
        }
    }

    // This premultiplies a random Pauli string to a vector
    // and applies the corresponding pi/2 Pauli exponential to a GSLC
    // Overall this should be O(n2^n + n^3) time.
    fn apply_random(&mut self, 
        rng: &mut impl rand::Rng, 
        vector: &mut nd::ArrayViewMut<c64, nd::Dim<[usize; 1]>>,
        gslc: &mut gslc::GSLC
    ) {
        // To do the multiplication P*v for some Pauli string v, we have
        // an iterative method. First, split P*v = P_1P_2P_3...P_n*v as
        // P*v = (P_1I..I)*(IP_2I..I)*(IIP_3I..I)*..*(I..IP_n)*v, and then
        // notice that for any matrix M, (I @ M)*(v_1; v_2) = (Mv_1; Mv_2)
        // and so (I..I @ M)*(v_1; v_2; ..; v_k) = (Mv_1; ..; Mv_k) where k
        // is 2^n/2^j if v is 2^n-dimensional and there are j I matrices.
        // Furthermore note that for any 2x2 matrix P, we have that
        // (P @ I...I)*(v_1; v_2) = (P_00*v_1 + P_01*v_2; P_10*v1 + P_11*v2)
        // and finally that I..IPI..I = I..I @ (P @ I..I) = I..IM, so we can 
        // do such a multiplication as 2^j block 2x2 multiplications.
        // If P is a Pauli operator then there are only two non-zero elements,
        // so each block multiplication reduces to a swap and/or scaling.

        loop {
            self.string.clear();
            for _ in 0..self.n {
                // Pick a random Pauli and record it
                let value = rng.gen_range(0usize..4);
                self.string.push(value);
            }

            // If there are an odd number of Ys and we are looking for reals try again
            if self.only_reals {
                let ys = self.string.iter().filter(|x| **x == 2).count();
                if ys % 2 == 0 {
                    break
                }
            } else {
                break
            }
        }

        for (i, &value) in self.string.iter().enumerate() {
            // If this is the identity there is nothing to do
            if value == 0 {
                continue;
            }

            // Otherwise, we can do a multiplication,
            // splitting the target vector into multiple blocks
            // and applying a 2x2 block multiplication to each of them
            let block_size = vector.len() / (1 << i);
            for j in (0..vector.len()).step_by(block_size) {
                let ia = vector.slice(nd::s![j..(j+block_size/2)]);
                let ib = vector.slice(nd::s![(j+block_size/2)..(j+block_size)]);
                
                let mut oa = self.temp.slice_mut(nd::s![j..(j+block_size/2)]);
                if value == 1 {
                    // If this is X, swap
                    oa.assign(&ib);
                } else if value == 2 {
                    // If this is Y, swap and scale
                    oa.assign(&ib);
                    oa *= -c64::i();
                } else if value == 3 {
                    // If this is Z, just scale
                    oa.assign(&ia);
                }

                // Do the same again on the other half of the block.
                let mut ob = self.temp.slice_mut(nd::s![(j+block_size/2)..(j+block_size)]);
                if value == 1 {
                    ob.assign(&ia);
                } else if value == 2 {
                    ob.assign(&ia);
                    ob *= c64::i();
                } else if value == 3 {
                    ob.assign(&ib);
                    ob *= c64::from(-1.0);
                }

                // Update this block of the vector.
                vector.slice_mut(nd::s![j..(j+block_size)])
                    .assign(&self.temp.slice(nd::s![j..(j+block_size)]));
            }
        }

        // Now to apply this to the GSLC we use the CNOT ladder construction
        // so we need to find a qubit which does not have the I Pauli applied.
        let first = if let Some(first) = self.string.iter().position(|&p| p > 0) {
            first
        } else {
            // If the string is just all identities, there is nothing to do here.
            return
        };

        // Then first do the local Cliffords on each qubit
        for (i, &v) in self.string.iter().enumerate() {
            match v {
                0 => continue,
                1 => gslc.apply_h(i),
                2 => {
                    // X[pi/2]
                    gslc.apply_h(i);
                    gslc.apply_s(i);
                    gslc.apply_h(i);
                },
                3 => continue,
                _ => unreachable!()
            }
        }

        // Now we have the phase-gadget. This is just the form where
        // everything is connected to one qubit, for simplicity.
        for (i, &v) in self.string.iter().enumerate() {
            if v > 0 && i != first {
                gslc.apply_h(first);
                gslc.apply_cz(first, i);
                gslc.apply_h(first);
            }
        }

        gslc.apply_s(first);

        for (i, &v) in self.string.iter().enumerate() {
            if v > 0 && i != first {
                gslc.apply_h(first);
                gslc.apply_cz(first, i);
                gslc.apply_h(first);
            }
        }

        // And then do the trailing local Cliffords
        for (i, &v) in self.string.iter().enumerate() {
            match v {
                0 => continue,
                1 => gslc.apply_h(i),
                2 => {
                    // X[-pi/2]
                    gslc.apply_h(i);
                    gslc.apply_z(i);
                    gslc.apply_s(i);
                    gslc.apply_h(i);
                },
                3 => continue,
                _ => unreachable!()
            }
        }
    }
}

/// The main structure of the program. Provides a way
/// to search for a Clifford decomposition of a target state.
pub struct RandomWalk<S> {
    m: usize,
    n: usize,
    moves: usize,
    chi: usize,
    rng: rand::rngs::ThreadRng,
    beta: S,
    paulis: PauliGenerator,
    target: nd::Array1<c64>,
    proj: nd::Array1<c64>,
    state: nd::Array2<c64>,
    prev: nd::Array1<c64>,
    gslcs: Vec<gslc::GSLC>,
    prevg: gslc::GSLC,
    decomp: qr::QRDecomposition,
    fitness: f64
}

impl<S: Iterator<Item = f64>> RandomWalk<S> {
    /// Construct a new random walk process that searches for the state given in target.
    /// n is the number of qubits in the state, chi is the number of terms in the decomposition
    /// beta is a geometric sequence defining the annealing parameter.
    pub fn new(n: usize, chi: usize, m: usize, beta: S, target: nd::Array1<c64>) -> Self {
        let state = nd::Array2::from_elem((1 << n, chi), 1.0.into());
        let decomp = qr::QRDecomposition::new(&state);
        let proj = target.clone();
        let only_reals = target.iter().all(|x| x.im == 0.0);
        RandomWalk {
            m, n, chi, beta, rng: rand::thread_rng(),
            moves: 0,
            paulis: PauliGenerator::new(n, only_reals),
            target,
            proj,
            state,
            prev: nd::Array1::from_elem(1 << n, 0.0.into()),
            gslcs: vec![gslc::GSLC::new(n); chi],
            prevg: gslc::GSLC::new(n),
            decomp,
            fitness: 0.0
        }
    }

    // Apply a random Pauli exponential to a random term.
    fn make_move(&mut self, index: usize) {
        self.moves += 1;
        let mut column = self.state.slice_mut(nd::s![.., index]);
        // prev = column
        column.assign_to(&mut self.prev);
        self.prevg = self.gslcs[index].clone();
        // column = P*prev
        self.paulis.apply_random(&mut self.rng, &mut column, &mut self.gslcs[index]);
        // column = (I + P)*prev
        column += &self.prev;
        // column = k*(I + P)*prev = e^(i(pi/4)P)*prev
        column /= c64::from(column.norm());
        // column = e^(i(pi/4)P)*prev - prev
        //column -= &self.prev;
        // qr_column = (e^(i(pi/4)P)*prev - prev) + prev = e^(i(pi/4)P)*prev
        //self.decomp.update_column(index, column.view());
        // column = e^(i(pi/4)P)*prev
        //column += &self.prev;
        // To prevent error from accumulating, periodically we will reset the QR.
        if true {//self.moves % (2 * self.n) == 0 {
            self.decomp = qr::QRDecomposition::new(&self.state);
        }
    }

    // Undo whatever what just applied.
    fn unmake_move(&mut self, index: usize) {
        let mut column = self.state.slice_mut(nd::s![.., index]);
        // column = -e^(i(pi/4)P)*prev
        //column *= c64::from(-1.0);
        // column = prev - e^(i(pi/4)P)*prev 
        //column += &self.prev;
        // qr_column = e^(i(pi/4)P)*prev + (prev - e^(i(pi/4)P)*prev) = prev
        //self.decomp.update_column(index, column.view());
        // column = prev
        column.assign(&self.prev);
        self.gslcs[index] = self.prevg.clone();
    }

    // Project the target into the subspace formed
    // by the decomposition terms and measure its norm.
    // When this is 1, a decomposition has been found.
    fn compute_fitness(&mut self) -> f64 {
        self.proj.assign(&self.target);
        self.decomp.project(&mut self.proj);
        self.proj.norm()
    }

    // Get the coefficients of the terms that make up the decomposition.
    fn get_coeffs(&mut self) -> Vec<c64> {
        self.decomp.get_coeffs(&self.target).iter().copied().collect()
    }

    /// Execute one step of the annealing schedule
    pub fn step(&mut self) -> Option<f64> {
        let beta = self.beta.next()?;

        for _ in 0..self.m {
            let index = self.rng.gen_range(0..self.chi);

            self.make_move(index);

            let new_fitness = self.compute_fitness();
            let accept_prob = (-beta * (self.fitness - new_fitness)).exp();

            if self.rng.gen_range(0.0..1.0f64) <= accept_prob {
                self.fitness = new_fitness;
            } else {
                self.unmake_move(index);
            }

            if (self.fitness - 1.0).abs() <= 1e-6 {
                return Some(self.fitness)
            }
        }
       
        Some(self.fitness)
    }

    /// Return the maximum fitness achieved,
    /// statevectors of the terms, and GSLC forms of the terms.
    pub fn finish(mut self) -> (f64, nd::Array2<c64>, Vec<gslc::GSLC>) {
        let coeffs = self.get_coeffs();
        for (i, coeff) in coeffs.into_iter().enumerate() {
            let mut col = self.state.column_mut(i);
            col *= coeff;
        }

        (self.fitness, self.state, self.gslcs)
    }
}

impl<S: Iterator<Item = f64>> Iterator for RandomWalk<S> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let fitness = self.step()?;
        if (fitness - 1.0).abs() <= 1e-6 {
            None
        } else {
            Some(fitness)
        }
    }
}
