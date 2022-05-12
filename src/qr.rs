use ndarray as nd;
use ndarray_linalg as ndl;

use nd::s;
use ndl::{c64, QR, Norm, Solve};

/// An implementation of an online QR decomposition.
/// The rank-1 update algorithm is taken from:
/// "Daniel JW, Gragg WB, Kaufman L, Stewart GW. Reorthogonalization and stable algorithms 
/// for updating the Gram-Schmidt QR factorization. Mathematics of Computation. 1976;30(136):772-95."
/// When A is m x n, this structure takes m(n + 1) + (n + 1)n + (n + 1) space total.
/// Each update is O(mn) time, and the initial construction is O(mn^2).
pub struct QRDecomposition {
    q: nd::Array2<c64>,
    r: nd::Array2<c64>,
    w: nd::Array1<c64>
}

impl QRDecomposition {
    pub fn new(init: &nd::Array2<c64>) -> QRDecomposition {
        let (mut q, mut r) = init.qr().unwrap();
        q.push_column(nd::Array1::zeros(q.shape()[0]).view()).unwrap();
        r.push_row(nd::Array1::zeros(r.shape()[1]).view()).unwrap();
        let w = nd::Array1::zeros(r.shape()[0]);
        QRDecomposition { q, r, w }
    }

    /// Perform a rank-1 update of the decomposition: A => A + u@(e_i)^T
    pub fn update_column(&mut self, i: usize, u: nd::ArrayView1<c64>) {
        let n = self.r.shape()[1];
        // Compute the coefficients of the projection in the image of Q
        // and the its component orthogonal to Q
        // w[:-1] = Q[:,:-1].T @ u
        nd::linalg::general_mat_vec_mul(
            c64::from(1.0),
            &self.q.slice(s![.., ..n]).t(),
            &u,
            c64::from(0.0),
            &mut self.w.slice_mut(s![..n])
        );
        // Q[:,-1] = u - Q[:,:-1] @ w[:-1]
        let (qmain, mut qlast) = self.q.multi_slice_mut((s![.., ..n], s![.., n]));
        qlast.assign(&u);
        nd::linalg::general_mat_vec_mul(
            c64::from(-1.0),
            &qmain,
            &self.w.slice(s![..n]),
            c64::from(1.0),
            &mut qlast
        );
        // rho = np.linalg.norm(Q[:,-1])
        let rho = qlast.norm();
        // Q[:,-1] /= rho
        if rho != 0.0 {
            qlast /= c64::from(rho);
        }
        // w[-1] = rho
        self.w[n] = c64::from(rho);
        // Q now has the orthogonal component in the last column.
        // Apply the update to R - R is now not triangular in one column.
        // R[-1,:] = 0
        self.r.slice_mut(s![n, ..]).fill(0.0.into());
        // R += np.outer(w, v)
        self.r.column_mut(i).scaled_add(1.0.into(), &self.w);

        // Do givens rotations on the nontriangular column and 
        // make the whole matrix upper Hessenberg.
        // for i in range(n, 0, -1):
        for i in (1..=n).rev() {
            // c, s = givens(w[i-1, 0], w[i, 0])
            let (c, s) = givens::generate(self.w[i-1], self.w[i]);
            // givens_pre(w, i-1, i, c, s)
            givens::apply_pre(
                &mut self.w.slice_mut(s![.., nd::NewAxis]), 
                i-1, i, c, s
            );
            // givens_pre(R, i-1, i, c, s)
            givens::apply_pre(
                &mut self.r.view_mut(),
                i-1, i, c, s
            );
            // givens_post(Q, i-1, i, c, s)
            givens::apply_post(
                &mut self.q.view_mut(),
                i-1, i, c, s
            );
        }

        // Apply Givens rotations to make R triangular again by 
        // zeroing the subdiagonal. By magic, the last column of Q and 
        // row of R will be zero after this, so we can forget them!
        // for i in range(0, n):
        for i in 0..n {
            // c, s = givens(R[i, i], R[i+1, i])
            let (c, s) = givens::generate(self.r[(i, i)], self.r[(i+1, i)]);
            // givens_pre(R, i, i+1, c, s)
            givens::apply_pre(
                &mut self.r.view_mut(),
                i, i+1, c, s
            );
            // givens_post(Q, i, i+1, c, s)
            givens::apply_post(
                &mut self.q.view_mut(),
                i, i+1, c, s
            )
        }
    }

    /// Project a vector into the span of the columns of A.
    /// i.e v = Q @ Q.T @ v
    pub fn project(&mut self, v: &mut nd::Array1<c64>) {
        // w[:-1] = Q[:, :-1].T @ v
        let n = self.r.shape()[1];
        nd::linalg::general_mat_vec_mul(
            1.0.into(),
            &self.q.slice(s![.., ..n]).t(),
            v,
            0.0.into(),
            &mut self.w.slice_mut(s![..n])
        );
        // v = Q[:, :-1] @ w[:-1]
        nd::linalg::general_mat_vec_mul(
            1.0.into(),
            &self.q.slice(s![.., ..n]),
            &self.w.slice(s![..n]),
            0.0.into(),
            v
        )
    }

    /// Get the coefficients of the projection of a vector
    /// into the column space of A.
    pub fn get_coeffs(&mut self, v: &nd::Array1<c64>) -> nd::ArrayView1<c64> {
        let n = self.r.shape()[1];
        // w[:-1] = Q[:, :-1].T @ v
        nd::linalg::general_mat_vec_mul(
            1.0.into(),
            &self.q.slice(s![.., ..n]).t(),
            v,
            0.0.into(),
            &mut self.w.slice_mut(s![..n])
        );
        // w[:-1] = np.linalg.solve(R[:-1, :], w[:-1])
        self.r.slice(s![..n, ..]).solve_inplace(&mut self.w.slice_mut(s![..n])).unwrap();
        // return w[:-1]
        self.w.slice(s![..n])
    }
}

mod givens {
    use ndarray as nd;
    use ndarray_linalg::c64;

    // Generate the coefficients for a Givens rotation that takes [a, b] to [c, 0]
    pub fn generate(a: c64, b: c64) -> (c64, c64) {
        let (c, s) = if b == c64::from(0.0) {
            (1.0.into(), 0.0.into())
        } else if a == c64::from(0.0) {
            (0.0.into(), -b / b.norm())
        } else {
            let denom = (a.norm_sqr() + b.norm_sqr()).sqrt();
            let absa = a.norm();
            ((absa / denom).into(), -(a.conj() / absa) * b / denom)
        };

        (c, s)
    }

    // Premultiply a matrix by the adjoint of a Givens rotation: A = J(c, s)_(i, k)^* @ A
    pub fn apply_pre(a: &mut nd::ArrayViewMut2<c64>, i: usize, k: usize, c: c64, s: c64) {
        for j in 0..a.shape()[1] {
            let t1 = a[(i, j)];
            let t2 = a[(k, j)];
            a[(i, j)] = c*t1 - s.conj()*t2;
            a[(k, j)] = s*t1 + c*t2;
        }
    }

    // Postmultiply a matrix by a Givens rotation: A = A @ J(c, s)_(i, k)
    pub fn apply_post(a: &mut nd::ArrayViewMut2<c64>, i: usize, k: usize, c: c64, s: c64) {
        for j in 0..a.shape()[0] {
            let t1 = a[(j, i)];
            let t2 = a[(j, k)];
            a[(j, i)] = c*t1 - s*t2;
            a[(j, k)] = s.conj()*t1 + c*t2;
        }
    }
}
