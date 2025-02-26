import unittest
import numpy as np
import scipy
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt.solve import optimize_wireframe, \
                          bnorm_obj_matrices, \
                          rcls_wireframe, \
                          regularized_constrained_least_squares

def print_matrix(M):

    for i in range(M.shape[0]):
        line = '            '
        for j in range(M.shape[1]):
            line += '%5.2f  ' % (M[i][j])
        print(line)



class WireframeFactorizationTests(unittest.TestCase):

    def test_1_qr_identity(self):

        Mat = np.eye(3)

        Q, R = scipy.linalg.qr(Mat)

        print('R matrix from QR-factorization of 3x3 identity:')
        print_matrix(R)

    def test_2_small_wireframe(self):

        surf_wf = SurfaceRZFourier(nfp=2)
        surf_wf.extend_via_normal(1.0)


        wf = ToroidalWireframe(surf_wf, 2, 2)
        C, d = wf.constraint_matrices()

        print('transpose of constraint matrix for 2x2 wireframe:')
        print_matrix(C.T)

        Q, R = scipy.linalg.qr(C.T)
        
        print('R from QR-factorization of 2x2 wireframe constr matrix:')
        if np.all(np.isfinite(R)):
            print('    Contains no infinite/NaN elements!')
        else:
            print('    Contains some infinite/NaN elements!')
        print('    R.T = ')
        print_matrix(R.T)
        #print('    Q = ')
        #print_matrix(Q)

        
    def test_3_2x4_wf_rcls(self):

        reg_W = 1e-10
        surf_plas = SurfaceRZFourier(nfp=2)
        surf_wf = SurfaceRZFourier(nfp=2)
        surf_wf.extend_via_normal(1.0)
        wf = ToroidalWireframe(surf_wf, 2, 4)
        
        print('Testing QR directly on the constraint matrices')
        C, d = wf.constraint_matrices()

        print('    Transpose of constraint matrix:')
        print_matrix(C.T)

        Q, R = scipy.linalg.qr(C.T)
        print('    R from QR-factorization of 2x4 wireframe constr matrix:')
        if np.all(np.isfinite(R)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(R.T[:,:6])
        #print('    Q = ')
        #print_matrix(Q)

        print('Redoing with processing of C as done by RCLS function')
        Ctra = np.array(C).T
        print('    Transpose of constraint matrix:')
        print_matrix(Ctra)
        Q2, R2 = scipy.linalg.qr(Ctra)
        print('    R from QR-factorization of 2x4 wf constr matrix (processed):')
        if np.all(np.isfinite(R2)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(R2.T[:,:6])
        #print('    Q = ')
        #print_matrix(Q2)

        print('Redoing, matching all operations on C')
        C3, d3 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)
        C3tra = np.array(C3).T
        print('    Transpose of constraint matrix:')
        print_matrix(C3tra)
        Q3, R3 = scipy.linalg.qr(C3tra)
        print('    R from QR-factorization of 2x4 wf constr matrix (processed):')
        if np.all(np.isfinite(R3)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(R3.T[:,:6])

        print('Performing the exact sequence of steps in regularized_constrained_least_squares up to QR')
        A4, b4 = bnorm_obj_matrices(wf, surf_plas)
        C4, d4 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)

        # Recast inputs as Numpy arrays
        Amat = np.array(A4)
        bvec = np.array(b4).reshape((-1,1))
        Ctra = np.array(C4).T # Transpose will be used for the calculations
        dvec = np.array(d4).reshape((-1,1))
    
        # Check the inputs
        m, n = Amat.shape
        if bvec.shape[0] != m:
            raise ValueError('Number of elements in b must match rows in A')
        n_C, p = Ctra.shape
        if n_C != n:
            raise ValueError('A and C must have the same number of columns')
        if dvec.shape[0] != p:
            raise ValueError('Number of elements in d must match rows in C')
    
        if np.isscalar(reg_W):
            Wmat = reg_W*np.eye(n)
        else:
            Wmat = np.squeeze(reg_W)
            if len(Wmat.shape) == 1:
                if Wmat.shape[0] != n:
                    raise ValueError('Number of elements in vector-form W ' \
                                     'must match columns in A')
                Wmat = np.diag(Wmat)
            elif len(Wmat.shape) == 2:
                if Wmat.shape[0] != n or Wmat.shape[1] != n:
                    raise ValueError('Number of rows and columns in matrix-form W '\
                                     'must both equal number of columns in A')
            else:
                raise ValueError('W must be a scalar, 1d array, or 2d array')
                
        # Compute the QR factorization of the transpose of the constraint matrix
        Qfull, Rtall = scipy.linalg.qr(Ctra)
        Q1mat = Qfull[:,:p]  # Orthonormal vectors in the constrained subspace
        Q2mat = Qfull[:,p:]  # Orthonormal vectors in the free subspace
        Rmat = Rtall[:p,:]

        print('    Transpose of constraint matrix:')
        print_matrix(Ctra)
        print('    R from QR-factorization:')
        if np.all(np.isfinite(Rmat)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(Rmat.T[:,:6])



        print('Calling regularized_constrained_least_squares')
        A5, b5 = bnorm_obj_matrices(wf, surf_plas)
        C5, d5 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)
        x5 = regularized_constrained_least_squares(A5, b5, reg_W, C5, d5)

        print('Calling rcls_wireframe')
        A6, b6 = bnorm_obj_matrices(wf, surf_plas)
        x6, f_B6, f_R6, f6 = rcls_wireframe(wf, A6, b6, reg_W, False, True)

        print('Calling optimize_wireframe')
        res = optimize_wireframe(wf, 'rcls', {'reg_W': reg_W}, 
                                 surf_plas=surf_plas)

