import unittest
import numpy as np
import scipy
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt.solve import optimize_wireframe

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
        print('    Q = ')
        print_matrix(Q)

        
    def test_3_2x4_wf_rcls(self):

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
        print('    Q = ')
        print_matrix(Q)

        print('Now calling RCLS')
        res = optimize_wireframe(wf, 'rcls', {'reg_W': 1e-10}, 
                                 surf_plas=surf_plas)

