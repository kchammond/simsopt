import unittest
import numpy as np
import scipy
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt.field import WireframeField
from simsopt.solve import optimize_wireframe, \
                          bnorm_obj_matrices, \
                          rcls_wireframe, \
                          regularized_constrained_least_squares

def print_matrix(M, precise=False):

    for i in range(M.shape[0]):
        line = '            '
        for j in range(M.shape[1]):
            if not precise:
                line += '%5.2f  ' % (M[i][j])
            else:
                line += '%10.2e ' % (M[i][j])
        print(line)

def calc_qr_twice():

    Q, R = scipy.linalg.qr(np.eye(6))
    print('    First QR evaluation:')
    if np.all(np.isfinite(R)):
        print('     R Contains no infinite/NaN elements.')
    else:
        print('     R Contains some infinite/NaN elements!!!')
    print('    Second QR evaluation:')
    Q, R = scipy.linalg.qr(np.eye(6))
    if np.all(np.isfinite(R)):
        print('     R Contains no infinite/NaN elements.')
    else:
        print('     R Contains some infinite/NaN elements!!!')


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
        
        print('Trial 1: Testing QR directly on the constraint matrices')
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

        print('Trial 2: Redoing with processing of C as done by RCLS function')
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

        print('Trial 3: Redoing, matching all operations on C')
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

        print('Trial 4: Performing the seemingly relevant steps in regularized_constrained_least_squares up to QR')
        C4, d4 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)

        # Recast inputs as Numpy arrays
        Ctra = np.array(C4).T # Transpose will be used for the calculations
    
        # Check the inputs
        n_C, p = Ctra.shape
                
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

        print('Trial 5: Performing the seemingly relevant steps plus d operations')
        C5, d5 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)

        # Recast inputs as Numpy arrays
        Ctra5 = np.array(C5).T # Transpose will be used for the calculations
        dvec = np.array(d5).reshape((-1,1))
    
        # Check the inputs
        n_C, p = Ctra5.shape
                
        # Compute the QR factorization of the transpose of the constraint matrix
        Qfull, Rtall = scipy.linalg.qr(Ctra5)
        Q1mat = Qfull[:,:p]  # Orthonormal vectors in the constrained subspace
        Q2mat = Qfull[:,p:]  # Orthonormal vectors in the free subspace
        Rmat = Rtall[:p,:]

        print('    Transpose of constraint matrix:')
        print_matrix(Ctra5)
        print('    R from QR-factorization:')
        if np.all(np.isfinite(Rmat)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(Rmat.T[:,:6])

        print('Trial 6: Performing the seemingly relevant steps plus A, b, and d operations')
        A6, b6 = bnorm_obj_matrices(wf, surf_plas)
        C6, d6 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)

        # Recast inputs as Numpy arrays
        Amat = np.array(A6)
        bvec = np.array(b6).reshape((-1,1))
        Ctra6 = np.array(C6).T # Transpose will be used for the calculations
        dvec = np.array(d6).reshape((-1,1))
    
        # Check the inputs
        m, n = Amat.shape
        n_C, p = Ctra6.shape
                
        # Compute the QR factorization of the transpose of the constraint matrix
        Qfull, Rtall = scipy.linalg.qr(Ctra6)
        Q1mat = Qfull[:,:p]  # Orthonormal vectors in the constrained subspace
        Q2mat = Qfull[:,p:]  # Orthonormal vectors in the free subspace
        Rmat = Rtall[:p,:]

        print('    Transpose of constraint matrix:')
        print_matrix(Ctra6)
        print('    R from QR-factorization:')
        if np.all(np.isfinite(Rmat)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(Rmat.T[:,:6])
        print('    Difference between Ctra from trials 5 and 6:')
        print_matrix(Ctra6 - Ctra5, precise=True)

        print('Trial 7: Performing the seemingly relevant steps plus A, b, and d operations; A, b not recast')
        A7, b7 = bnorm_obj_matrices(wf, surf_plas)
        C7, d7 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)

        # Recast inputs as Numpy arrays
        Ctra7 = np.array(C7).T # Transpose will be used for the calculations
        dvec = np.array(d7).reshape((-1,1))
    
        # Check the inputs
        n_C, p = Ctra7.shape
                
        # Compute the QR factorization of the transpose of the constraint matrix
        Qfull, Rtall = scipy.linalg.qr(Ctra7)
        Q1mat = Qfull[:,:p]  # Orthonormal vectors in the constrained subspace
        Q2mat = Qfull[:,p:]  # Orthonormal vectors in the free subspace
        Rmat = Rtall[:p,:]

        print('    Transpose of constraint matrix:')
        print_matrix(Ctra6)
        print('    R from QR-factorization:')
        if np.all(np.isfinite(Rmat)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(Rmat.T[:,:6])
        print('    Difference between Ctra from trials 5 and 7:')
        print_matrix(Ctra7 - Ctra5, precise=True)

        print('Trial 8: Performing the seemingly relevant steps plus A, b, and d operations; A, b calculated after C, d')
        C8, d8 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)
        A8, b8 = bnorm_obj_matrices(wf, surf_plas)

        # Recast inputs as Numpy arrays
        Ctra8 = np.array(C8).T # Transpose will be used for the calculations
        dvec = np.array(d8).reshape((-1,1))
    
        # Check the inputs
        n_C, p = Ctra8.shape
                
        # Compute the QR factorization of the transpose of the constraint matrix
        Qfull, Rtall = scipy.linalg.qr(Ctra8)
        Q1mat = Qfull[:,:p]  # Orthonormal vectors in the constrained subspace
        Q2mat = Qfull[:,p:]  # Orthonormal vectors in the free subspace
        Rmat = Rtall[:p,:]

        print('    Transpose of constraint matrix:')
        print_matrix(Ctra8)
        print('    R from QR-factorization:')
        if np.all(np.isfinite(Rmat)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(Rmat.T[:,:6])
        print('    Difference between Ctra from trials 5 and 8:')
        print_matrix(Ctra8 - Ctra5, precise=True)

        print('Trial 9: Performing the seemingly relevant steps plus A, b, and d operations; A, b calculated after C, d get recast')
        C9, d9 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)

        # Recast inputs as Numpy arrays
        Ctra9 = np.array(C9).T # Transpose will be used for the calculations
        dvec = np.array(d9).reshape((-1,1))
    
        A9, b9 = bnorm_obj_matrices(wf, surf_plas)

        # Check the inputs
        n_C, p = Ctra9.shape
                
        # Compute the QR factorization of the transpose of the constraint matrix
        Qfull, Rtall = scipy.linalg.qr(Ctra9)
        Q1mat = Qfull[:,:p]  # Orthonormal vectors in the constrained subspace
        Q2mat = Qfull[:,p:]  # Orthonormal vectors in the free subspace
        Rmat = Rtall[:p,:]

        print('    Transpose of constraint matrix:')
        print_matrix(Ctra9)
        print('    R from QR-factorization:')
        if np.all(np.isfinite(Rmat)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(Rmat.T[:,:6])
        print('    Difference between Ctra from trials 5 and 8:')
        print_matrix(Ctra8 - Ctra5, precise=True)

        print('Trial 10: Performing the seemingly relevant steps plus A, b, d, and W operations... but doing QR with a C.T that previously worked!')
        A10, b10 = bnorm_obj_matrices(wf, surf_plas)
        C10, d10 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)

        # Recast inputs as Numpy arrays
        Amat = np.array(A10)
        bvec = np.array(b10).reshape((-1,1))
        Ctra10 = np.array(C10).T # Transpose will be used for the calculations
        dvec = np.array(d10).reshape((-1,1))
    
        # Check the inputs
        m, n = Amat.shape
        n_C, p = Ctra10.shape
                
        if np.isscalar(reg_W):
            Wmat = reg_W*np.eye(n)
        else:
            print('Unexpected W operation needed!')

        # Compute the QR factorization of the transpose of the constraint matrix
        Qfull, Rtall = scipy.linalg.qr(Ctra5)
        Q1mat = Qfull[:,:p]  # Orthonormal vectors in the constrained subspace
        Q2mat = Qfull[:,p:]  # Orthonormal vectors in the free subspace
        Rmat = Rtall[:p,:]

        print('    Transpose of constraint matrix (from Trial 5):')
        print_matrix(Ctra5)
        print('    R from QR-factorization:')
        if np.all(np.isfinite(Rmat)):
            print('        Contains no infinite/NaN elements!')
        else:
            print('        Contains some infinite/NaN elements!')
        print('        R.T[:,:6] = ')
        print_matrix(Rmat.T[:,:6])

        print('Trial 11: Performing the exact sequence of steps in regularized_constrained_least_squares up to QR, but performing the QR twice')
        A11, b11 = bnorm_obj_matrices(wf, surf_plas)
        C11, d11 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)

        # Recast inputs as Numpy arrays
        Amat = np.array(A11)
        bvec = np.array(b11).reshape((-1,1))
        Ctra11 = np.array(C11).T # Transpose will be used for the calculations
        dvec = np.array(d11).reshape((-1,1))
    
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
        Qfull, Rtall = scipy.linalg.qr(Ctra11)
        Q1mat = Qfull[:,:p]  # Orthonormal vectors in the constrained subspace
        Q2mat = Qfull[:,p:]  # Orthonormal vectors in the free subspace
        Rmat = Rtall[:p,:]

        # Compute the QR factorization of the transpose of the constraint matrix
        Qfull, Rtall = scipy.linalg.qr(Ctra11)
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


        print('Trial 5: repeat')
        C5, d5 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)

        # Recast inputs as Numpy arrays
        Ctra5 = np.array(C5).T # Transpose will be used for the calculations
        dvec = np.array(d5).reshape((-1,1))
    
        # Check the inputs
        n_C, p = Ctra5.shape
                
        # Compute the QR factorization of the transpose of the constraint matrix
        Qfull, Rtall = scipy.linalg.qr(Ctra5)
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


        print('Trial 12: Calling regularized_constrained_least_squares')
        A12, b12 = bnorm_obj_matrices(wf, surf_plas)
        C12, d12 = wf.constraint_matrices(assume_no_crossings=False,
                                        remove_constrained_segments=True)
        x12 = regularized_constrained_least_squares(A12, b12, reg_W, C12, d12)

        print('Trial 13: Calling rcls_wireframe')
        A13, b13 = bnorm_obj_matrices(wf, surf_plas)
        x13, f_B13, f_R13, f13 = rcls_wireframe(wf, A13, b13, reg_W, False, True)

        print('Trial 14: Calling optimize_wireframe')
        res = optimize_wireframe(wf, 'rcls', {'reg_W': reg_W}, 
                                 surf_plas=surf_plas)

    def test_4_min_working(self):

        surf_plas = SurfaceRZFourier(nfp=2)
        surf_wf = SurfaceRZFourier(nfp=2)
        surf_wf.extend_via_normal(1.0)
        wf = ToroidalWireframe(surf_wf, 2, 4)
 
        print('Trial 1: forming A, b and then finding QR of unrelated matrix')
        A, b = bnorm_obj_matrices(wf, surf_plas)
        calc_qr_twice()


        print('Trial 2: calculating a WireframeField, then finding QR of unrelated matrix')
        print('  2a: forming the wireframe field')
        mf_wf = WireframeField(wf)
        calc_qr_twice()

        print('  2b: setting the points')
        mf_wf.set_points(surf_plas.gamma().reshape((-1,3)))
        calc_qr_twice()

        print('  2h: performing matrix calculation steps all at once')
        points = mf_wf.get_points_cart_ref()
        nPoints = len(points)
        n = surf_plas.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:,:,None]
        fac = np.sqrt(absn/float(absn.size))
        matrix = np.ascontiguousarray(np.zeros((nPoints, wf.nSegments)))
        dB_dsc = mf_wf.dB_by_dsegmentcurrents(0)
        for i in range(wf.nSegments):
            dB_dsc_i = dB_dsc[i].reshape(n.shape)
            matrix[:,i] = (fac*np.sum(dB_dsc_i * unitn, axis=2)).reshape((-1))
        calc_qr_twice()

        print('  2c: calculating the A matrix (unweighted)')
        Aunw = mf_wf.dBnormal_by_dsegmentcurrents_matrix(surf_plas)
        calc_qr_twice()

        print('  2d: calculating the magnetic field')
        Bfield = mf_wf.B()
        calc_qr_twice()

        print('  2e: calculating derivative of field wrt segment currents')
        dB_dsc = mf_wf.dB_by_dsegmentcurrents(0)
        calc_qr_twice()

        print('  2f: retrieving magnetic field test points')
        points = mf_wf.get_points_cart_ref()
        nPoints = len(points)
        calc_qr_twice()

        print('  2g: calculating the unit normal')
        n = surf_plas.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:,:,None]
        fac = np.sqrt(absn/float(absn.size))
        calc_qr_twice()

        print('  2e: forming a matrix')
        matrix = np.ascontiguousarray(np.zeros((nPoints, wf.nSegments)))
        calc_qr_twice()

        print('  2f: populating the matrix')
        for i in range(wf.nSegments):
            dB_dsc_i = dB_dsc[i].reshape(n.shape)
            matrix[:,i] = (fac*np.sum(dB_dsc_i * unitn, axis=2)).reshape((-1))
        calc_qr_twice()

        print('  2g: calculating the A matrix (weighted)')
        A = mf_wf.dBnormal_by_dsegmentcurrents_matrix(surf_plas,
                area_weighted=True)
        calc_qr_twice()

        print('Is `A` the same as `matrix`? ', np.allclose(A, matrix))


