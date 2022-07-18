from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.magneticfieldclasses import InterpolatedField, DipoleField
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.objectives.fluxobjective import SquaredFlux
import simsoptpp as sopp
import numpy as np
import unittest
# File for the desired boundary magnetic surface:
filename = 'tests/test_files/input.LandremanPaul2021_QA'


class Testing(unittest.TestCase):

    def test_bad_params(self):
        """
            Test that the permanent magnet optimizer initialization
            correctly catches bad instances of the function arguments.
        """
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, coil_offset=0.1, dr=-0.05, plasma_offset=0.1,
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, coil_offset=-0.1, dr=0.05, plasma_offset=0.1,
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, coil_offset=0.1, dr=0.05, plasma_offset=0.0,
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                10,
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=10
            )
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_outer_surface=10
            )
        inner = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta-2)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=inner, rz_outer_surface=outer 
            )
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta+2)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=inner, rz_outer_surface=outer 
            )
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi-2, ntheta=ntheta)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=inner, rz_outer_surface=outer 
            )
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi+2, ntheta=ntheta)
        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, rz_inner_surface=inner, rz_outer_surface=outer 
            )

        with self.assertRaises(FileNotFoundError):
            PermanentMagnetOptimizer(
                s, filename='nonsense'
            )

        with self.assertRaises(TypeError):
            PermanentMagnetOptimizer(
                s, surface_flag='wout', filename=filename
            )

        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, is_premade_famus_grid=True 
            )

        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, Bn=0.0 
            )

        with self.assertRaises(ValueError):
            PermanentMagnetOptimizer(
                s, Bn=np.zeros((nphi, ntheta // 2))
            )

        with self.assertRaises(OSError):
            PermanentMagnetOptimizer(
                s, is_premade_famus_grid=True, pms_name='nonsense' 
            )

    def test_optimize_bad_parameters(self):
        """
            Test that the permanent magnet optimize
            correctly catches bad instances of the function arguments.
        """
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        # Create some initial coils:
        base_curves = create_equally_spaced_curves(2, s.nfp, stellsym=False, R0=1.0, R1=0.5, order=5)
        base_currents = [Current(1e5) for i in range(2)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        # Create PM class
        Bn = np.sum(bs.B().reshape(nphi, ntheta, 3) * s.unitnormal(), axis=-1)
        pm_opt = PermanentMagnetOptimizer(s, filename=filename, dr=0.15, Bn=Bn)

        # Note that the rest of the optimization parameters are checked
        # interactively when python permanent_magnet_optimization.py True 
        # is used on the command line. 
        with self.assertRaises(ValueError):
            _, _, _, _, = pm_opt._optimize()

        with self.assertRaises(ValueError):
            _, _, _, = pm_opt._optimize(m0=np.zeros(pm_opt.ndipoles * 3 // 2))

        with self.assertRaises(ValueError):
            _, _, _, = pm_opt._optimize(m0=np.zeros((pm_opt.ndipoles, 3)))

        with self.assertRaises(ValueError):
            mmax = np.ravel(np.array([pm_opt.m_maxima, pm_opt.m_maxima, pm_opt.m_maxima]).T)
            _, _, _, = pm_opt._optimize(m0=np.ones((pm_opt.ndipoles * 3)) * mmax)

    def test_projected_normal(self):
        """
            Make two RZFourier surfaces, extend one of them with
            the projected normal vector in the RZ plane, and check
            that both surfaces still have the same values of the
            toroidal angle. 
        """
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        p = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s.extend_via_projected_normal(s.quadpoints_phi, 0.1)
        pgamma = p.gamma().reshape(-1, 3)
        pphi = np.arctan2(pgamma[:, 1], pgamma[:, 0])
        sgamma = s.gamma().reshape(-1, 3)
        sphi = np.arctan2(sgamma[:, 1], sgamma[:, 0])
        assert np.allclose(pphi, sphi)
        assert not np.allclose(pgamma[:, 0], sgamma[:, 0])
        assert not np.allclose(pgamma[:, 1], sgamma[:, 1])
        assert not np.allclose(pgamma[:, 2], sgamma[:, 2])

    def test_Bn(self):
        """
            Creates a realistic QA configuration, some permanent magnets, and optimizes it
            with the default parameters. Checks the the pm_opt object agrees with Bn and
            f_B with the DipoleField + SquaredFlux way of calculating Bn and f_B.
        """
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        # Create some initial coils:
        base_curves = create_equally_spaced_curves(2, s.nfp, stellsym=False, R0=1.0, R1=0.5, order=5)
        base_currents = [Current(1e5) for i in range(2)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
        # Create PM class
        Bn = np.sum(bs.B().reshape(nphi, ntheta, 3) * s.unitnormal(), axis=-1)
        pm_opt = PermanentMagnetOptimizer(s, filename=filename, dr=0.15, Bn=Bn)
        _, _, _, = pm_opt._optimize()
        b_dipole = DipoleField(pm_opt)
        b_dipole.set_points(s.gamma().reshape(-1, 3))

        # check Bn
        Nnorms = np.ravel(np.sqrt(np.sum(s.normal() ** 2, axis=-1)))
        Ngrid = nphi * ntheta
        Bn_Am = (pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms) 
        assert np.allclose(Bn_Am.reshape(nphi, ntheta), np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
        # check <Bn>
        B_opt = np.mean(np.abs(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms))
        B_dipole_field = np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        assert np.isclose(B_opt, B_dipole_field)
        # check integral Bn^2
        f_B_Am = 0.5 * np.linalg.norm(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj, ord=2) ** 2
        f_B = SquaredFlux(s, b_dipole, -Bn).J()
        assert np.isclose(f_B, f_B_Am)


if __name__ == "__main__":
    unittest.main()