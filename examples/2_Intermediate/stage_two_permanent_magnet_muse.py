#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule, DipoleField
from simsopt.field.coil import Current, Coil, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, MinimumDistance, \
    MeanSquaredCurvature, LpCurveCurvature
from simsopt.geo.plot import plot
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
#from simsopt.field.tracing import SurfaceClassifier, \
#    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data, \
#    IterationStoppingCriterion
import time


def read_focus_coils(filename):
    ncoils = np.loadtxt(filename, skiprows=1, max_rows=1, dtype=int)
    order = np.loadtxt(filename, skiprows=8, max_rows=1, dtype=int)
    coilcurrents = np.zeros(ncoils)
    xc = np.zeros((ncoils, order + 1))
    xs = np.zeros((ncoils, order + 1))
    yc = np.zeros((ncoils, order + 1))
    ys = np.zeros((ncoils, order + 1))
    zc = np.zeros((ncoils, order + 1))
    zs = np.zeros((ncoils, order + 1))
    for i in range(ncoils):
        coilcurrents[i] = np.loadtxt(filename, skiprows=6 + 14 * i, max_rows=1, usecols=1)
        xc[i, :] = np.loadtxt(filename, skiprows=10 + 14 * i, max_rows=1, usecols=range(order + 1))
        xs[i, :] = np.loadtxt(filename, skiprows=11 + 14 * i, max_rows=1, usecols=range(order + 1))
        yc[i, :] = np.loadtxt(filename, skiprows=12 + 14 * i, max_rows=1, usecols=range(order + 1))
        ys[i, :] = np.loadtxt(filename, skiprows=13 + 14 * i, max_rows=1, usecols=range(order + 1))
        zc[i, :] = np.loadtxt(filename, skiprows=14 + 14 * i, max_rows=1, usecols=range(order + 1))
        zs[i, :] = np.loadtxt(filename, skiprows=15 + 14 * i, max_rows=1, usecols=range(order + 1))

    # CurveXYZFourier wants data in order sin_x, cos_x, sin_y, cos_y, ...
    coil_data = np.zeros((order + 1, ncoils * 6))
    for i in range(ncoils):
        coil_data[:, i * 6 + 0] = xs[i, :]
        coil_data[:, i * 6 + 1] = xc[i, :]
        coil_data[:, i * 6 + 2] = ys[i, :]
        coil_data[:, i * 6 + 3] = yc[i, :]
        coil_data[:, i * 6 + 4] = zs[i, :]
        coil_data[:, i * 6 + 5] = zc[i, :]
    # coilcurrents = coilcurrents * 1e3  # rescale from kA to A
    base_currents = [Current(coilcurrents[i]) for i in range(ncoils)]
    ppp = 20
    coils = [CurveXYZFourier(order*ppp, order) for i in range(ncoils)]
    for ic in range(ncoils):
        dofs = coils[ic].dofs
        dofs[0][0] = coil_data[0, 6*ic + 1]
        dofs[1][0] = coil_data[0, 6*ic + 3]
        dofs[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, min(order, coil_data.shape[0]-1)):
            dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].local_x = np.concatenate(dofs)
    return coils, base_currents, ncoils


#try:
#    from mpi4py import MPI
#    comm = MPI.COMM_WORLD
#except ImportError:
#    comm = None

# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
ci = False
nfieldlines = 40 if ci else 40
tmax_fl = 30000 if ci else 50000
degree = 2 if ci else 4

# File for the desired TF coils 
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'muse_tf_coils.focus'
base_curves, base_currents, ncoils = read_focus_coils(filename)
print("Done loading in MUSE coils")
coils = []
for i in range(ncoils):
    coils.append(Coil(base_curves[i], base_currents[i]))
print("Done loading initializing coils in SIMSOPT")

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.MUSE'

# Directory for output
OUT_DIR = "./output_muse/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
t1 = time.time()
nphi = 32
ntheta = 64
s = SurfaceRZFourier.from_focus(filename, range="half period", nphi=nphi, ntheta=ntheta)
t2 = time.time()
print("Done loading in MUSE plasma boundary surface, t = ", t2 - t1)

bs = BiotSavart(coils)
bspoints = np.zeros((nphi, 3))
R0 = s.get_rc(0, 0)
for i in range(nphi):
    bspoints[i] = np.array([R0 * np.cos(s.quadpoints_phi[i]), R0 * np.sin(s.quadpoints_phi[i]), 0.0]) 
bs.set_points(bspoints)
B0 = np.linalg.norm(bs.B(), axis=-1)
B0avg = np.mean(np.linalg.norm(bs.B(), axis=-1))
surface_area = s.area()
bnormalization = B0avg * surface_area
print("Bmag at R = ", R0, ", Z = 0: ", B0) 
print("toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg) 
bs.set_points(s.gamma().reshape((-1, 3)))
print("Done setting up biot savart")

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init_muse")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init_muse", extra_data=pointData)
print("Done writing coils and initial surface to vtk")

# permanent magnet optimization now. 
t1 = time.time()
pm_opt = PermanentMagnetOptimizer(
    s, coil_offset=0.055, dr=0.01, plasma_offset=0.035,
    B_plasma_surface=bs.B().reshape((nphi, ntheta, 3)),
    filename=filename, FOCUS=True
)
t2 = time.time()
max_iter_MwPGP = 10000
print('Done initializing the permanent magnet object')
print('Process took t = ', t2 - t1, ' s')
t1 = time.time()
MwPGP_history, RS_history, m_history, dipoles = pm_opt._optimize(
    max_iter_MwPGP=max_iter_MwPGP, 
    max_iter_RS=10, reg_l0=1e-5,
)
t2 = time.time()
print('Done optimizing the permanent magnet object')
print('Process took t = ', t2 - t1, ' s')
M_max = 1.4 / (4 * np.pi * 1e-7)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) / M_max)

# recompute normal error using the dipole field and bs field
# to check nothing got mistranslated
t1 = time.time()
b_dipole = DipoleField(pm_opt.dipole_grid, dipoles, pm_opt, nfp=s.nfp, stellsym=s.stellsym)
b_dipole.set_points(s.gamma().reshape((-1, 3)))
b_dipole._toVTK("Dipole_Fields_muse")
pm_opt._plot_final_dipoles()
t2 = time.time()
print('Done setting up the Dipole Field class')
print('Process took t = ', t2 - t1, ' s')

# b_dipole._toVTK("Dipole_Fields_surf", dim=())

dphi = (pm_opt.phi[1] - pm_opt.phi[0]) * 2 * np.pi
dtheta = (pm_opt.theta[1] - pm_opt.theta[0]) * 2 * np.pi
print("Average Bn without the PMs = ", 
      np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal() * dphi * dtheta, axis=2))))
print("Average Bn with the PMs = ", 
      np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal() * dphi * dtheta, axis=2))))
print("Average Bn (normalized) without the PMs = ", 
      np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal() * dphi * dtheta, axis=2))) / bnormalization)
print("Average Bn (normalized) with the PMs = ", 
      np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal() * dphi * dtheta, axis=2))) / bnormalization)

dipole_grid = pm_opt.dipole_grid
plt.figure()
ax = plt.axes(projection="3d")
colors = []
dipoles = dipoles.reshape(pm_opt.ndipoles, 3)
for i in range(pm_opt.ndipoles):
    colors.append(np.sqrt(dipoles[i, 0] ** 2 + dipoles[i, 1] ** 2 + dipoles[i, 2] ** 2))
sax = ax.scatter(dipole_grid[:, 0], dipole_grid[:, 1], dipole_grid[:, 2], c=colors)
plt.colorbar(sax)
plt.axis('off')
plt.grid(None)
plt.savefig('PMs_optimized_muse.png')

dipoles = np.ravel(dipoles)
print('Dipole field setup done')

make_plots = True 
if make_plots:
    # Make plot of ATA element values
    plt.figure()
    plt.hist(np.ravel(np.abs(pm_opt.ATA)), bins=np.logspace(-20, -2, 100), log=True)
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('histogram_ATA_values_muse.png')

    # Make plot of the relax-and-split convergence
    plt.figure()
    plt.semilogy(MwPGP_history)
    plt.grid(True)
    plt.savefig('objective_history_muse.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    plt.hist(abs(np.ravel(m_history[:, :, -1])) / np.ravel(np.outer(pm_opt.m_maxima, np.ones(3))), bins=np.linspace(0, 1, 30), log=True)
    plt.savefig('m_histogram_muse.png')
    print('Done optimizing the permanent magnets')

s = SurfaceRZFourier.from_focus(filename, range="full torus", nphi=nphi, ntheta=ntheta)
#sc_fieldline = SurfaceClassifier(s, h=0.1, p=2)
#sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)


def trace_fieldlines(bfield, label): 
    t1 = time.time()
    R0 = np.linspace(0.2, 0.4, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-15, comm=None,  # comm = comm
        #phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        phis=phis, stopping_criteria=[IterationStoppingCriterion(400000)])
    t2 = time.time()
    # print(fieldlines_phi_hits, np.shape(fieldlines_phi_hits))
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}_muse')
    plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}_muse.png', dpi=300)


n = 16
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2 * np.pi / s.nfp, n * 2)
zrange = (0, np.max(zs), n // 2)
t1 = time.time()
bsh = InterpolatedField(
    bs + b_dipole, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=s.stellsym
)
bsh.to_vtk('dipole_fields')
t2 = time.time()
#trace_fieldlines(bsh, 'bsh_PMs')
print('Done with Poincare plots with the permanent magnets, t = ', t2 - t1)

bs.set_points(s.gamma().reshape((-1, 3)))
b_dipole.set_points(s.gamma().reshape((-1, 3)))
# For plotting Bn on the full torus surface at the end with just the dipole fields
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "biot_savart_opt_muse", extra_data=pointData)
pointData = {"B_N": np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "only_pms_opt_muse", extra_data=pointData)
pointData = {"B_N": np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "pms_opt_muse", extra_data=pointData)
# plt.show()