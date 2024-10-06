import pyvista as pv 

import meshio
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, io, nls, mesh, log
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from dolfinx.mesh import create_box, CellType
from ufl import (
    as_matrix,
    dot,
    cos,
    sin,
    SpatialCoordinate,
    Identity, 
    inner,
    grad,  
    diff,
    inv,
    ln,
    tr,
    det, 
    sym,
    variable, 
    derivative,
    TestFunction,
    TrialFunction,  
    Constant,  
    Measure,   
)   

from basix.ufl import element, mixed_element

from petsc4py import PETSc  

dtype = PETSc.ScalarType

def main(): 

    # load fwd solution geometry 
    # msh_vtu = meshio.read("fwd_cube_warped.vtu")
    # msh_vtu.write("mesh.xdmf")  

    # with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    #     msh = xdmf.read_mesh(name="Grid") 

    # load fwd solution geometry 
    msh = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1, 1, 1]], [1, 1, 1], mesh.CellType.tetrahedron)

    # mixed function space, 2nd order for displacement and 1st order for pressure  
    # Following fenics 0.8.0 tutorial on mixed spaces: https://docs.fenicsproject.org/dolfinx/v0.8.0/python/demos/demo_mixed-poisson.html
    V = element(family="Lagrange", cell=msh.basix_cell(), degree=2, shape=(3,)) # displacement (x, y, z) -> (dx, dy, dz) : shape=(3,) is required
    P = element(family="Lagrange", cell=msh.basix_cell(), degree=1) # pressure (x, y, z) -> (p) : shape=None is default
    U = fem.functionspace(msh, mixed_element([V, P])) 

    # Following Arnaud's code for the mixed formulation: https://github.com/Ziemnono/fenics-inverseFEM/blob/main/examples/Sagging_block/incompressible-neo-hooke.py
    rho = dolfinx.fem.Constant(msh, 1000.0)   
    mu = dolfinx.fem.Constant(msh, 2.0e4)
    lmbda = dolfinx.fem.Constant(msh, 8.0e5)  

    # body force, gravity in this case 
    g = dolfinx.fem.Constant(msh, 9.81)    

    # split the mixed function space 
    u_p_ = fem.Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)

    u_t, p_t = ufl.split(v)
    u_, p_ = ufl.split(u_p_) 

    # kinematics with deformation gradient, right Green-Cauchy and Lagrangian strain tensor  
    I = Identity(3)
    F = variable(inv(grad(u_) + I)) # for the unloading case, we use the deformed configuration as the reference configuration
    C = variable(F.T * F)
    E = variable(0.5 * (C - I)) 

    # invariariants and Jacobian
    I_C = tr(C)
    II_C = (1.0 / 2.0) * (tr(C) ** 2 - tr(C * C))
    III_C = det(C)
    Jdet = III_C ** (1.0 / 2.0)  

    # stored strain energy density (incompressible Neo-Hooke model)
    psi = (mu / 2.0) * (I_C - 2) - mu * ln(Jdet) + p_ * ln(Jdet) - (1.0 / (2.0 * lmbda)) * p_ ** 2 

    # second Piola-Kirchoff conjugate with incremental Green-Lagrange
    S = 2.0*diff(psi, C)
    sigma = Jdet**(-1.0)*F*S*F.T 
    dx = Measure("dx", msh)
    G = inner(sigma, sym(grad(u_t)))*dx + inner((ln(Jdet) - p_/lmbda), p_t)*dx - inner(g*rho, -u_t[2])*dx
    J = derivative(G, u_p_, u) 

    # define the ROI (Region Of Interest) corresponding to the bottom of the cube 
    # and apply null Dirichlet BCs on the bottom 
    # following: https://docs.fenicsproject.org/dolfinx/v0.8.0/python/demos/demo_elasticity.html
    tdim = msh.topology.dim 
    fdim = tdim - 1   
    msh.topology.create_connectivity(tdim, fdim)  


    facets = dolfinx.mesh.locate_entities_boundary(
        msh, dim=fdim, marker=lambda x: np.isclose(x[0], 0.0)
    )  

    # Had some difficulty applying Dirichlet bc on a vector sub-space of a mixed element, e.g. U.sub(0), displacement of mixed element U. 
    # Followed solution by dokken on fenics project discourse: https://fenicsproject.discourse.group/t/how-to-impose-a-dirichlet-bc-on-a-vector-sub-space/8528/2
    U1, _ = U.sub(0).collapse()
    u_D = fem.Function(U1)
    u_D.x.array[:] = 0

    boundary_dofs = fem.locate_dofs_topological((U.sub(0), U1), fdim, facets)
    bcs = fem.dirichletbc(u_D, boundary_dofs, U.sub(0)) 

    # Solve nonlinear problem  
    problem = fem.petsc.NonlinearProblem(G, # PDE residual 
                                        u_p_, # The 'unkown' function
                                        [bcs], # Dirichlet boundary conditions  
                                        J=J # Jacobian of the residual 
                                        )

    solver = dolfinx.nls.petsc.NewtonSolver(msh.comm, problem) 

    # Set Newton solver options
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"  
    solver.linear_solver = "mumps"
    solver.maximum_iterations = 10  
    solver.report = True 

    log.set_log_level(log.LogLevel.INFO) 
    num_its, converged = solver.solve(u_p_)
    print(f"Number of iterations: {num_its}, Converged: {converged}")  

    # We start by creating a unit square mesh and interpolating a function
    # into a first order Lagrange space
    dim = msh.topology.dim
    print(f"Mesh topology dimension d={dim}.")

    degree = 1
    shape = (dim,)
    U_plot = fem.functionspace(msh, ("Lagrange", degree, shape))  
    u_plot = fem.Function(U_plot, dtype=np.float64)

    # Interpolate the solution to the function space 
    u_plot.interpolate(u_p_.sub(0)) 

    # As we want to visualize the function u, we have to create a grid to
    # attached the dof values to We do this by creating a topology and
    # geometry based on the function space V
    cells, types, x = dolfinx.plot.vtk_mesh(U_plot._mesh)
    grid = pv.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u_plot.x.array.reshape(-1, 3) 


    # We set the function "u" as the active scalar for the mesh, and warp
    # the mesh in z-direction by its values
    grid.set_active_scalars("u")
    warped = grid.warp_by_vector()

    # We create a plotting window consisting of to plots, one of the scalar
    # values, and one where the mesh is warped by these values
    subplotter = pv.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text("Undeformed Cube", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    subplotter.subplot(0, 1)
    subplotter.add_text("Deformed Cube", position="upper_edge", font_size=14, color="black")
    subplotter.add_mesh(warped, show_edges=True) 
    subplotter.add_axes()
    subplotter.show() 

    # Export to vtu file for paraview 
    grid.save("inv_cube.vtu") 
    warped.save("inv_cube_warped.vtu") 

if __name__ == "__main__": 
    # run with: mpirun -n 8 python hyperelastic_script.py
    main()