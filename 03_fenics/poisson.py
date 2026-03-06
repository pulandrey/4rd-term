import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, io
import ufl
from dolfinx.fem.petsc import LinearProblem

# параметры
T = 5.0
num_steps = 50
dt = T / num_steps
omega = 2*np.pi/2

# сетка
nx = ny = 100
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)

# пространство функций
V = fem.functionspace(domain, ("Lagrange", 1))

# координаты сердца
cx, cy = 0.5, 0.5
scale = 0.25  # масштаб сердца относительно квадрата

# Trial и Test
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# коэффициент теплопроводности
k = 0.005
a = u*v*ufl.dx + dt*k*ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx

# исходная температура
u_n = fem.Function(V)
u_n.x.array[:] = 0.0

# файл для XDMF
xdmf = io.XDMFFile(MPI.COMM_WORLD, "heart_pulse_visible.xdmf", "w")
xdmf.write_mesh(domain)
uh = fem.Function(V)

# координаты всех узлов
coords = V.tabulate_dof_coordinates()

# создаём маску сердца
mask_vals = np.zeros(len(coords))
for j, xy in enumerate(coords):
    X = (xy[0]-cx)/scale
    Y = (xy[1]-cy)/scale
    # формула сердца
    if (X**2 + Y**2 -1)**3 - X**2 * Y**3 <= 0:
        mask_vals[j] = 1.0
mask = fem.Function(V)
mask.x.array[:] = mask_vals

for i in range(num_steps):
    t = dt*i
    pulse = 5.0 + 5.0*np.cos(omega*t)  # сильная пульсация
    f = fem.Constant(domain, ScalarType(pulse))
    
    L = (u_n + dt*f*mask)*v*ufl.dx
    problem = LinearProblem(a, L, bcs=[], petsc_options_prefix="heat")
    uh = problem.solve()

    xdmf.write_function(uh, t)
    u_n.x.array[:] = uh.x.array

xdmf.close()
print("Готово, файл heart_pulse_visible.xdmf создан.")