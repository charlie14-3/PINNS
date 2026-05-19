import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the dimensions of the blood vessel
length = 2.0  # Length of the artery segment
height = 0.5  # Diameter/height of the artery

# Create the healthy artery (a simple 2D rectangle)
# Bottom-left corner is [0, 0], top-right is [length, height]
artery = dde.geometry.Rectangle([0, 0], [length, height])

# 2. Define the blockage (Stenosis)
bump_center = [1.0, 0.0]  # Place it halfway down the artery on the bottom wall
bump_radius = 0.2         # Size of the blockage

# Create a circle to represent the plaque
plaque = dde.geometry.Disk(bump_center, bump_radius)

# 3. Create the final shape by subtracting the plaque from the artery
# This cuts the bump out of our rectangular flow domain
stenosis_geom = dde.geometry.CSGDifference(artery, plaque)

# 4. Generate some random points inside the shape to visualize it
# The AI will eventually use these points to learn the physics
points = stenosis_geom.random_points(2000)

# Plot the points to see the shape of our narrowed artery
plt.figure(figsize=(10, 3))
plt.scatter(points[:, 0], points[:, 1], s=1, c='darkred')
plt.title("2D Arterial Stenosis Geometry")
plt.xlabel("Length (x)")
plt.ylabel("Height (y)")
plt.xlim(0, length)
plt.ylim(0, height)
plt.show()


# 5. Define the Physics (Navier-Stokes Equations)
def navier_stokes(x, y):
    # x contains our coordinates: x[:, 0:1] is X, x[:, 1:2] is Y
    # y contains the AI's predictions: y[:, 0:1] is u, y[:, 1:2] is v, y[:, 2:3] is pressure
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    
    # Calculate gradients (how velocity and pressure change over space)
    # du_x means "change in u with respect to x"
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    
    # Calculate second derivatives (needed for the viscosity part)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    
    # Blood properties (simplified kinematic viscosity)
    nu = 0.01 
    
    # The Equations (translating the math formulas into code)
    continuity = du_x + dv_y
    x_momentum = u * du_x + v * du_y + dp_x - nu * (du_xx + du_yy)
    y_momentum = u * dv_x + v * dv_y + dp_y - nu * (dv_xx + dv_yy)
    
    # The AI will try to make all three of these equations equal exactly zero
    return [continuity, x_momentum, y_momentum]

# 6. Define Boundary Locations
def boundary_inlet(x, on_boundary):
    # Left edge (x = 0)
    return on_boundary and np.isclose(x[0], 0)

def boundary_outlet(x, on_boundary):
    # Right edge (x = length)
    return on_boundary and np.isclose(x[0], length)

def boundary_wall(x, on_boundary):
    # Any boundary that is NOT the inlet or outlet is a wall (including the bump)
    return on_boundary and not (np.isclose(x[0], 0) or np.isclose(x[0], length))

# 7. Define the Inlet Velocity Profile
def inlet_velocity(x):
    # Parabolic flow: fastest in the middle (y = 0.25), zero at edges (y=0, y=0.5)
    y = x[:, 1:2]
    v_max = 1.0 # Maximum velocity
    return 4.0 * v_max * (y / height) * (1.0 - y / height)

# 8. Apply the Conditions to the AI
# u is horizontal velocity (component 0), v is vertical velocity (component 1), p is pressure (component 2)

# Inlet: Horizontal velocity follows our parabola, vertical velocity is 0
inlet_u = dde.icbc.DirichletBC(stenosis_geom, inlet_velocity, boundary_inlet, component=0)
inlet_v = dde.icbc.DirichletBC(stenosis_geom, lambda x: 0, boundary_inlet, component=1)

# Walls: No-slip condition (both velocities are 0)
wall_u = dde.icbc.DirichletBC(stenosis_geom, lambda x: 0, boundary_wall, component=0)
wall_v = dde.icbc.DirichletBC(stenosis_geom, lambda x: 0, boundary_wall, component=1)

# Outlet: Pressure is 0
outlet_p = dde.icbc.DirichletBC(stenosis_geom, lambda x: 0, boundary_outlet, component=2)

# Group all boundary conditions together
bcs = [inlet_u, inlet_v, wall_u, wall_v, outlet_p]

# 9. Package the Data (Geometry, Physics, Boundaries)
# We sample points inside the domain and on the boundaries to train the network
data = dde.data.PDE(
    stenosis_geom,
    navier_stokes,
    bcs,
    num_domain=3000,   # Points inside the artery
    num_boundary=1000, # Points on the walls
    num_test=1000      # Points to test accuracy
)

# 10. Build the Neural Network
# 2 inputs (x, y), 3 outputs (u, v, p), 4 hidden layers with 50 neurons each
net = dde.nn.FNN([2] + [50] * 4 + [3], "tanh", "Glorot uniform")

# 11. Compile and Train the Model
model = dde.Model(data, net)
model.compile("adam", lr=1e-3) # Adam optimizer is standard for AI

print("Starting to train the AI... This might take a few minutes!")

# Train for 5000 iterations (a good starting point for a laptop)
losshistory, train_state = model.train(iterations=5000)

# 12. Save and visualize the final results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)