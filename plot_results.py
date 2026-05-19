import numpy as np
import matplotlib.pyplot as plt
import json # ADD THIS LINE
# 1. Load the data the AI saved
# test.dat contains columns: x, y, u (x-velocity), v (y-velocity), p (pressure)
data = np.loadtxt("test.dat")
x = data[:, 0]
y = data[:, 1]
u = data[:, 2]
v = data[:, 3]
p = data[:, 4]

# 2. Calculate overall speed (Velocity Magnitude) using Pythagorean theorem
speed = np.sqrt(u**2 + v**2)

# 3. Set up the figure side-by-side plots
plt.figure(figsize=(14, 4))

# Plot 1: Blood Speed (Velocity Heatmap)
plt.subplot(1, 2, 1)
# tricontourf is perfect for plotting the scattered points our AI generated
plt.tricontourf(x, y, speed, levels=100, cmap='jet') 
plt.colorbar(label='Speed')
plt.title('Blood Flow Speed (Velocity)')
plt.xlabel('Vessel Length')
plt.ylabel('Vessel Height')

# Plot 2: Pressure Drop
plt.subplot(1, 2, 2)
plt.tricontourf(x, y, p, levels=100, cmap='coolwarm')
plt.colorbar(label='Pressure')
plt.title('Pressure Field')
plt.xlabel('Vessel Length')
plt.ylabel('Vessel Height')

plt.tight_layout()

# --- NEW EXPORT CODE ---
# 1. Save the heatmap as an image for the website
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
print("Saved heatmap.png")

# 2. Extract Clinical Metrics
max_pressure = float(np.max(p)) # Convert to standard float for JSON
min_pressure = float(np.min(p))
pressure_drop = max_pressure - min_pressure
max_speed = float(np.max(speed))

# 3. Save the metrics as a JSON file for JavaScript to read
metrics = {
    "pressure_drop": round(pressure_drop, 4),
    "peak_velocity": round(max_speed, 4),
    "stenosis_level": "50% Blockage" # We can change this later for different models
}

with open('data.json', 'w') as f:
    json.dump(metrics, f)
print("Saved data.json")