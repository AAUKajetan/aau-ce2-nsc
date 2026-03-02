import numpy as np
import matplotlib.pyplot as plt

# Parameters
xmin, xmax = -2.5, 2.5
ymin, ymax = -2.5, 2.5
width, height = 1024, 1024
max_iter = 300

# Create grid of complex numbers
x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)
X, Y = np.meshgrid(x, y)
# C is a complex grid representing points in the complex plane
c = X + 1j * Y
# c[0, 0] = -2.0 - 1.5j      # Top-left corner
# c[0, 1023] = 1.0 - 1.5j    # Top-right corner
# c[511, 511] ≈ -0.5 + 0j    # Near center

# Initialize z and output array
z = np.zeros_like(c)  # Start with z=0 for all points
output = np.zeros(c.shape, dtype=int)  # Will store iteration count when each point escapes

# Iterate the Mandelbrot formula: z = z^2 + c
for n in range(max_iter):
    mask = np.abs(z) <= 2  # Points still bounded (haven't escaped yet)
    z[mask] = z[mask] ** 2 + c[mask]  # Apply formula only to non-escaped points
    output[mask & (np.abs(z) > 2) & (output == 0)] = n  # Record iteration count when point first escapes
# Points that never escaped (stayed bounded) get max_iter value - these form the Mandelbrot set
output[output == 0] = max_iter
print("Mandelbrot set calculated.")

# Visualize
plt.figure(figsize=(8,8))
plt.imshow(output, extent=[xmin, xmax, ymin, ymax], cmap='hot')
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Mandelbrot Set')
plt.colorbar(label='Iterations')
plt.show()

