import numpy as np
import matplotlib.pyplot as plt

# Parameters explanation:
# xmin, xmax: Range of real axis (horizontal) in the complex plane
# ymin, ymax: Range of imaginary axis (vertical) in the complex plane
# width, height: Resolution of the output image (number of pixels)
# max_iter: Maximum number of iterations to test for divergence

# Parameters
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width, height = 1024, 1024
max_iter = 100

# Create grid of complex numbers
x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)
X, Y = np.meshgrid(x, y)
c = X + 1j * Y

# Initialize z and output array
z = np.zeros_like(c)
output = np.zeros(c.shape, dtype=int)

for n in range(max_iter):
    mask = np.abs(z) <= 2
    z[mask] = z[mask] ** 2 + c[mask]
    output[mask & (np.abs(z) > 2) & (output == 0)] = n
output[output == 0] = max_iter

# Visualize
plt.figure(figsize=(8,8))
plt.imshow(output, extent=[xmin, xmax, ymin, ymax], cmap='hot')
plt.xlabel('Re')
plt.ylabel('Im')
plt.title('Mandelbrot Set')
plt.colorbar(label='Iterations')
plt.show()

# Plot the square root function for real and complex results
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 400)
y_real = np.zeros_like(x)
y_imag = np.zeros_like(x)

for i, val in enumerate(x):
    sqrt_val = np.sqrt(val + 0j)  # Always returns a complex number
    y_real[i] = sqrt_val.real
    y_imag[i] = sqrt_val.imag

plt.figure(figsize=(8, 4))
plt.plot(x, y_real, label='Real part of sqrt(x)')
plt.plot(x, y_imag, label='Imaginary part of sqrt(x)', linestyle='--')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('sqrt(x)')
plt.title('Square Root of x (Real and Imaginary Parts)')
plt.legend()
plt.grid(True)
plt.show()
print("Square root of -1:", np.sqrt(-1 + 0j))
