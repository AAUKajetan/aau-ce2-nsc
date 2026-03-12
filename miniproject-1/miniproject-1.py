import matplotlib.pyplot as plt
import time
import numpy as np

#-----------------------COFNIG---------------------------------------------

YMIN, YMAX = -1.5, 1.5
XMIN, XMAX = -2.0, 1.0
MAX_ITER = 100
WIDTH, HEIGHT = 1024, 1024

#-----------------Utilities (visualization + CSV Exports)------------------
def visualize(title, mb_set, xmin, xmax, ymin, ymax):
    """Visualize Mandelbrot set"""
    plt.figure(figsize=(8,8))
    plt.imshow(mb_set, extent=[xmin, xmax, ymin, ymax], cmap='hot', origin='lower')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title(title)
    plt.colorbar(label='Iterations')
    plt.show()

#---------------- Different ways to calculate the MB ----------------------
#                                                                          |
#--------------------------------------------------------------------------
def native_python_implementation(xmin, xmax, ymin, ymax, width, height, max_iter):
    grid = []
    for y in range(height):
        row = []
        # Fix: Start from ymax and go down to ymin
        p_y = ymax - (y / height) * (ymax - ymin)
        for x in range(width):
            p_x = xmin + (x / width) * (xmax - xmin)
            row.append((p_x, p_y))        
        grid.append(row)
    
    c_grid = []
    for row in grid:
        c_row = []
        for point in row:
            (x, y) = point
            c = x + 1J * y
            c_row.append(c)
        c_grid.append(c_row)

    mb_set = []
    for c_row in c_grid:
        mb_row = []
        for c in c_row:
            z = 0
            n = 0
            while (abs(z) <= 2 and n < max_iter):
                z = z ** 2 + c
                n += 1
            mb_row.append(n)
        mb_set.append(mb_row)
    return mb_set

def numpy_implementation(xmin, xmax, ymin, ymax, width, height, max_iter):
    #set up the grid:
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    # C is a complex grid representing points in the complex plane
    c = X + 1j * Y

    #innitialize the z = 0 for all
    z = np.zeros_like(c)
    #prep the output array
    output = np.zeros(c.shape, dtype=int)

    for n in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]
        output[mask] = n + 1  # min iterations = 1

    return output

#--------------------------------Execute and time------------------------------------------------
print(60*"=")
native_start = time.perf_counter()
native_mb_set = native_python_implementation(XMIN, XMAX, YMIN, YMAX, WIDTH, HEIGHT, MAX_ITER)
native_end = time.perf_counter()
print(f"Native Python: {native_end - native_start:.6f} seconds")
print(60*"=")

numpy_start = time.perf_counter()
numpy_mb_set = numpy_implementation(XMIN, XMAX, YMIN, YMAX, WIDTH, HEIGHT, MAX_ITER)
numpy_end = time.perf_counter()
print(f"NumPy Implementation: {numpy_end - numpy_start:.6f} seconds")
print(60*"=")


visualize("Native Implementation", native_mb_set, XMIN, XMAX, YMIN, YMAX)
visualize("NumPy Implementation", numpy_mb_set, XMIN, XMAX, YMIN, YMAX)