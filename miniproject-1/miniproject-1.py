import matplotlib.pyplot as plt
import time
import numpy as np
from numba import jit

#-----------------------COFNIG---------------------------------------------

YMIN, YMAX = -1.5, 1.5
XMIN, XMAX = -2.0, 1.0
MAX_ITER = 100
WIDTH, HEIGHT = 1024, 1024

#-----------------Utilities (visualization + CSV Exports)------------------
def visualize(title, mb_set, xmin, xmax, ymin, ymax):
    plt.figure(figsize=(8,8))
    plt.imshow(mb_set, extent=[xmin, xmax, ymin, ymax], cmap='hot', origin='lower')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title(title)
    plt.colorbar(label='Iterations')
    plt.show()

def export_to_csv(filename, data):
    if isinstance(data, list) and isinstance(data[0], tuple):
        # Timing results
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Implementation', 'Time (seconds)'])
            writer.writerows(data)
    else:
        # Mandelbrot set array
        np.savetxt(filename, data, delimiter=',', fmt='%d')

#---------------- Different ways to calculate the MB ----------------------
#                                                                          |
#--------------------------------------------------------------------------
def native_python_implementation(xmin, xmax, ymin, ymax, width, height, max_iter):
    mb_set = []

    for y in range(height):
        mb_row = []
        #Y coordinates  space the points with the given space and resolution
        p_y = ymin + (y / (height - 1)) * (ymax - ymin)
        
        for x in range(width):
            # X coordinates space the points with the given space and resolution
            p_x = xmin + (x / (width - 1)) * (xmax - xmin)
            c = p_x + 1j * p_y
            
            z = 0
            n = 0
            # Match NumPy behavior: record when escape happens
            while abs(z) <= 2 and n < max_iter:
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

@jit(nopython=True)
def numba_implementation(xmin, xmax, ymin, ymax, width, height, max_iter):
    mb_set = []
    
    for y in range(height):
        mb_row = []
        # Fix: Map y from top to bottom correctly
        p_y = ymin + (y / (height - 1)) * (ymax - ymin)
        
        for x in range(width):
            p_x = xmin + (x / (width - 1)) * (xmax - xmin)
            c = p_x + 1j * p_y
            
            z = 0
            n = 0
            # Match NumPy behavior: record when escape happens
            while abs(z) <= 2 and n < max_iter:
                z = z ** 2 + c
                n += 1
            
            mb_row.append(n)
        mb_set.append(mb_row)
    
    return mb_set

#--------------------------------Execute and time------------------------------------------------
results = []
native_start = time.perf_counter()
native_mb_set = native_python_implementation(XMIN, XMAX, YMIN, YMAX, WIDTH, HEIGHT, MAX_ITER)
native_end = time.perf_counter()
results.append(("Native Python", native_end - native_start))

numpy_start = time.perf_counter()
numpy_mb_set = numpy_implementation(XMIN, XMAX, YMIN, YMAX, WIDTH, HEIGHT, MAX_ITER)
numpy_end = time.perf_counter()
results.append(("NumPy Implementation", numpy_end - numpy_start))
# Warm up JIT
_ = numba_implementation(1, 1, 1, 1, 2, 2, 1)

numba_start = time.perf_counter()
numba_mb_set = numba_implementation(XMIN, XMAX, YMIN, YMAX, WIDTH, HEIGHT, MAX_ITER)
numba_end = time.perf_counter()
results.append(("Numba Implementation", numba_end - numba_start))

#uncoment if you want to see the nce visualization of the sets :)
#visualize("Native Implementation", native_mb_set, XMIN, XMAX, YMIN, YMAX)
#visualize("NumPy Implementation", numpy_mb_set, XMIN, XMAX, YMIN, YMAX)
#visualize("Numba Implementation", numba_mb_set, XMIN, XMAX, YMIN, YMAX)

# Print the results
print("\nResults:")
for name, time_taken in results:
    print(60*"=")
    print(f"{name}: {time_taken:.6f} seconds")

save_csv = input("Do you want to save the results to CSV? (y/n): ")
if save_csv.lower() == 'y': 
    export_to_csv("results/native_mb_set.csv", native_mb_set)
    export_to_csv("results/numpy_mb_set.csv", numpy_mb_set)
    export_to_csv("results/numba_mb_set.csv", numba_mb_set)
    export_to_csv("results/timing_results.csv", results)
    print("Results saved to CSV files.")

