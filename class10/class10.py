import numpy as np
import time

# ── Simulation parameters ─────────────────────────────────────────────────────
G   = 1.0    # gravitational constant
M   = 1.0    # particle mass
EPS = 0.1    # softening (avoids singularity at r=0)
DT  = 0.005  # timestep

def init_galaxy(N, seed=42):
    """Random disk of N particles with approximate circular velocities."""
    rng = np.random.default_rng(seed)
    r   = 2.0 * np.sqrt(rng.uniform(0.05, 1.0, N))
    phi = rng.uniform(0, 2*np.pi, N)
    pos = np.zeros((N, 3)); vel = np.zeros((N, 3))
    pos[:,0] = r*np.cos(phi); pos[:,1] = r*np.sin(phi)
    pos[:,2] = rng.normal(0, 0.05, N)
    v = np.sqrt(G*M*N / (r+EPS))
    vel[:,0] = -v*np.sin(phi); vel[:,1] = v*np.cos(phi)
    return pos.astype(np.float64), vel.astype(np.float64)

# Reference forces (used for correctness checks)
def forces_ref(pos):
    diff      = pos[np.newaxis,:,:] - pos[:,np.newaxis,:]   # (N,N,3)
    dist2     = np.sum(diff**2, axis=2) + EPS**2
    inv_dist3 = dist2**(-1.5)
    return G*M**2 * np.sum(diff * inv_dist3[:,:,np.newaxis], axis=1)

def check(label, F, F_ref, tol=1e-6):
    err = np.max(np.abs(np.asarray(F) - np.asarray(F_ref)))
    ok  = err < tol
    print(f"  {label}: max|err| = {err:.2e}  {'✓' if ok else f'✗ (tol={tol:.0e})'}")

N_TEST = 50
pos_t, vel_t = init_galaxy(N_TEST)
F_ref = forces_ref(pos_t)
print(f"Setup OK — N_TEST={N_TEST}")


def forces_naive(pos):
    N = pos.shape[0]
    F = np.zeros_like(pos)
    for i in range(N):
        diff     = pos - pos[i]                      # (N,3) — vector from i to all j
        dist2    = np.sum(diff**2, axis=1) + EPS**2  # (N,)
        inv3     = dist2**(-1.5)
        inv3[i]  = 0.0                               # zero self-force
        F[i]     = G * M**2 * np.dot(inv3, diff)
    return F

def step_naive(pos, vel):
    F = forces_naive(pos)
    vel += F / M * DT
    pos += vel * DT
    return pos, vel

check('naive', forces_naive(pos_t), F_ref)

def forces_vectorized(pos):
    diff      = pos[np.newaxis,:,:] - pos[:,np.newaxis,:]  # (N,N,3)
    dist2     = np.sum(diff**2, axis=2) + EPS**2            # (N,N)
    inv_dist3 = dist2**(-1.5)                                # (N,N)
    # diff[i,i] = 0 → self-force = 0 automatically
    return G * M**2 * np.sum(diff * inv_dist3[:,:,np.newaxis], axis=1)  # (N,3)

def step_vectorized(pos, vel):
    F = forces_vectorized(pos)
    vel += F / M * DT
    pos += vel * DT
    return pos, vel

check('vectorized', forces_vectorized(pos_t), F_ref)

from numba import njit, prange

@njit(parallel=True, fastmath=True)
def forces_prange(pos, G, M, EPS):
    N    = pos.shape[0]
    F    = np.zeros_like(pos)
    Gm2  = G * M * M
    eps2 = EPS * EPS
    for i in prange(N):              # ← parallel over particles
        fx = fy = fz = 0.0
        xi = pos[i,0]; yi = pos[i,1]; zi = pos[i,2]
        for j in range(N):           # ← sequential: all j per thread
            dx = pos[j,0]-xi; dy = pos[j,1]-yi; dz = pos[j,2]-zi
            d3 = (dx*dx + dy*dy + dz*dz + eps2)**(-1.5)
            fx += Gm2*dx*d3; fy += Gm2*dy*d3; fz += Gm2*dz*d3
        F[i,0] = fx; F[i,1] = fy; F[i,2] = fz
    return F

def step_prange(pos, vel):
    F = forces_prange(pos, G, M, EPS)
    vel += F / M * DT
    pos += vel * DT
    return pos, vel

forces_prange(pos_t, G, M, EPS)   # warmup (JIT compile)
check('prange', forces_prange(pos_t, G, M, EPS), F_ref)

import dask

def _chunk_forces(pos_chunk, pos_all):
    diff      = pos_all[np.newaxis,:,:] - pos_chunk[:,np.newaxis,:]  # (k,N,3)
    dist2     = np.sum(diff**2, axis=2) + EPS**2
    inv_dist3 = dist2**(-1.5)
    return G*M**2 * np.sum(diff * inv_dist3[:,:,np.newaxis], axis=1)

def forces_dask(pos, n_chunks=4):
    chunks    = np.array_split(pos, n_chunks, axis=0)
    delayed_f = [dask.delayed(_chunk_forces)(c, pos) for c in chunks]
    return np.vstack(dask.compute(*delayed_f, scheduler='threads'))

def step_dask(pos, vel, n_chunks=4):
    F = forces_dask(pos, n_chunks)
    vel += F / M * DT
    pos += vel * DT
    return pos, vel

check('dask', forces_dask(pos_t, n_chunks=4), F_ref)


import cupy as cp

def forces_cupy(pos_cp):
    diff      = pos_cp[cp.newaxis,:,:] - pos_cp[:,cp.newaxis,:]  # on GPU
    dist2     = cp.sum(diff**2, axis=2) + EPS**2
    inv_dist3 = dist2**(-1.5)
    return G*M**2 * cp.sum(diff * inv_dist3[:,:,cp.newaxis], axis=1)

def step_cupy(pos_cp, vel_cp):
    F       = forces_cupy(pos_cp)
    vel_cp += F / M * DT
    pos_cp += vel_cp * DT
    return pos_cp, vel_cp

pos_cp = cp.array(pos_t)
F_cp   = forces_cupy(pos_cp)
check('cupy', cp.asnumpy(F_cp), F_ref)

def step_all(pos, vel, method):
    if method == 'naive':
        return step_naive(pos, vel)
    elif method == 'vectorized':
        return step_vectorized(pos, vel)
    elif method == 'prange':
        return step_prange(pos, vel)
    elif method == 'dask':
        return step_dask(pos, vel)
    elif method == 'cupy':
        pos_cp = cp.array(pos); vel_cp = cp.array(vel)
        pos_cp, vel_cp = step_cupy(pos_cp, vel_cp)
        return cp.asnumpy(pos_cp), cp.asnumpy(vel_cp)
    else:
        raise ValueError(f"Unknown method: {method}")


N_VALUES = [100, 200, 500, 1000, 2000, 3000]
T = 10
results = {}   # paradigm → list of ms/step

METHODS = ['naive', 'vectorized', 'prange', 'dask', 'cupy']
for method in METHODS:
    results[method] = []
    for N in N_VALUES:
        pos, vel = init_galaxy(N)
        t0 = time.perf_counter()
        for _ in range(T):
            pos, vel = step_all(pos, vel, method=method)   # ← swap for other paradigms
        ms_per_step = (time.perf_counter() - t0) / T * 1000
        results[method].append(ms_per_step)
        print(f"N={N}  {ms_per_step:.3f} ms/step")


# Generated by Claude Opus 4.6 on 2026-04-20
import pandas as pd, socket, matplotlib.pyplot as plt

df = pd.DataFrame(results)
df.insert(0, 'N', N_VALUES)
csv_name = f'race_{socket.gethostname()}.csv'
df.to_csv(csv_name, index=False)
print(f"\nResults saved to {csv_name}")
print(df.to_string(index=False))

plt.figure(figsize=(10, 6))
for method in METHODS:
    plt.plot(N_VALUES, df[method], marker='o', label=method)
plt.xlabel('N (particles)')
plt.ylabel('Time per step (ms)')
plt.yscale('log')
plt.title('N-body simulation — method comparison')
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig(f'race_{socket.gethostname()}.png', dpi=150)
plt.show()
print("Graph saved.")