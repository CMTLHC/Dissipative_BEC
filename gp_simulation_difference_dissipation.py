import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# --- Grid and operators ---
def initialize_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y, dx, dy

def laplacian(psi, dx, dy):
    lap = np.zeros_like(psi, dtype=complex)
    lap[1:-1,1:-1] = (
        (psi[2:,1:-1] - 2*psi[1:-1,1:-1] + psi[:-2,1:-1]) / dx**2 +
        (psi[1:-1,2:] - 2*psi[1:-1,1:-1] + psi[1:-1,:-2]) / dy**2
    )
    return lap

# Finite-difference gradients for L_z operator
def gradient_x(psi, dx):
    grad = np.zeros_like(psi, dtype=complex)
    grad[1:-1,:] = (psi[2:,:] - psi[:-2,:]) / (2*dx)
    return grad

def gradient_y(psi, dy):
    grad = np.zeros_like(psi, dtype=complex)
    grad[:,1:-1] = (psi[:,2:] - psi[:,:-2]) / (2*dy)
    return grad

# --- Potentials and parameters ---
def potential_harmonic(X, Y, omega=1.0):
    return 0.25 * (X**2 + Y**2)

def potential_anisotropy(X, Y, epsx, epsy):
    return 0.25 * ((1+epsx) * X**2 +(1+epsy) * Y**2)

# --- Imaginary time evolution for ground state ---
def imaginary_time_step(psi, V, g, chem_potential, dx, dy, d_tau):
    # ∂ψ/∂τ = 1/2 ∇²ψ - (V + g|ψ|²)ψ
    lap = laplacian(psi, dx, dy)
    psi_new = psi + d_tau * (lap - (V - chem_potential + g * np.abs(psi)**2) * psi)
    # Renormalize
    norm = np.sqrt(np.sum(np.abs(psi_new)**2) * dx * dy)
    return psi_new / norm

# --- Real (unitary) time evolution via RK4 ---
def compute_rhs(psi, V, g, chem_potential, dx, dy, Omega, gamma):
    lap = laplacian(psi, dx, dy)
    nonlinear = g * np.abs(psi)**2 * psi
    psi_x = gradient_x(psi, dx)
    psi_y = gradient_y(psi, dy)
    Lz_full = -1j * (X * psi_y - Y * psi_x)
    # restrict to radius R <= R_max
    R2 = X**2 + Y**2
    mask = (R2 <= 12**2)
    Lz_term = Lz_full * mask
    return (-lap + (V - chem_potential) * psi + nonlinear - Omega * Lz_term)/(1j-gamma)

def rk4_step(psi, V, g, chem_potential, dx, dy, dt, Omega, gamma):
    k1 = compute_rhs(psi, V, g, chem_potential, dx, dy, Omega, gamma)
    k2 = compute_rhs(psi + 0.5*dt*k1, V, g, chem_potential, dx, dy, Omega, gamma)
    k3 = compute_rhs(psi + 0.5*dt*k2, V, g, chem_potential, dx, dy, Omega, gamma)
    k4 = compute_rhs(psi + dt*k3, V, g, chem_potential, dx, dy, Omega, gamma)
    return psi + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# --- Simulation parameters ---
Nx, Ny = 256, 256   # grid points
Lx, Ly = 30.0, 30.0 # domain size

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
dt = 0.004          # real time step
epsx = 0.09
epsy = 0.03
Omega = 0.75
chem_potential = 100
gamma = 0.03
R_zoom = 10.0

g1 = 0
g2 = -3j             # interaction strength
omega = 1.0         # trap frequency

# Initialize grid, potential, and initial guess (Gaussian)
X, Y, dx, dy = initialize_grid(Nx, Ny, Lx, Ly)
V = potential_harmonic(X, Y, omega)
Va = potential_anisotropy(X, Y, epsx, epsy)
psi1 = np.exp(-(X**2 + Y**2) / 2.0)
psi1 /= np.sqrt(np.sum(np.abs(psi1)**2) * dx * dy)
psi2 = np.exp(-(X**2 + Y**2) / 2.0)
psi2 /= np.sqrt(np.sum(np.abs(psi1)**2) * dx * dy)

# --- Prepare figure and animation containers ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
den = np.abs(psi1)**2
im1 = ax1.imshow(den, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='viridis', vmin=0, vmax=0.03)
cbar1 = fig.colorbar(im1, ax=ax1)
cbar1.set_label('|ψ|²')
ax1.set_title('Evolution of Density')
# restrict view to local region
ax1.set_xlim(-R_zoom, R_zoom)
ax1.set_ylim(-R_zoom, R_zoom)
ax1.set_xlabel('x'); ax1.set_ylabel('y')
# initial phase plot
phase = np.angle(psi1)
im2 = ax2.imshow(phase, extent=[-Lx/2,Lx/2,-Ly/2,Ly/2], cmap='twilight', vmin=-np.pi, vmax=np.pi)
cbar2 = fig.colorbar(im2, ax=ax2); cbar2.set_label('Phase')
ax2.set_title('Phase')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
# restrict view to same region

# Collect frames: first imaginary relaxation then real-time
frames, norms, times = [], [], []

# --- Imaginary time relaxation ---
d_tau = 0.001      # imaginary time step
i_steps = 5000     # number of iterations
N_im = 4000
for i in range(i_steps):
    psi1 = imaginary_time_step(psi1, V, g1, chem_potential, dx, dy, d_tau)
    if (i+1) % 1000 == 0:
        print(f"Imag time step {i+1}/{i_steps}")
    if i % N_im == 0:
        plt.figure()
        plt.imshow(np.abs(psi1)**2, extent=[-R_zoom, R_zoom, -R_zoom, R_zoom])
        plt.title('Ground State Density')
        plt.colorbar()
        plt.show()

for i in range(i_steps):
    psi2 = imaginary_time_step(psi2, V, g1, chem_potential, dx, dy, d_tau)
    if (i+1) % 1000 == 0:
        print(f"Imag time step2 {i+1}/{i_steps}")

# Ground state reached: |psi|^2 is stationary
#plt.figure()
#plt.imshow(np.abs(psi)**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
#plt.title('Ground State Density')
#plt.colorbar()
#plt.show()

# --- (Optional) Real-time evolution starting from ground state ---
t_max = 300
t_steps = int(t_max/dt)
N_rt = 400
for step in range(t_steps):
    psi1 = rk4_step(psi1, Va, g2, chem_potential, dx, dy, dt, Omega, gamma)
    psi2 = rk4_step(psi2, Va, g2, chem_potential, dx, dy, dt, Omega, 0)
    norm1 = np.sqrt(np.sum(np.abs(psi1)**2) * dx * dy)
    norm2 = np.sqrt(np.sum(np.abs(psi2)**2) * dx * dy)
    ratio = norm1/norm2
#    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
 #   psi /= norm
    if step % 1000 == 0:
        print(f"Real time step {step}/{t_steps}")
    if step % N_rt == 0:
        frames.append(psi1/ratio.copy())
        N_part = np.sqrt(np.sum(np.abs(psi2)**2) * dx * dy)
        norms.append(N_part)
        times.append(step*dt*1.474)
    if step % 25000 ==0:
        plt.figure()
        plt.imshow(np.abs(psi1/ratio), extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
        plt.title('Density at t=100')
        plt.colorbar()
        plt.show()

# Add time label on density axis
time_text = fig.suptitle('', y=0.98, fontsize=12)

# Animation update
def update(frame_time):
    frame, t = frame_time
    den = np.abs(frame)**2
    ph = np.angle(frame)
    im1.set_array(den)
    vmin1, vmax1 = den.min(), den.max()
    im1.set_clim(vmin1, vmax1)
    im2.set_array(ph)
    time_text.set_text(f't = {t:.1f} ms')
    return [im1, im2, time_text]

anim = animation.FuncAnimation(fig, update, frames=zip(frames, times), save_count=len(frames), blit=False, interval=100)

# Save to file
anim.save('./gpe_evolution_dissipation_Sebastian.mp4', writer='ffmpeg', dpi=150)
print('Animation saved as gpe_evolution.mp4')

# Plot particle number dynamics
plt.figure()
plt.plot(times, norms, '-o')
plt.xlabel('Time')
plt.ylabel('Total Particle Number (sqrt norm)')
plt.title('Particle Number vs Time')
plt.grid(True)
plt.show()

plt.close(fig)  # Close static figure

# Final density
#plt.figure()
#plt.imshow(np.abs(psi)**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
#plt.title(f'Density at t = {t_max}')
#plt.colorbar()
#plt.show()
