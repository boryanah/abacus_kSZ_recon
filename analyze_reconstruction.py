import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# parameters
tracer = "LRG"
Mean_z = 0.5
Delta_z = 0.4
h = 0.6736
a = 1./(1+Mean_z)
mode = "box" #"lc" #"box"

def get_Psi_dot(Psi_dot, want_rsd=False):
    # divide by 1+f in los direction
    #Psi_dot = (unit_los*Psi_dot)/(1.+f) # real space
    Psi_dot *= a*f*H # km/s/h
    Psi_dot /= h # km/s
    Psi_dot_r = np.sum(Psi_dot*unit_los, axis=1)
    if want_rsd:
        Psi_dot_r /= (1+f) # real space
    N = Psi_dot.shape[0]
    print("number", N)
    assert len(vel_r) == vel.shape[0] == len(Psi_dot_r) == N
    return Psi_dot, Psi_dot_r, N

def print_coeff(Psi_dot, want_rsd=False):
    # compute velocity from displacement
    Psi_dot, Psi_dot_r, N = get_Psi_dot(Psi_dot, want_rsd=want_rsd)

    # compute rms
    Psi_dot_rms = np.sqrt(np.mean(Psi_dot[:, 0]**2+Psi_dot[:, 1]**2+Psi_dot[:, 2]**2))/np.sqrt(3.)
    Psi_dot_r_rms = np.sqrt(np.mean(Psi_dot_r**2))
    print("Psi_dot_rms", Psi_dot_rms)
    print("Psi_dot_r_rms", Psi_dot_r_rms)

    # compute statistics
    print("overall", np.sum((Psi_dot*vel))/N/(3.*vel_rms*Psi_dot_rms))
    print("in each direction", np.sum(Psi_dot*vel, axis=0)/N/(vel_rms*Psi_dot_rms))
    print("in los direction", np.sum(Psi_dot_r*vel_r)/N/(vel_r_rms*Psi_dot_r_rms))

    plt.figure(figsize=(9, 7))
    plt.scatter(vel_r, Psi_dot_r, s=1, alpha=0.1, color='teal')
    if want_rsd:
        plt.savefig("figs/vel_rec_rsd.png")
    else:
        plt.savefig("figs/vel_rec.png")
    plt.close()

# load data
tmp_dir = "/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/"
if mode == "lc":
    data = np.load(f"{tmp_dir}/displacements_{tracer}_postrecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
elif mode == "box":
    data = np.load(f"{tmp_dir}/displacements_{tracer}_postrecon_z{Mean_z:.3f}.npz")
print(data.files) # ['displacements', 'velocities', 'growth_factor', 'Hubble_z']
Psi_dot = data['displacements'] # cMpc/h
Psi_dot_rsd = data['displacements_rsd'] # cMpc/h
vel = data['velocities'] # km/s
if mode == "lc":
    unit_los = data['unit_los']
elif mode == "box":
    unit_los = np.array([0, 0, 1.])
f = data['growth_factor']
H = data['Hubble_z'] # km/s/Mpc

# velocity in los direction
vel_r = np.sum(vel*unit_los, axis=1)

# compute the rms
vel_rms = np.sqrt(np.mean(vel[:, 0]**2+vel[:, 1]**2+vel[:, 2]**2))/np.sqrt(3.)
vel_r_rms = np.sqrt(np.mean(vel_r**2))
print("vel_rms", vel_rms)
print("vel_r_rms", vel_r_rms)

print("no rsd")
print_coeff(Psi_dot, want_rsd=False)
print("rsd")
print_coeff(Psi_dot_rsd, want_rsd=True)
