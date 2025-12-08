import numpy as np
import matplotlib.pyplot as plt

# ----------------- PARÁMETROS -----------------
num_snapshots = 3000
nu = 3.8
sigma = 8
Pt = 0
R = 1

# ----------------- COORDENADAS CELDAS -----------------
def hex_grid():
    coords = [(0,0)]
    ring1 = [
        (np.sqrt(3)*R, 0),
        (np.sqrt(3)/2*R, 1.5*R),
        (-np.sqrt(3)/2*R, 1.5*R),
        (-np.sqrt(3)*R, 0),
        (-np.sqrt(3)/2*R, -1.5*R),
        (np.sqrt(3)/2*R, -1.5*R)
    ]
    ring2 = []
    for x,y in ring1:
        ring2.append((2*x, 2*y))
    return [(0,0)] + ring1 + ring2

cell_centers = hex_grid()

# ----------------- POSICIÓN ALEATORIA -----------------
def random_user_position():
    r = R * np.sqrt(np.random.uniform(0,1))
    theta = np.random.uniform(-np.pi/3, np.pi/3)
    return r*np.cos(theta), r*np.sin(theta), theta

# ----------------- CANAL -----------------
def received_power(d):
    PL = 10 * nu * np.log10(d)
    X = np.random.normal(0, sigma)
    return Pt - PL + X

# ----------------- POWER CONTROL -----------------
def received_power_PC(d, alpha):
    PL = 10 * nu * np.log10(d)
    PC = 10 * alpha * np.log10(d)
    X = np.random.normal(0, sigma)
    return Pt - PL + PC + X

# ----------------- SIR SIN POWER CONTROL -----------------
def compute_SIR(reuse):
    SIR_vals = []

    for _ in range(num_snapshots):
        ux, uy, theta_user = random_user_position()
        d_des = np.sqrt(ux**2 + uy**2)
        P_des = received_power(d_des)

        P_int_sum = 0

        for cx, cy in cell_centers[1:]:
            dx = ux - cx
            dy = uy - cy
            d_i = np.sqrt(dx**2 + dy**2)

            theta_i = np.arctan2(dy, dx)
            theta_diff = np.abs(theta_i - theta_user)
            if theta_diff > np.pi:
                theta_diff = 2*np.pi - theta_diff

            if reuse == 1:
                active = True
            elif reuse == 3:
                active = theta_diff < np.pi/3
            elif reuse == 9:
                active = False
            else:
                raise ValueError("reuse debe ser 1, 3 o 9")

            if active:
                P_int = received_power(d_i)
                P_int_sum += 10**(P_int/10)

        if reuse == 9:
            SIR = 10**(P_des/10) / 1e-18
        else:
            SIR = 10**(P_des/10) / P_int_sum

        SIR_vals.append(10*np.log10(SIR))

    return np.array(SIR_vals)

# ----------------- SIR CON POWER CONTROL -----------------
def compute_SIR_PC(reuse, alpha):
    SIR_vals = []

    for _ in range(num_snapshots):
        ux, uy, theta_user = random_user_position()
        d_des = np.sqrt(ux**2 + uy**2)
        P_des = received_power_PC(d_des, alpha)

        P_int_sum = 0

        for cx, cy in cell_centers[1:]:
            dx = ux - cx
            dy = uy - cy
            d_i = np.sqrt(dx**2 + dy**2)

            theta_i = np.arctan2(dy, dx)
            theta_diff = np.abs(theta_i - theta_user)
            if theta_diff > np.pi:
                theta_diff = 2*np.pi - theta_diff

            if reuse == 3:
                active = theta_diff < np.pi/3
            else:
                active = True

            if active:
                P_int = received_power_PC(d_i, alpha)
                P_int_sum += 10**(P_int/10)

        SIR = 10**(P_des/10) / P_int_sum
        SIR_vals.append(10*np.log10(SIR))

    return np.array(SIR_vals)

# ----------------- SIMULACIÓN PRINCIPAL REUSE -----------------
SIR1 = compute_SIR(1)
SIR3 = compute_SIR(3)
SIR9 = compute_SIR(9)

plt.figure()
for SIR, label in [(SIR1, "Reuse 1"), (SIR3, "Reuse 3"), (SIR9, "Reuse 9")]:
    plt.hist(SIR, bins=200, density=True, cumulative=True, histtype='step', label=label)

plt.xlabel("SIR (dB)")
plt.ylabel("CDF")
plt.title("SIR CDF vs Reuse Factor")
plt.grid(True)
plt.legend()
plt.show()
