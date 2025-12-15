import numpy as np
import matplotlib.pyplot as plt

# ================= PARAMETERS =================
NUM_SNAPSHOTS = 3000
R = 1.0
NU = 3.8
SIGMA = 8
P0 = 0
SECTOR_ANGLES = np.array([0, 2*np.pi/3, 4*np.pi/3])

# ================= HEX GRID =================
def hex_grid():
    centers = [(0, 0)]
    ring1 = [(np.sqrt(3)*R, 0),
             (np.sqrt(3)/2*R, 1.5*R),
             (-np.sqrt(3)/2*R, 1.5*R),
             (-np.sqrt(3)*R, 0),
             (-np.sqrt(3)/2*R, -1.5*R),
             (np.sqrt(3)/2*R, -1.5*R)]
    ring2 = [(2*x, 2*y) for x, y in ring1]
    return centers + ring1 + ring2

CELL_CENTERS = hex_grid()

# ================= USER POSITION =================
def random_user_in_sector(sector_angle):
    r = R * np.sqrt(np.random.rand())
    theta = sector_angle + np.random.uniform(-np.pi/3, np.pi/3)
    return r*np.cos(theta), r*np.sin(theta)

# ================= CHANNEL =================
def rx_power(d, alpha=0):
    PL = 10 * NU * np.log10(d)
    PC = 10 * alpha * np.log10(d)
    X = np.random.normal(0, SIGMA)
    return P0 - PL + PC + X

# ================= SIR COMPUTATION =================
def compute_SIR(reuse, alpha=0):
    sir_vals = []

    for _ in range(NUM_SNAPSHOTS):
        for s_idx, s_angle in enumerate(SECTOR_ANGLES):
            ux, uy = random_user_in_sector(s_angle)
            d0 = np.hypot(ux, uy)
            P_sig = rx_power(d0, alpha)

            interf_lin = 0

            for c_idx, (cx, cy) in enumerate(CELL_CENTERS):
                if c_idx == 0:
                    continue

                if reuse == 9 and c_idx % 3 != 0:
                    continue

                for is_idx, is_angle in enumerate(SECTOR_ANGLES):
                    if reuse in [3, 9] and is_idx != s_idx:
                        continue

                    ix, iy = random_user_in_sector(is_angle)
                    ix += cx
                    iy += cy

                    d_int = np.hypot(ix, iy)
                    interf_lin += 10**(rx_power(d_int, alpha)/10)

            sir = 10**(P_sig/10) / interf_lin
            sir_vals.append(10*np.log10(sir))

    return np.array(sir_vals)

# ================= THROUGHPUT =================
def throughput(SIR_dB, reuse):
    B = 100e6 / reuse
    gap = 10**(4/10)
    SIR = 10**(SIR_dB/10)
    return B * np.log2(1 + SIR/gap)

# ================= QUESTION 1 =================
def question_1():
    for reuse in [1, 3, 9]:
        SIR = compute_SIR(reuse)
        print(f"Reuse {reuse}: P(SIR ≥ -5 dB) = {np.mean(SIR >= -5):.3f}")

        s = np.sort(SIR)
        plt.plot(s, np.linspace(0,1,len(s)), label=f"Reuse {reuse}")

    plt.axvline(-5, linestyle='--', color='k', label='Threshold: -5 dB')
    plt.title("SIR-CDF for different frequency reuse factors")
    plt.xlabel("SIR (dB)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.show()

# ================= QUESTION 2 =================
def question_2():
    best = (0, 0, None)

    for alpha in np.arange(0, 1.05, 0.05):
        SIR = compute_SIR(3, alpha)
        p = np.mean(SIR >= -5)
        if p > best[1]:
            best = (alpha, p, SIR)

    print(f"Best α = {best[0]:.2f}, P = {best[1]:.3f}")

    s = np.sort(best[2])
    plt.plot(s, np.linspace(0,1,len(s)))
    plt.axvline(-5, linestyle='--', label='Threshold: -5 dB')
    plt.title(f"SIR-CDF for best α = {best[0]:.2f}")
    plt.xlabel("SIR (dB)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.show()

# ================= QUESTION 3 =================
def question_3():
    global NU
    for nu in [3.0, 3.8, 4.5]:
        NU = nu
        best = (0, 0, None)

        for alpha in np.arange(0, 1.05, 0.05):
            SIR = compute_SIR(3, alpha)
            p = np.mean(SIR >= -5)
            if p > best[1]:
                best = (alpha, p, SIR)

        s = np.sort(best[2])
        plt.plot(s, np.linspace(0,1,len(s)), label=f"ν={nu}")

    plt.axvline(-5, linestyle='--', label='Threshold: -5 dB')
    plt.title("SIR-CDF for different path-loss exponents")
    plt.xlabel("SIR (dB)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.show()

# ================= QUESTION 4 =================
def question_4():
    for reuse in [1, 3, 9]:
        SIR = compute_SIR(reuse)
        R = throughput(SIR, reuse)

        print(f"Reuse {reuse}: Avg={np.mean(R)/1e6:.2f} Mbps, "
              f"97%={np.percentile(R,97)/1e6:.2f} Mbps")

        s = np.sort(R)/1e6
        plt.plot(s, np.linspace(0,1,len(s)), label=f"Reuse {reuse}")

    plt.title("Throughput CDF for different frequency reuse factors")
    plt.xlabel("Throughput (Mbps)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.show()
