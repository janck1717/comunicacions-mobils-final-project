import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
NUM_SNAPSHOTS = 3000     # Monte Carlo realizations per sector
R = 1.0                 # Cell radius (normalized units)
NU = 3.8                # Path-loss exponent ν
SIGMA = 8               # Shadowing standard deviation (dB)
P0 = 0                  # Reference transmit power (dBm, arbitrary)
SECTOR_ANGLES = np.array([0, 2*np.pi/3, 4*np.pi/3])  # 3-sector BS

# =========================================================
# HEXAGONAL CELL GRID
# =========================================================
def hex_grid():
    """
    Generates a simple hexagonal cellular layout:
    - One central cell
    - First ring of 6 neighbors
    - Second ring (scaled version of ring 1)
    Cell centers are expressed in Cartesian coordinates.
    """
    centers = [(0, 0)]

    # First ring of interferers
    ring1 = [(np.sqrt(3)*R, 0),
             (np.sqrt(3)/2*R, 1.5*R),
             (-np.sqrt(3)/2*R, 1.5*R),
             (-np.sqrt(3)*R, 0),
             (-np.sqrt(3)/2*R, -1.5*R),
             (np.sqrt(3)/2*R, -1.5*R)]

    # Second ring (farther interferers)
    ring2 = [(2*x, 2*y) for x, y in ring1]

    return centers + ring1 + ring2

CELL_CENTERS = hex_grid()

# =========================================================
# USER LOCATION MODEL
# =========================================================
def random_user_in_sector(sector_angle):
    """
    Generates a random user uniformly distributed
    within a 120° sector of a cell.
    
    - Radius uses sqrt() to ensure uniform spatial density
    - Angle spans ±60° around the sector boresight
    """
    r = R * np.sqrt(np.random.rand())
    theta = sector_angle + np.random.uniform(-np.pi/3, np.pi/3)
    return r*np.cos(theta), r*np.sin(theta)

# =========================================================
# CHANNEL / RECEIVED POWER MODEL
# =========================================================
def rx_power(d, alpha=0):
    """
    Computes received power in dB:
    
    - Path loss: 10*ν*log10(d)
    - Power control term: 10*α*log10(d)
      α = 0  → no power control
      α = 1  → full path-loss compensation
    - Log-normal shadowing 
    """
    PL = 10 * NU * np.log10(d)       # Path loss
    PC = 10 * alpha * np.log10(d)    # Power control
    X = np.random.normal(0, SIGMA)   # Shadowing
    return P0 - PL + PC + X

# =========================================================
# SIR COMPUTATION
# =========================================================
def compute_SIR(reuse, alpha=0):
    """
    Computes SIR samples (in dB) for a given:
    - Frequency reuse factor (1, 3, or 9)
    - Power control factor α
    
    Interference is accumulated in linear scale.
    """
    sir_vals = []

    for _ in range(NUM_SNAPSHOTS):

        # Loop over the 3 sectors of the serving cell
        for s_idx, s_angle in enumerate(SECTOR_ANGLES):

            # Desired user location and received signal power
            ux, uy = random_user_in_sector(s_angle)
            d0 = np.hypot(ux, uy)
            P_sig = rx_power(d0, alpha)

            interf_lin = 0  # Linear interference power sum

            # Loop over neighboring cells
            for c_idx, (cx, cy) in enumerate(CELL_CENTERS):

                if c_idx == 0:
                    continue  # Skip serving cell

                # For reuse 9, only 1/3 of cells reuse same frequency
                if reuse == 9 and c_idx % 3 != 0:
                    continue

                # Loop over interfering sectors
                for is_idx, is_angle in enumerate(SECTOR_ANGLES):

                    # For reuse 3 or 9, only same-sector interferers
                    if reuse in [3, 9] and is_idx != s_idx:
                        continue

                    # Interfering user position
                    ix, iy = random_user_in_sector(is_angle)
                    ix += cx
                    iy += cy

                    d_int = np.hypot(ix, iy)
                    interf_lin += 10**(rx_power(d_int, alpha)/10)

            # SIR computation (linear → dB)
            sir = 10**(P_sig/10) / interf_lin
            sir_vals.append(10*np.log10(sir))

    return np.array(sir_vals)

# =========================================================
# THROUGHPUT MODEL
# =========================================================
def throughput(SIR_dB, reuse):
    """
    Computes user throughput using Shannon capacity with:
    - Bandwidth divided by reuse factor
    - SNR gap (4 dB) to model practical modulation/coding
    """
    B = 100e6 / reuse       # Effective bandwidth
    gap = 10**(4/10)        # Implementation gap
    SIR = 10**(SIR_dB/10)
    return B * np.log2(1 + SIR/gap)

# =========================================================
# QUESTION 1: SIR DISTRIBUTION VS REUSE
# =========================================================
def question_1():
    """
    Evaluates how frequency reuse improves SIR statistics.
    Outputs:
    - Probability that SIR ≥ -5 dB
    - CDF of SIR for reuse 1, 3, and 9
    """
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

# =========================================================
# QUESTION 2: OPTIMAL POWER CONTROL α
# =========================================================
def question_2():
    """
    Searches for the power-control factor α that maximizes
    coverage probability P(SIR ≥ -5 dB) for reuse 3.
    """
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

# =========================================================
# QUESTION 3: IMPACT OF PATH-LOSS EXPONENT ν
# =========================================================
def question_3():
    """
    Studies how propagation conditions (ν) affect SIR
    when power control is optimized.
    """
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

# =========================================================
# QUESTION 4: THROUGHPUT PERFORMANCE
# =========================================================
def question_4():
    """
    Converts SIR statistics into throughput and compares:
    - Average throughput
    - Cell-edge (97% CDF) throughput
    """
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
