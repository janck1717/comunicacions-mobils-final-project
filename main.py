import numpy as np
import matplotlib.pyplot as plt
# ----------------- PARÁMETROS -----------------
num_snapshots = 2000          # >1000 para curvas suaves
nu = 3.8                      # exponente path-loss
sigma = 8                     # shadow fading en dB
Pt = 0                        # Potencia transmitida (0 dB valor relativo)
R = 1                         # Radio de la celda normalizado

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

# ----------------- GENERAR POSICIÓN ALEATORIA EN SECTOR CENTRAL -----------------
def random_user_position():
    r = R * np.sqrt(np.random.uniform(0,1))
    theta = np.random.uniform(-np.pi/3, np.pi/3)    # sector 120°
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# ----------------- DISTANCIA Y POTENCIA RECIBIDA -----------------
def pathloss_db(d):
    return 10 * nu * np.log10(d)

def received_power(d):
    PL = pathloss_db(d)
    X = np.random.normal(0, sigma)   # shadow fading
    return Pt - PL + X

# ----------------- LOOP MONTE-CARLO -----------------
SIR_list = []

for _ in range(num_snapshots):
    # Usuario deseado en celda central
    ux, uy = random_user_position()
    
    # Deseado
    d_des = np.sqrt(ux**2 + uy**2)
    P_des = received_power(d_des)
    
    # Interferentes: usuario en misma orientación sectorial
    P_int_sum = 0
    for cx, cy in cell_centers[1:]:
        dx = ux - cx
        dy = uy - cy
        d_i = np.sqrt(dx**2 + dy**2)
        P_int = received_power(d_i)
        P_int_sum += 10**(P_int/10)  # pasar a potencias lineales
    
    # SIR
    SIR = 10**(P_des/10) / P_int_sum
    SIR_dB = 10 * np.log10(SIR)
    
    SIR_list.append(SIR_dB)

SIR_list = np.array(SIR_list)




##comprobaciones##

""""
plt.figure()
plt.hist(SIR_list, bins=200, density=True, cumulative=True, histtype='step')
plt.xlabel("SIR (dB)")
plt.ylabel("CDF")
plt.grid(True)
plt.show()
"""

# -------- DISTRIBUCIÓN DEL USUARIO --------
xs, ys = [], []
for _ in range(5000):
    x, y = random_user_position()
    xs.append(x)
    ys.append(y)

plt.figure()
plt.scatter(xs,ys,s=2)
plt.title("Distribución del usuario")
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()

# -------- PATHLOSS --------
dvals = np.linspace(0.01, 5, 100)
PL = 10 * nu * np.log10(dvals)

plt.figure()
plt.plot(dvals, PL)
plt.title("Pathloss")
plt.grid(True)
plt.show()

# -------- SHADOW FADING --------
samples = np.random.normal(0, sigma, 10000)
plt.figure()
plt.hist(samples, bins=50, density=True)
plt.title("Shadow fading σ=8 dB")
plt.grid(True)
plt.show()

# -------- PERCENTILES SIR --------
print(
    np.percentile(SIR1, [1,5,50,95,99]),
    np.percentile(SIR3, [1,5,50,95,99]),
    np.percentile(SIR9, [1,5,50,95,99])
)

