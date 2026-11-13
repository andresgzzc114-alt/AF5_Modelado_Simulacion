import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del sistema
Ir = 0.16
m = 5.0
g = 9.81
lc = 0.01
b = 0.14
fc = 0.45

# Torque aplicado por el motor
def torque_motor(t):
    return 1.5 * np.sin(t)

# Modelo dinámico del robot-péndulo
def modelo_pendulo(t, x):
    q = x[0]      # Posición angular [rad]
    qd = x[1]     # Velocidad angular [rad/s]

    # Aproximación suave de la fricción de Coulomb
    friccion_coulomb = fc * np.tanh(100000.0 * qd)

    # Aceleración angular
    qdd = (
        torque_motor(t)
        - m * g * lc * np.sin(q)
        - b * qd
        - friccion_coulomb
    ) / Ir

    return [qd, qdd]

# Tiempo de simulación
ti = 0.0
tf = 10.0
h = 0.0025

n_puntos = int(round((tf - ti) / h)) + 1
t_eval = np.linspace(ti, tf, n_puntos)

# Condiciones iniciales
x0 = [0.0, 0.0]

# Resolver ecuaciones diferenciales
sol = solve_ivp(
    modelo_pendulo,
    [ti, tf],
    x0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-7,
    atol=1e-9
)

# Resultados
t = sol.t
q = sol.y[0]
qd = sol.y[1]

# Convertir de radianes a grados
q_deg = np.rad2deg(q)
qd_deg = np.rad2deg(qd)

# Gráfica de posición
plt.figure(figsize=(10, 4.5))
plt.plot(t, q_deg)
plt.xlabel("Tiempo [s]")
plt.ylabel("Posición angular [°]")
plt.title("Posición angular del robot-péndulo")
plt.grid(True)
plt.show()

# Gráfica de velocidad
plt.figure(figsize=(10, 4.5))
plt.plot(t, qd_deg)
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad angular [°/s]")
plt.title("Velocidad angular del robot-péndulo")
plt.grid(True)
plt.show()

# Valores principales
print(f"Posición final: {q_deg[-1]:.2f}°")
print(f"Velocidad final: {qd_deg[-1]:.2f}°/s")
print(f"Posición máxima: {np.max(q_deg):.2f}°")
print(f"Posición mínima: {np.min(q_deg):.2f}°")
print(f"Velocidad máxima: {np.max(qd_deg):.2f}°/s")
print(f"Velocidad mínima: {np.min(qd_deg):.2f}°/s")
