import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

q_max = 1.95269 
Ka = 0.3    # Constante de adsorção (L/mg·min) - ajustado
Kd = 0.15   # Constante de dessorção (1/min) - ajustado
m = 50.0    # Massa do adsorvente (mg) - arbitrário
v = 100.0   # Volume da solução (mL) - arbitrário

def adsorption_model_derivatives(q, c):
    """Sistema de EDOs do modelo cinético de Langmuir"""
    dq_dt = Ka * c * (q_max - q) - Kd * q
    dc_dt = -(m / v) * dq_dt  # Balanço de massa
    return dq_dt, dc_dt

# --- Campo Vetorial ---
q_vals = np.linspace(0, 2.2, 25)
c_vals = np.linspace(0, 8, 25)
Q, C = np.meshgrid(q_vals, c_vals)

dQ_dt, dC_dt = np.zeros(Q.shape), np.zeros(C.shape)

for i in range(len(q_vals)):
    for j in range(len(c_vals)):
        dQ_dt[j, i], dC_dt[j, i] = adsorption_model_derivatives(Q[j, i], C[j, i])

# --- Plotando ---
plt.figure(figsize=(12, 8))

# Campo vetorial
plt.streamplot(Q, C, dQ_dt, dC_dt, color='blue', density=1.5, linewidth=1.2, 
               arrowsize=1.5, cmap='Blues')

# Curva de equilíbrio (dq/dt = 0)
ce_eq = np.linspace(0.01, 8, 100)
qe_eq = (q_max * Ka * ce_eq) / (Kd + Ka * ce_eq)
plt.plot(qe_eq, ce_eq, color='red', linewidth=3, 
         label=f'Curva de Equilíbrio\n(q_max = {q_max:.2f} mg/g)')

# --- Trajetória baseada nos SEUS dados experimentais ---
def wrapper_model(t, y):
    return adsorption_model_derivatives(y[0], y[1])

# Condição inicial próxima dos dados
y0_exp = [0.0, 5.0]  # q=0, c=5 mg/L (c seria a concentração do poluente absorvido pelo adsorvente)
t_span = [0, 90] # intervalo de tempo 

sol_exp = solve_ivp(wrapper_model, t_span, y0_exp, method='RK45', 
                   t_eval=np.linspace(0, 90, 100))

plt.plot(sol_exp.y[0], sol_exp.y[1], color='green', linewidth=3, 
         label='Trajetória Experimental Simulada')
plt.scatter(sol_exp.y[0][::10], sol_exp.y[1][::10], color='green', s=50, zorder=5)

# Ponto final teórico (q = q_max)
plt.scatter([q_max], [0], color='purple', s=150, marker='*', 
           label=f'Capacidade Máxima Teórica', zorder=6)

plt.title('Diagrama de Fase 2D - Sistema de Adsorção Langmuir', 
          fontsize=14, fontweight='bold')
plt.xlabel('Capacidade Adsortiva, q (mg/g)', fontsize=12, fontweight='bold')
plt.ylabel('Concentração, c (mg/L)', fontsize=12, fontweight='bold')
plt.xlim(0, 2.2)
plt.ylim(0, 8)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n📊 PARÂMETROS USADOS:")
print(f"q_max = {q_max:.4f} mg/g (do seu modelo PPO)")
print(f"Ka = {Ka:.3f} L/mg·min (constante de adsorção)")
print(f"Kd = {Kd:.3f} 1/min (constante de dessorção)")
print(f"Razão Ka/Kd = {Ka/Kd:.3f} L/mg (afinidade)")