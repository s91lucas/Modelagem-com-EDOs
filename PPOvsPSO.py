import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Dados Experimentais ---
t_experimental = np.array([0, 2, 5, 10, 15, 20, 30, 40, 50, 60, 90])
q_experimental = np.array([0, 1.3990, 1.5189, 1.8487, 1.9986, 1.9886, 1.9187, 1.9487, 2.0386, 1.9986, 2.0186])

# --- 2. Definições dos Modelos ---

def ppo_model(t, qe, k1):
    """ Modelo de Pseudo-Primeira Ordem (PPO) """
    # qt = qe(1 - exp(-k1*t))
    # Adiciona uma pequena verificação para evitar overflow com t=0
    with np.errstate(over='ignore', invalid='ignore'):
        return np.nan_to_num(qe * (1 - np.exp(-k1 * t)))

def pso_model(t, qe, k2):
    """ Modelo de Pseudo-Segunda Ordem (PSO) """
    # qt = (k2 * qe^2 * t) / (1 + k2 * qe * t)
    with np.errstate(over='ignore', invalid='ignore'):
        numerator = k2 * (qe**2) * t
        denominator = 1 + k2 * qe * t
        # Evita divisão por zero no caso t=0
        result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        return result

# --- 3. Função para calcular as estatísticas ---
def get_stats(y_data, y_model, num_params):
    """Calcula estatísticas de ajuste R^2, R^2_adj, e Red. Chi-Sqr."""
    N = len(y_data)
    P = num_params
    DoF = N - P
    
    SS_res = np.sum((y_data - y_model)**2)
    SS_tot = np.sum((y_data - np.mean(y_data))**2)
    
    r_square = 1 - (SS_res / SS_tot)
    adj_r_square = 1 - (1 - r_square) * (N - 1) / (DoF - 1)
    red_chi_sqr = SS_res / DoF
    
    return r_square, adj_r_square, red_chi_sqr

# --- 4. Ajuste do Modelo PPO ---
print("Ajustando Modelo PPO...")
# Chutes iniciais [qe, k1]
popt_ppo, pcov_ppo = curve_fit(ppo_model, t_experimental, q_experimental, p0=[2.0, 0.5])
qe_ppo, k1_ppo = popt_ppo
q_model_ppo = ppo_model(t_experimental, qe_ppo, k1_ppo)
stats_ppo = get_stats(q_experimental, q_model_ppo, 2)

# --- 5. Ajuste do Modelo PSO ---
print("Ajustando Modelo PSO...")
# Chutes iniciais [qe, k2]
popt_pso, pcov_pso = curve_fit(pso_model, t_experimental, q_experimental, p0=[2.0, 0.5])
qe_pso, k2_pso = popt_pso
q_model_pso = pso_model(t_experimental, qe_pso, k2_pso)
stats_pso = get_stats(q_experimental, q_model_pso, 2)

# --- 6. Dados finais ---
print("\n--- Relatório de Ajuste (PPO) ---")
print(f"  qe = {qe_ppo:.4f}")
print(f"  k1 = {k1_ppo:.4f}")
print(f"  R²:     {stats_ppo[0]:.5f}")


print("\n--- Relatório de Ajuste (PSO) ---")
print(f"  qe = {qe_pso:.4f}")
print(f"  k2 = {k2_pso:.4f}")
print(f"  R²:     {stats_pso[0]:.5f}")


# --- 7. Plotagem dos Gráficos ---
t_plot = np.linspace(0, 90, 200)
q_plot_ppo = ppo_model(t_plot, qe_ppo, k1_ppo)
q_plot_pso = pso_model(t_plot, qe_pso, k2_pso)

plt.figure(figsize=(12, 8))
plt.plot(t_experimental, q_experimental, 'o', color='black', markersize=8, label='Dados Experimentais')
plt.plot(t_plot, q_plot_ppo, 'r-', linewidth=2, label=f'Modelo PPO (R²={stats_ppo[0]:.4f})')
plt.plot(t_plot, q_plot_pso, 'b--', linewidth=2, label=f'Modelo PSO (R²={stats_pso[0]:.4f})')

plt.title('Comparação dos Modelos Cinéticos PPO e PSO')
plt.xlabel('Tempo (min)')
plt.ylabel('Capacidade Adsortiva q (mg/g)')
plt.legend()
plt.grid(True)
plt.show()