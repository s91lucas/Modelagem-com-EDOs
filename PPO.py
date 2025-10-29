import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Dados Experimentais ---
# 
t_experimental = np.array([0, 2, 5, 10, 15, 20, 30, 40, 50, 60, 90])
q_experimental = np.array([0, 1.3990, 1.5189, 1.8487, 1.9986, 1.9886, 1.9187, 1.9487, 2.0386, 1.9986, 2.0186])

# --- 2. Definição da Função do Modelo (PPO) ---
def ppo_model(t, qe, k1):
    """
    Define a equação integrada do modelo de Pseudo-Primeira Ordem (PPO).
    """
    # qt = qe(1 - exp(-k1*t))
    # Usamos np.exp() para que funcione com arrays do numpy
    return qe * (1 - np.exp(-k1 * t))

# --- 3. Otimização (Ajuste de Curva) ---
# Chutes iniciais para [qe, k1]
initial_guesses = [1.9, 0.5]

# 'curve_fit' encontra os melhores parâmetros (popt) e a matriz de covariância (pcov)
popt, pcov = curve_fit(
    ppo_model,        # A função do modelo que queremos ajustar
    t_experimental,   # Nossos dados 'x' (tempo)
    q_experimental,   # Nossos dados 'y' (capacidade adsortiva)
    p0=initial_guesses 
)

# Extrai os parâmetros otimizados
qe_fit, k1_fit = popt

# Extrai os erros dos parâmetros (o '±' valor)
# São a raiz quadrada da diagonal da matriz de covariância
perr = np.sqrt(np.diag(pcov))
qe_err, k1_err = perr

print("Otimização concluída.")

# --- 4. Cálculo das Estatísticas de Ajuste ---

print("\n--- Relatório de Ajuste do Modelo ---")

# Parâmetros básicos
N = len(q_experimental) # Número de pontos de dados
P = len(popt)           # Número de parâmetros (2: qe, k1)
DoF = N - P             # Graus de liberdade

# Calcula os valores simulados usando os parâmetros ajustados
q_simulado = ppo_model(t_experimental, qe_fit, k1_fit)

# Resíduos (erros)
residuos = q_simulado - q_experimental
# Soma dos Quadrados dos Resíduos (Erro)
SS_res = np.sum(residuos**2)
# Soma Total dos Quadrados
SS_tot = np.sum((q_experimental - np.mean(q_experimental))**2)

# R-Square (Coeficiente de Determinação, COD)
r_square = 1 - (SS_res / SS_tot)

# Imprime o relatório formatado
print(f"Model:                PPO (Pseudo-Primeira Ordem)")
print(f"Equation:             qe*(1-exp(-k1*t))")

print("--- Parâmetros Otimizados ---")
#  para formato
print(f"qe:                   {qe_fit: .5f} ± {qe_err: .5f}")
print(f"k1:                   {k1_fit: .5f} ± {k1_err: .5f}")
print(f"R²:                   {r_square: .5f}")
print("---------------------------------------")


# --- 5. Visualização do Resultado (A curva) ---
t_plot = np.linspace(0, 90, 100)
q_plot = ppo_model(t_plot, qe_fit, k1_fit)

plt.figure(figsize=(10, 6))
# Dados Experimentais 
plt.plot(t_experimental, q_experimental, 'o', label='Dados Experimentais', markersize=8, color='black')
# Curva Ajustada 
plt.plot(t_plot, q_plot, '-', label='Modelo PPO Ajustado', color='red', linewidth=2)

plt.title('Ajuste do Modelo de Pseudo-Primeira Ordem (PPO)')
plt.xlabel('Tempo (min)')
plt.ylabel('Capacidade Adsortiva (mg/g)')
plt.legend()
plt.grid(True)
plt.show()
