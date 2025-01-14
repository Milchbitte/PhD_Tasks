import numpy as np
import matplotlib.pyplot as plt

#Adjustable Parameters
R = 8.314  #(J/mol·K)
T_m_A = 8750  #Melting temperature of pure component Mg (K)
T_m_B = 4647  #Melting temperature of pure component Fe (K)
H_f_A = 25810 #Heat of fusion for component Mg (J/mol)
H_f_B = 111310  #Heat of fusion for component Fe (J/mol)
Omega = 10000  #Interaction parameter (J/mol)
tolerance = 1000  #Tolerance for chemical potential difference (J/mol)

#Temperature range and composition ranges 
T_range = np.concatenate(( 
    np.linspace(4000, 8000, 100), 
    np.linspace(8000, 9000, 200) 
    )) 
x_A_L_range = np.concatenate(( 
    np.linspace(0.01, 0.1, 100), 
    np.linspace(0.1, 0.8, 500), 
    np.linspace(0.8, 0.99, 500)
    )) 
x_A_S_range = np.linspace(0.01, 0.99, 200) 

#Functions
def delta_g_m(H_f, T, T_m):
    return H_f * (1 - T / T_m)

def safe_log(x):
    return np.log(np.clip(x, 1e-8, 1))

def chemical_potential(T, x_A, G_S, Delta_G_m, Omega, is_solid=True):
    x_B = 1 - x_A
    #Near-pure B region: handle x_A -> 0
    if x_A < 1e-6:  
        if is_solid:
            return G_S  
        else:
            return Delta_G_m
    #Near-pure A region: handle x_B -> 0
    if x_B < 1e-6:  
        if is_solid:
            return G_S  
        else:
            return Delta_G_m  
    #Standard calculation for other regions
    if is_solid:
        return G_S + R * T * safe_log(x_A)
    else:
        return G_S + Delta_G_m + R * T * safe_log(x_A) + Omega * x_B

        
def find_equilibrium(T, x_A_L, Delta_G_m_A, Delta_G_m_B, G_A_S, G_B_S, x_A_S_range, tolerance):
    for x_A_S in x_A_S_range:
        #Calculate chemical potentials
        mu_A_L = chemical_potential(T, x_A_L, G_A_S, Delta_G_m_A, Omega, is_solid=False)
        mu_B_L = chemical_potential(T, 1 - x_A_L, G_B_S, Delta_G_m_B, Omega, is_solid=False)
        mu_A_S = chemical_potential(T, x_A_S, G_A_S, Delta_G_m_A, Omega, is_solid=True)
        mu_B_S = chemical_potential(T, 1 - x_A_S, G_B_S, Delta_G_m_B, Omega, is_solid=True)

        #Check equilibrium conditions
        if abs(mu_A_L - mu_A_S) < tolerance and abs(mu_B_L - mu_B_S) < tolerance:
            return x_A_L, x_A_S
    return None, None

#Main Loop with Variable Tolerance
liquidus = []
solidus = []

for T in T_range:
    for x_A_L in x_A_L_range:
        #Calculate Delta G_m for A and B
        Delta_G_m_A = delta_g_m(H_f_A, T, T_m_A)
        Delta_G_m_B = delta_g_m(H_f_B, T, T_m_B)

        #Reference states for solid phases (set to zero for simplicity)
        G_A_S = 0
        G_B_S = 0

        #Adjust tolerance based on composition
        #tolerance = 50 if x_A_L <= 0.8 else 40
        tolerance = 30 if T > 8000 else 1000
        #Find equilibrium compositions
        x_L, x_S = find_equilibrium(T, x_A_L, Delta_G_m_A, Delta_G_m_B, G_A_S, G_B_S, x_A_S_range, tolerance)
        if x_L is not None and x_S is not None:
            liquidus.append((x_L, T))
            solidus.append((x_S, T))
            break

#Convert data to arrays for plotting
liquidus = np.array(liquidus, dtype=float)
solidus = np.array(solidus, dtype=float)

#Plot Phase Diagram
plt.figure(figsize=(8, 6))
plt.plot(liquidus[:, 0] * 100, liquidus[:, 1], 'r-', label='Liquidus Line')
plt.plot(solidus[:, 0] * 100, solidus[:, 1], 'b-', label='Solidus Line')
plt.fill_betweenx(liquidus[:, 1], solidus[:, 0] * 100, liquidus[:, 0] * 100, color='lightblue', alpha=0.3, label='Two-phase Region')

plt.xlim(0, 100)
plt.ylim(4000, 9000)

plt.xlabel('Mg#')
plt.ylabel('Te (K)')
plt.title('MgO-FeO Binary Phase Diagram')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
