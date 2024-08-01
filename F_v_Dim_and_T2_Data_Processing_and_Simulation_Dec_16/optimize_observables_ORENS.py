#%%The code retrieves optimised displacements, will be stored in 'final_disps'
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
from scipy.optimize import minimize
from IPython.display import display, clear_output
from qutip import *  # destroy, Qobj, basis
from helpers.QN_regression import QN_regression
import time

# Start timing
start_time = time.time()

# Constants and initializations
FD = 6 # Dimension to read
n_disps = FD**2 - 1  # Number of displacement amplitudes
st = 0.5 # Strength of random initial amplitudes
cdim = 40  # Cavity dimension

# Operators
a = destroy(cdim).full()
adag = a.T.conj()

# Projection operators
P = np.zeros([FD, cdim, cdim], dtype=np.complex_)
for j in range(FD):
    P[j] = (basis(cdim, j) * basis(cdim, j).dag()).full()

NN = 5  # Number of different initial random seeds
dx = 0.001 # Infinitesimal for derivatives

# Which probability to measure: P_0, ..., P_(FD-1)
PN = np.zeros(FD, dtype=np.int_)
PN[FD-1] = FD**2 - 1 # Using only a single one, P_(FD-1)

# Getting the K matrix, rho = K * Y + Theta
Ntr = FD**2
X_R = np.zeros([1 + n_disps, Ntr], dtype=np.complex_)  # Store readouts
X_R[0, :] = np.ones([1, Ntr])  # Setting the ones
Y_R = np.zeros([FD**2, Ntr], dtype=np.complex_)  # Store the targets
for j in np.arange(0, Ntr):
    rd1 = np.zeros([cdim, cdim], dtype=np.complex_)
    u_rand = rand_ket(FD)
    r_rand = (u_rand * u_rand.dag()).full()
    rd1[0:FD, 0:FD] = r_rand

    cw = 1
    for j1 in np.arange(0, FD - 1):
        X_R[cw, j] = rd1[j1, j1].real
        cw += 1
    for j1 in np.arange(0, FD - 1):
        for j2 in np.arange(j1 + 1, FD):
            X_R[cw, j] = rd1[j1, j2].real
            cw += 1
            X_R[cw, j] = rd1[j1, j2].imag
            cw += 1

    Y_R[:, j] = np.transpose((np.transpose(r_rand)).reshape((FD**2, 1)))
Error, beta = QN_regression(X_R, Y_R, 0)
print(f'Error is {Error}')
K = beta[:, 1:n_disps + 1]

def wigner_mat(disps):
    ND = len(disps)
    M = np.zeros([ND, FD**2], dtype=complex)
    ct = 0
    for jj in range(len(PN)):
        if PN[jj] > 0:
            for k in range(PN[jj]):
                beta = disps[ct]
                D = expm(beta * adag - np.conj(beta) * a)
                Ms = D @ P[jj] @ np.conj(np.transpose(D))
                Mst = Ms[0:FD, 0:FD]
                M[ct, :] = Mst.reshape((1, FD**2))
                ct += 1
    M_new = np.matmul(M, K)
    return M_new

def wigner_mat_and_grad(disps):
    ND = len(disps)
    wig_tens = wigner_mat(disps)
    grad_mat_r = np.zeros((ND, FD**2), dtype=complex)
    grad_mat_i = np.zeros((ND, FD**2), dtype=complex)

    disps_dr = disps + dx
    disps_di = disps + dx * 1j

    wig_tens_dr = wigner_mat(disps_dr)
    wig_tens_di = wigner_mat(disps_di)

    grad_mat_r = (wig_tens_dr - wig_tens) / dx
    grad_mat_i = (wig_tens_di - wig_tens) / dx

    return wig_tens, grad_mat_r, grad_mat_i

def cost_and_grad(r_disps):
    N = len(r_disps)
    c_disps = r_disps[:int(N / 2)] + 1j * r_disps[int(N / 2):]
    M, dM_rs, dM_is = wigner_mat_and_grad(c_disps)
    U, S, Vd = svd(M)
    NS = len(Vd)
    cn = S[0] / S[-1]
    dS_r = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_rs, Vd.conj().T).real
    dS_i = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_is, Vd.conj().T).real
    
    grad_cn_r = (dS_r[0] * S[-1] - S[0] * dS_r[-1]) / (S[-1]**2)
    grad_cn_i = (dS_i[0] * S[-1] - S[0] * dS_i[-1]) / (S[-1]**2)
    
    return cn, np.concatenate((grad_cn_r, grad_cn_i))

best_cost = float('inf')

def wrap_cost(disps):
    global best_cost
    cost, grad = cost_and_grad(disps)
    best_cost = min(cost, best_cost)
    return cost, grad

CD = np.zeros(NN)
DIS = np.zeros([NN, n_disps], dtype=np.complex_)
for vv in np.arange(0, NN):
    init_disps = np.random.normal(0, st, 2 * n_disps)
    ret = minimize(wrap_cost, init_disps, method='L-BFGS-B', jac=True, options=dict(ftol=1e-6))
    new_disps = ret.x[:n_disps] + 1j * ret.x[n_disps:]

    CD[vv] = ret.fun
    DIS[vv, :] = new_disps

sorted_index = np.argsort(CD)
final_disps = DIS[sorted_index[0], :]
final_CD = CD[sorted_index[0]]

# Plotting the optimum displacements
fig = plt.figure(figsize=(3, 3))
for k in np.arange(0, len(final_disps)):
    plt.plot(final_disps[k].real, final_disps[k].imag, 'ok')
plt.xlabel('Re' + r'$(\alpha)$')
plt.ylabel('Im' + r'$(\alpha)$')
plt.title('CD = %.4f' % (final_CD))
plt.grid()

cabs = np.zeros(len(final_disps))
for j in range(len(final_disps)):
    cabs[j] = np.abs(final_disps[j])
print(f'max dis amplitude is {np.max(cabs)}')

print("")
print("--- %s seconds ---" % (time.time() - start_time))