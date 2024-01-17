from scipy import optimize
import time
import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, Qobj, rand_dm, displace, expect, fock, fock_dm
from QN_regression import QN_regression
import exp_params

start_time = time.time()  # checking how long the code takes

Ntr = exp_params.D**2  # no of training for obtaining the map, at least D^2


cdim = 30  # truncation
a = destroy(cdim)  # annihilation for cavity


# displacement operator
def Dis(alpha):
    U = (alpha * a.dag() - np.conj(alpha) * a).expm()
    return U


# this part if for obtaining the map X=MY+V, where X are the observables (peaks) and Y are the elements of rho
X_r = np.zeros([1 + exp_params.nD, Ntr])  # store readouts 36x35
X_r[0, :] = np.ones([1, Ntr])  # setting the ones in first row of X
Y_rnd = np.zeros([exp_params.D**2 - 1, Ntr])  # store the targets 35x36

for j in np.arange(0, Ntr):
    # Assign targets
    rho_rnd = np.zeros([cdim, cdim], dtype=np.complex_)
    rho_rnd[0 : exp_params.D, 0 : exp_params.D] = rand_dm(
        exp_params.D
    )  # random density matrix
    Y_rnd[0 : exp_params.D - 1, j] = np.diagonal(rho_rnd).real[
        0 : exp_params.D - 1
    ]  # Diagonal of rho
    off_diag = rho_rnd[np.triu_indices(exp_params.D, 1)]  # Upper triangle of rho
    Y_rnd[exp_params.D - 1 :: 2, j] = np.real(off_diag)
    Y_rnd[exp_params.D :: 2, j] = np.imag(off_diag)
    r0 = Qobj(rho_rnd)

    w = 0
    for v in np.arange(0, exp_params.TM1):
        # T = float(T)#because the optimize gives T an np.array type
        U = Dis(exp_params.disp_points[w])
        rt = U * r0 * U.dag()
        X_r[w + 1, j] = rt[1, 1].real  # + 0*np.random.normal(0, xi, 1)
        w += 1
    for v1 in np.arange(0, exp_params.TM2):
        # T = float(T)#because the optimize gives T an np.array type
        U = Dis(exp_params.disp_points[w])
        rt = U * r0 * U.dag()
        X_r[w + 1, j] = rt[2, 2].real  # + 0*np.random.normal(0, xi, 1)
        w += 1
    for v2 in np.arange(0, exp_params.TM3):
        # T = float(T)#because the optimize gives T an np.array type
        U = Dis(exp_params.disp_points[w])
        rt = U * r0 * U.dag()
        X_r[w + 1, j] = rt[3, 3].real  # + 0*np.random.normal(0, xi, 1)
        w += 1
    for v3 in np.arange(0, exp_params.TM4):
        # T = float(T)#because the optimize gives T an np.array type
        U = Dis(exp_params.disp_points[w])
        rt = U * r0 * U.dag()
        X_r[w + 1, j] = rt[4, 4].real  # + 0*np.random.normal(0, xi, 1)
        w += 1
    for v4 in np.arange(0, exp_params.TM5):
        U = Dis(exp_params.disp_points[w])
        rt = U * r0 * U.dag()
        X_r[w + 1, j] = rt[5, 5].real  # + 0*np.random.normal(0, xi, 1)
        w += 1
    for v5 in np.arange(0, exp_params.TM6):
        U = Dis(exp_params.disp_points[w])
        rt = U * r0 * U.dag()
        X_r[w + 1, j] = rt[6, 6].real  # + 0*np.random.normal(0, xi, 1)
        w += 1
    for v6 in np.arange(0, exp_params.TM7):
        U = Dis(exp_params.disp_points[w])
        rt = U * r0 * U.dag()
        X_r[w + 1, j] = rt[7, 7].real  # + 0*np.random.normal(0, xi, 1)
        w += 1

# ridge regression
lamb = 0

# training, now to obtain the map
X_R = np.zeros([1 + exp_params.D**2 - 1, Ntr])  # will contain the parameters
X_R[0, :] = np.ones([1, Ntr])  # setting the ones
Y_R = np.zeros([exp_params.nD, Ntr])  # will contain the obs

# re-defining variables
X_R[1 : exp_params.nD + 1, :] = Y_rnd  # X_R are the elements of rho
Y_R[:, :] = X_r[1 : exp_params.nD + 1, :]  # Y_R are the observables

Error, beta = QN_regression(X_R, Y_R, lamb)  # beta has M and V s.t. X=MY+V
M = beta[:, 1 : exp_params.nD + 1]  # the map
W = np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), np.transpose(M))
eta = np.linalg.norm(M, 2) * np.linalg.norm(W, 2)
print("CD = ", eta)
print("")
print("--- %s seconds ---" % (time.time() - start_time))


np.savez(
    f"map_variables\map_variables_D={exp_params.D}_nD={exp_params.nD}_Q.npz",
    W=W,
    beta=beta,
)
