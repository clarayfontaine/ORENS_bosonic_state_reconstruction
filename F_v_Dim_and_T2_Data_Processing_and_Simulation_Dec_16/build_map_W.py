# %%
from scipy import optimize
import time
import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, Qobj, rand_ket
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

AL = -exp_params.disp_points

P = (1j * np.pi * a.dag() * a).expm()


# this part if for obtaining the map
X_r = np.zeros([1 + exp_params.nD, Ntr])  # store readouts
X_r[0, :] = np.ones([1, Ntr])  # setting the ones
Y_r = np.zeros([exp_params.nD, Ntr])  # store the targets
for j in np.arange(0, Ntr):
    # qudit mixed state embedded in the cavity mode
    rd1 = np.zeros([cdim, cdim], dtype=np.complex_)
    u_rand = rand_ket(exp_params.D)
    r_rand = (u_rand * u_rand.dag()).full()
    rd1[0 : exp_params.D, 0 : exp_params.D] = r_rand  # randRho(D)

    # assign targets
    cw = 0
    # diagonal elements
    for j1 in np.arange(0, exp_params.D - 1):
        Y_r[cw, j] = rd1[j1, j1].real
        cw += 1
    # off-diagonal elements
    for j1 in np.arange(0, exp_params.D - 1):
        for j2 in np.arange(j1 + 1, exp_params.D):
            Y_r[cw, j] = rd1[j1, j2].real
            cw += 1
            Y_r[cw, j] = rd1[j1, j2].imag
            cw += 1

    r0 = Qobj(rd1)
    # evolution (time multiplexing)
    # rt = r0
    w = 0
    for v in np.arange(0, exp_params.nD):
        # T = float(T)#because the optimize gives T an np.array type
        U = Dis(AL[w])
        rt = U.dag() * r0 * U
        X_r[w + 1, j] = (rt * P).tr().real  # + 0*np.random.normal(0, xi, 1)
        w += 1

# ridge regression
lamb = 0

# training, now to obtain the map
X_R = np.zeros([1 + exp_params.nD, Ntr])  # will contain the parameters
X_R[0, :] = np.ones([1, Ntr])  # setting the ones
Y_R = np.zeros([exp_params.nD, Ntr])  # will contain the obs

# re-defining variables
X_R[1 : exp_params.nD + 1, :] = Y_r
Y_R[:, :] = X_r[1 : exp_params.nD + 1, :]

Error, beta = QN_regression(X_R, Y_R, lamb)

M = beta[:, 1 : exp_params.nD + 1]  # the map
W = np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), np.transpose(M))
eta = np.linalg.norm(M, 2) * np.linalg.norm(W, 2)
print(eta)

print("")
print("--- %s seconds ---" % (time.time() - start_time))


np.savez(
    f"map_variables\map_variables_D={exp_params.D}_nD={exp_params.nD}_W.npz",
    W=W,
    beta=beta,
)
