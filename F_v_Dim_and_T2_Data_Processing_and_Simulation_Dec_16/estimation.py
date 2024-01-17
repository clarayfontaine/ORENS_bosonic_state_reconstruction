# %%
import numpy as np
from qutip import Qobj, qeye
from scipy.linalg import sqrtm
import time


def PSD_MLE_rho(rho):  # -> pos semi definite rho, given hermitian with trace 1
    d = rho.shape[0]
    w, v = np.linalg.eig(rho)
    idx = np.argsort(w)[::-1]  # reverse, now from largest to smallest
    w = w[idx]
    v = v[:, idx]

    la = 0 * w  # to store eigenvalues of new state
    a = 0  # accumulator
    i = d - 1  # index

    while w[i] + a / (i + 1) < 0:
        la[i] = 0
        a += w[i]
        i += -1

    for k in np.arange(0, i + 1):
        la[k] = w[k] + a / (i + 1)

    rho_f = 0 * rho.full()  # store final density matrix
    for x in np.arange(0, len(la)):
        rho_f = rho_f + (la[x] * Qobj(v[:, x]) * Qobj(v[:, x]).dag()).full()

    return Qobj(rho_f)


def tr1_rho(M, W):
    d = int(np.sqrt(M.shape[1]))
    Id = qeye(d)
    Idvec = (np.transpose(Id)).reshape((d**2, 1))  # identity in vec form
    MM = np.zeros([d**2 + 1, d**2 + 1], dtype=complex)
    XX = np.zeros([d**2 + 1, 1], dtype=complex)
    MM[: d**2, : d**2] = M.dag() * M
    MM[: d**2, d**2] = np.transpose(Idvec)  # apparently it works like this
    MM[d**2, : d**2] = np.transpose(Idvec)
    MM[d**2, d**2] = 0
    XX[: d**2, 0] = np.transpose(M.dag() * W)  # apparently it works like this
    XX[d**2, 0] = 1
    YY = np.matmul(np.linalg.inv(MM), XX)
    rvec_est = YY[: d**2, 0]
    return rvec_est


def bayesian_rho_est(numSamp, N, rho_tar, rhoLS):
    start_time = time.time()

    def fidelity_f(rho1, rho2):
        sqrt_rho1 = sqrtm(rho1)
        term = sqrt_rho1 @ rho2 @ sqrt_rho1
        trace_term = np.trace(sqrtm(term))
        fidelity_value = np.abs(trace_term) ** 2
        return fidelity_value

    # loop parameters
    THIN = np.array([2**7])  # 2**np.arange(8)#np.array([2**7])#
    samplers = 1
    # inputs

    alpha = 1
    Mb = 500
    r = 1.1

    D = rho_tar.shape[0]
    numParam = 2 * D**2 + D

    sigma = 1 / np.sqrt(N)
    # rhoLS = mat_data['rhoLS']#(rho_tar + 0.05*qeye(cdim)).unit()#rho_est
    # print(f"fidelity for LS is {fidelity(rho_tar, rhoLS)}")
    rhoLSvec = rhoLS.reshape([D**2, 1])  # rho_est col
    # sampling loop
    Fmean = np.zeros([samplers, len(THIN)])
    Fstd = np.zeros([samplers, len(THIN)])
    samplerTime = np.zeros([samplers, len(THIN)])

    # param to rho col function
    def paramToRhoCol(par):
        # D = 9
        # par = np.random.random(171)
        Xr = np.transpose(
            par[0 : D**2].reshape([D, D])
        )  # to be consistent with MATLAB code reshaping
        Xi = np.transpose(par[D**2 : 2 * D**2].reshape([D, D]))
        X = Xr + 1j * Xi  # matrix of column vectors (not normalised)
        NORM = np.linalg.norm(X, axis=0)  # norm of each column
        W = X / NORM  # normalise each column
        Y = par[2 * D**2 :]  # projector weights
        gamma = Y / np.sum(Y)  # normalise
        rho = W @ np.diag(gamma) @ W.transpose().conj()
        z = rho.reshape([D**2, 1])
        return z

    rho_BME = rhoLS * 0
    # the loop
    for k in range(len(THIN)):
        for m in range(samplers):
            param0 = np.zeros(numParam)
            Fest = np.zeros([numSamp, 1])
            np.random.seed(
                int(time.time())
            )  # change the seed based on time to ensure good random
            param0[0 : 2 * D**2] = np.random.randn(2 * D**2)  # initial seed
            param0[2 * D**2 :] = np.random.gamma(alpha, scale=1, size=D)
            beta1 = 0.1  # initial parameters for stepsize
            beta2 = 0.1
            acc = 0  # counter of acceptances
            # initial point
            x = param0
            rhoX = paramToRhoCol(x)
            logX = -1 / (2 * sigma**2) * np.linalg.norm(
                rhoX - rhoLSvec
            ) ** 2 + np.sum(alpha * np.log(x[2 * D**2 :]) - x[2 * D**2 :])
            # pCN loop
            tt = time.time()
            for j in range(numSamp * THIN[k]):
                # proposed update parameters:
                newGauss = np.sqrt(1 - beta1**2) * x[
                    0 : 2 * D**2
                ] + beta1 * np.random.randn(2 * D**2)
                newGamma = x[2 * D**2 :] * np.exp(beta2 * np.random.randn(D))
                y = np.concatenate((newGauss, newGamma))
                rhoY = paramToRhoCol(y)
                logY = -1 / (2 * sigma**2) * np.linalg.norm(
                    rhoY - rhoLSvec
                ) ** 2 + np.sum(alpha * np.log(y[2 * D**2 :]) - y[2 * D**2 :])
                if np.log(np.random.random(1)[0]) < logY - logX:
                    x = y
                    logX = logY
                    acc += 1
                if j % Mb == 0:  # stepsize adaptation
                    rat = (
                        acc / Mb
                    )  # estimate acceptance probability, and keep near optimal 0.234
                    if rat > 0.3:
                        beta1 *= r
                        beta2 *= r
                    elif rat < 0.1:
                        beta1 /= r
                        beta2 /= r
                    acc = 0
                if j % THIN[k] == 0:
                    rhoAsVec = paramToRhoCol(x)
                    rhoEst = rhoAsVec.reshape([D, D])
                    rho_BME += rhoEst
                    Fest[int(j / THIN[k])] = fidelity_f(rho_tar, rhoEst)
            samplerTime[m, k] = time.time() - tt
            # quantities of interest
            Fmean[m, k] = np.mean(Fest)
            Fstd[m, k] = np.std(Fest, ddof=1)
    return Fmean[0, 0], Fstd[0, 0], rho_BME / numSamp  # this gives rho_BME


def get_LS_and_MLE_rho_est(data, W, beta, D, nD):
    # Builds a density matrix from the vector Y
    def rho_from_Y(Y_est):
        rho_est = np.zeros([D, D], dtype=np.complex_)
        diagonal = np.append(Y_est[: D - 1], 1 - sum(Y_est[: D - 1]))
        np.fill_diagonal(rho_est, diagonal)  # Populate diagonal of rho

        index_i_list = np.triu_indices(D, 1)[0]
        index_j_list = np.triu_indices(D, 1)[1]
        for k in range(len(index_i_list)):  # Populate off-diagonals of rho
            index_i = index_i_list[k]
            index_j = index_j_list[k]
            rho_est[index_i, index_j] = Y_est[D - 1 + 2 * k] + 1j * Y_est[D + 2 * k]
            rho_est[index_j, index_i] = Y_est[D - 1 + 2 * k] - 1j * Y_est[D + 2 * k]

        return Qobj(rho_est)

    # Mapping
    C = np.matmul(-W, beta[:, 0])
    BETA = np.zeros([D**2 - 1, nD + 1])
    BETA[:, 0] = C
    BETA[:, 1 : nD + 1] = W

    # Experimental observables
    X_exp = np.zeros([1 + len(data)])
    X_exp[0] = 1
    X_exp[1:] = data

    # Estimate the state by applying the inverse map to the experimental data
    Y_est = np.zeros(nD)
    Y_est = np.matmul(BETA, X_exp)
    rho_est = rho_from_Y(Y_est)  # just a reshaping
    qRho_est = PSD_MLE_rho(rho_est)  # MLE of rho
    # Figures of merit

    return rho_est, qRho_est


def plot_wigner(rho, fig=None, ax=None):
    """
    Plot the Wigner function and the Fock state distribution given a density matrix for
    a harmonic oscillator mode.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if isket(rho):
        rho = ket2dm(rho)

    xvec = np.linspace(-1.8, 1.8, 200)

    W = wigner(rho, xvec, xvec)
    wlim = abs(W).max()

    ax.contourf(
        xvec,
        xvec,
        W,
        100,
        norm=mpl.colors.Normalize(-wlim, wlim),
        cmap=mpl.cm.get_cmap("RdBu"),
    )
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)

    return fig, ax
