import numpy as np
from qutip import *


def cavity_target_state(name, cdim):
    if len(name) == 5:  # fock0, fock1, fock2...
        target = fock(cdim, int(name[-1]))
        return target

    elif len(name) == 6:  # fock01, fock02, fock34...
        target = (fock(cdim, int(name[-2])) + fock(cdim, int(name[-1]))).unit()
        return target

    elif len(name) == 7:  # fock0i1, fock0i2, fock3i4...
        target = (fock(cdim, int(name[-3])) + 1j * fock(cdim, int(name[-1]))).unit()
        return target

    elif len(name) == 8:  # fock0-i1, fock0-i2
        target = (fock(cdim, int(name[-4])) - 1j * fock(cdim, int(name[-1]))).unit()
        return target
    elif "cat-eve" in name:  # cat-eve-1, cat-odd-1, cat-nop-1
        target = (
            coherent(cdim, int(name[-1])) + coherent(cdim, -1 * int(name[-1]))
        ).unit()
        return target
    elif "cat-odd" in name:
        target = (
            coherent(cdim, int(name[-1])) - coherent(cdim, -1 * int(name[-1]))
        ).unit()
        return target
    elif "cat-nop" in name:
        target = (
            coherent(cdim, int(name[-1])) + 1j * coherent(cdim, -1 * int(name[-1]))
        ).unit()
        return target
    elif "cat-nmp" in name:
        target = (
            coherent(cdim, int(name[-1])) - 1j * coherent(cdim, -1 * int(name[-1]))
        ).unit()
        return target
    else:
        print("State ", name, "invalid!")
        pass


def Y_target(state_name, states_directory, qdim, cdim):
    # Target states
    
    file = np.load(states_directory + "/" + state_name + ".npz", "r")
    # 6 is to remove "pulse_"
    rho_tar = Qobj(file["rho"], dims=[[qdim, cdim], [qdim, cdim]])
    return rho_tar
