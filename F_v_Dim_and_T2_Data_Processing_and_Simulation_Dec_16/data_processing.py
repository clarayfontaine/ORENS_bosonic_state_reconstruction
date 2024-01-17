import numpy as np
import h5py
from pathlib import Path


# Sorting helper methods
def find_avg_photon_number(state):
    if len(state) == 5:  # fock0, fock1, fock2...
        return int(state[-1])

    elif len(state) == 6:  # fock01, fock02, fock34...
        return np.mean([int(state[-2]), int(state[-1])])

    elif len(state) == 7:  # fock0i1, fock0i2, fock3i4...
        return np.mean([int(state[-3]), int(state[-1])])

    elif len(state) == 8:  # fock0-i1, fock0-i2
        return np.mean([int(state[-4]), int(state[-1])])

    elif len(state) == 9:  # cat-eve-1, cat-odd-1, cat-nop-1
        return 10

    else:
        print("State ", state, "invalid!")
        pass


def sort_state_list_by_photon_number(state_list):
    """Takes in an alphabetically sorted state_list"""
    state_list = np.sort(np.asarray(state_list))
    # state_list = state_list[np.array(state_list).argsort()]
    state_avg_photon_number = np.asarray(
        [find_avg_photon_number(x) for x in state_list]
    )

    ind = np.lexsort((state_list, state_avg_photon_number))

    sorted = np.asarray([(state_list[i], state_avg_photon_number[i]) for i in ind])
    return ind, sorted[:, 0]


def extract_states_from_exp_data(directory, D):
    state_list = []
    for file in Path(directory).glob("*"):
        file_name = file.stem[::]
        state_name = "".join(file_name.split(f"D={D}_grape_")[1].split("_point")[0])
        if state_name not in state_list:
            state_list.append(state_name)

    # Sort the state list in increasing photon number
    ind, state_list_sorted = sort_state_list_by_photon_number(state_list)

    return state_list_sorted


def extract_states_from_exp_data_T2(directory, D):
    state_list = []
    for file in Path(directory).glob("*"):
        file_name = file.stem[::]
        state_name = "".join(file_name.split(f"_grape_")[1].split("_point")[0])
        if state_name not in state_list:
            state_list.append(state_name)

    # Sort the state list in increasing photon number
    ind, state_list_sorted = sort_state_list_by_photon_number(state_list)

    return state_list_sorted


def post_selection(filepath):
    file = h5py.File(filepath, "r")
    data = file["data"]
    # threshold = file["operations/readout_pulse"].attrs["threshold"]
    threshold = 2.450955206532979e-05
    I = data["I"][::]
    x = data["x"][::]  # Assume all sweep points are the same
    # sweep_points = int(np.shape(x)[0])  # 5
    flat_data = np.array(I.flatten())
    I_first = flat_data[0::2]
    I_second = flat_data[1::2]

    I_first[I_first > threshold] = 1
    I_first[I_first != 1] = 0
    I_second[I_second > threshold] = 1
    I_second[I_second != 1] = 0

    select_mask = np.where(I_first == 0)[0]  # first_selection

    thrownrate = 100 * (len(I_first) - len(select_mask)) / len(I_first)

    selected_data = I_second[select_mask]

    return selected_data.mean()


def post_selection_W(filepath):
    file = h5py.File(filepath, "r")
    data = file["data"]

    threshold = 2.9011118282404986e-05
    I = data["I"][::]

    flat_data = np.array(I.flatten())
    flat_data[flat_data > threshold] = 1
    flat_data[flat_data != 1] = 0

    I_first = flat_data[0::4]
    I_second = flat_data[1::4]

    I_first_m = flat_data[2::4]
    I_second_m = flat_data[3::4]

    select_mask = np.where(I_first == 0)[0]  # first_selection
    select_mask_m = np.where(I_first_m == 0)[0]

    thrownrate = 100 * (len(I_first) - len(select_mask)) / len(I_first)
    thrownrate_m = 100 * (len(I_first_m) - len(select_mask_m)) / len(I_first_m)
    # print('{} % data are thrown'.format(thrownrate))

    selected_data = I_second[select_mask]
    selected_data_m = I_second_m[select_mask_m]

    return selected_data.mean(), selected_data_m.mean()

