# Demonstrating efficient and robust bosonic state reconstruction via optimized excitation counting
This repository contains all data and processing scripts for replicating the analysis presented in the manuscript "Demonstrating efficient and robust bosonic state reconstruction via optimized excitation counting" (arXiv:2403.03080).

## Abstract
Quantum state reconstruction is an essential element in quantum information processing. However, efficient and reliable reconstruction of non-trivial quantum states in the presence of hardware imperfections can be challenging. This task is particularly demanding for high-dimensional states encoded in continuous-variable (CV) systems, as many error-prone measurements are needed to cover the relevant degrees of freedom of the system in phase space. In this work, we introduce an efficient and robust technique for optimized reconstruction based on excitation number sampling (ORENS). We use a standard bosonic circuit quantum electrodynamics (cQED) setup to experimentally demonstrate the robustness of ORENS and show that it outperforms the existing cQED reconstruction techniques such as Wigner and Husimi Q tomography. Our investigation highlights that ORENS is naturally free of parasitic system dynamics and resilient to decoherence effects in the hardware. Finally, ORENS relies only on the ability to accurately measure the excitation number of the state, making it a versatile and accessible tool for a wide range of CV platforms and readily scalable to multimode systems. Thus, our work provides a crucial and valuable primitive for practical quantum information processing using bosonic modes.


## Workflow
Here, we describe step-by-step how to determine the measurement observables for the experiment, and how to transform the measurement outcomes to an optimal estimator for state reconstruction.

Note: "Q" and "ORENS" are used effectively interchangeably in filenames. 


### Before the experiment: optimizing measurement observables

Refer to Appendix D1 and D2 of the manuscript for more details.

For CV systems, when the state does not extend beyond a certain dimension $D$, its Hilbert space can be truncated. As such, only $D^2-1$ independent real parameters must be obtained from measurements for informational completeness. For ORENS, a minimal set of $D^2-1$ independent measurement observables are optimized for each dimension $D$ that can effectively reconstruct any arbitrary state of dimension $D$. For a given $D$, the set of measurements consists of $D^2-1$ independent displacements each followed by a photon number measurement. To obtain this set of displacements, we sweep over the excitation number $n \in [1, D-1]$, where for each set, we run a gradient-descent algorithm over the set of displacements $\{\alpha_k\}^{D^2-1}_{k=1}$ to minimize the condition number of the measurement matrix M.


These scripts below returns the set of optimized displacements for ORENS given a photon number to sample, and the set of optimized displacements for the benchmark Wigner tomography. These optimized displacements are obtained for a given dimension $D$, and they will allow us to reconstruct any arbitrary state bounded by the dimension $D$.

    optimize_observables_ORENS.py
    optimize_observables_W.py

We verify the set of displacements and linear inversion measurement matrices in these scripts below. They print the condition number and save the relevant linear inversion variables in <code>map_variables</code> to later build the least-squares (likely non-physical) estimator once data is collected.

    build_map_ORENS.py
    build_map_W.py

At this stage, the experiment can be conducted to collect measurement outcomes for data processing. 


### Data processing: inverting Born's rule and Bayesian estimation
Upon collection of measurement data, the linear inversion, maximum likehood estimation, and Bayesian estimation (Appendix D1 and D3) are computed to generate the the final state estimators and their fidelities. 

    apply_map_ORENS.ipynb
    apply_map_W.ipynb

### Simulation
QuTiP simulations of ORENS and Wigner with real experimental parameters are implemented in these scripts below. 

    simulate_ORENS.ipynb
    simulate_W.ipynb

## Additional file and folder descriptions

### exp_params.py

<code>which_ORENS_or_W</code> variable to be changed when processing and simulating results, to ensure that the correct measurement observables are used.

Stores all experimental Hamiltonian and decoherence parameters, optimized displacements, qubit thermal populations, etc.

### figs
Barebones figures used in the manuscript, generated from 
    
    plot_all_figures.ipynb

### helpers
Various helper functions. 

### map_variables
Linear inversion matrices computed from optimized displacements. Each dimension has one set of optimized measurement observables, so thus each dimension has one set of linear inversion map variables. 

### results_dimensions
Final estimators (both maximumum likelihood and Bayesian) and their fidelities for each dimension, both Wigner and ORENS (Q), and both simulation and experiment. 

### results_t2
Final estimators (both maximumum likelihood and Bayesian) and their fidelities across variable qubit dephasing (T2) times, for both Wigner and ORENS (Q), and both simulation and experiment. 

### target_states
Target states used for computing fidelity, calculated by simulating numerical pulses that are used in experiment with real experimental parameters.





