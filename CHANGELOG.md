## Feature: LAMBDA2 Fit Method, N-Dimensional Support, and Core Bug Fixes

This major update introduces a new physical fitting model tailored for full scattering matrix analysis, alongside significant upgrades to the library's data handling capabilities, vectorization, and backward-compatible bug fixes.

### 1. The `LAMBDA2` Fitting Model (Core Physical Innovations)
A new class `LAMBDA2` has been added to `fit_methods`, extending the library's scope beyond traditional $\lambda/4$ notch resonators to accurately model transmission and reflection in $\lambda/2$ architectures and complex coupling environments.
* **Full $2 \times 2$ S-Parameter Simultaneous Fitting:** Unlike legacy methods that extract the quality factor strictly from a single parameter (typically $S_{21}$), `LAMBDA2` evaluates $S_{11}$, $S_{21}$, $S_{12}$, and $S_{22}$ altogether. This holistic approach leverages the entire scattering matrix to resolve asymmetries and cross-talk that single-parameter models often misinterpret.
* **Coupling Robustness and Phase Geometry ($n$-factor):** To account for the geometric coupling type and the parity of the resonance mode, the model introduces the parameter $n$, representing discrete $\pi/2$ phase shifts. A mathematical penalty term `3*(1 - cos(pi*n)^2)` acts as a soft constraint during optimization, forcing $n$ toward integer values. This allows the algorithm to automatically and robustly discover the correct phase alignment without manual hardcoding.
* **Advanced Error Propagation:** The `extractQi` function now features an analytical derivation of standard deviations for `inverseQi` and `Qi`, combining the covariance of $Q$, $Q_c$, and $\phi$ in quadrature for rigorous uncertainty quantification.

### 2. N-Dimensional Data Support and Vectorization
The mathematical backend has been significantly optimized to support large, multi-dimensional tensor arrays (e.g., power or temperature sweeps) natively.
* **Vectorized Band Partitioning:** Replaced slow `for` loops in `partitionFrequencyBand` with the C-optimized `np.heaviside` function, drastically reducing computation time for large datasets.
* **Multi-Dimensional Linear Regression:** Introduced `linregress_nd` in `utils.py` to calculate slopes and intercepts across specified axes for N-dimensional arrays.
* **Phase-Jump Correction:** The linear preprocessing routine now evaluates left and right data segments independently, applying a $2\pi$ unwrapping correction at the median frequency to prevent phase discontinuities from corrupting the fit.

### 3. API Enhancements and Legacy Bug Fixes
* **Method Shadowing Bug Fix:** Resolved a known upstream bug (`TypeError: 'bool' object is not callable`) where initializing `preprocess_circle=True` or `preprocess_linear=True` shadowed the internal class methods of the same name. The `Fitter` class now safely maps these legacy `kwargs` to new internal attributes (`circle_preprocessing`, `linear_preprocessing`), maintaining perfect backward compatibility while restoring access to the utility functions.
* **Extended Fit Output (`full_output`):** Added a `full_output=False` flag to `Fitter.fit()`. When set to `True`, the method returns both the extracted parameter dictionary and the complete `lmfit.ModelResult` object. This grants access to the covariance matrix, $\chi^2$ statistics, and MCMC flat chains (`emcee`), while safely defaulting to the legacy return type for existing scripts.
* **Graphing Standardization:** Capitalized axis labels in `plotres.py` (`Frequency (GHz)`, `Magnitude (dB)`, `Phase (rad)`) and ensured Touchstone/Network ingestion reliably maps to standard frequency units.

---

### Future Work & Roadmap
To fully capitalize on the mathematical groundwork laid by this update, the following improvements are planned for future releases:

1. **Translating `LAMBDA2` Innovations to Legacy Methods:** The soft-constrained phase shift parameter ($n$) and the simultaneous multi-parameter evaluation matrix can be adapted for the existing `DCM` model. Expanding `DCM` to ingest multiple S-parameters simultaneously will dramatically improve fitting accuracy for highly asymmetric notch resonators affected by impedance mismatches.
2. **Single-Parameter Fallback for `LAMBDA2`:** Modifying the `LAMBDA2` objective function to dynamically adjust its dimensionality. This will allow the model to operate on legacy datasets where only $S_{21}$ was recorded, enabling users to benefit from the new $\lambda/2$ physical modeling even when the full $2 \times 2$ scattering matrix is unavailable.
3. **Nonlinear Regime and Bifurcation Support:** Integrating the nonlinear kinetic inductance model detailed in the paper "New method for fitting complex resonance curve to study nonlinear superconducting resonators". The implementation strategy involves updating the `lmfit` objective functions (for both `LAMBDA2` and legacy methods) to replace the static resonance frequency $f_0$ with a dynamic, current-dependent $f_r(I)$. Architecturally, this requires introducing a new nonlinearity fitting parameter (e.g., `beta`) to the `lmfit.Parameters` object and computing the instantaneous frequency shift dynamically using the distance from the off-resonance point in the complex plane during the least-squares minimization iterations.
