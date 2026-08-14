# Structural Forest Integrity Index (SFII): Mathematical Formulation and Rationale

The Structural Forest Integrity Index (SFII) is a comprehensive, multi-dimensional metric designed to quantify subtle forest degradation by coupling historical spectral trajectories with present-day structural measurements. The SFII is constrained strictly to the interval $[0, 1]$, where $0$ indicates an intact reference forest and $1$ represents severe structural and ecological collapse.

## 1. Structural Biomass Proxy (SBP)
The SBP quantifies the physical structural state of the forest relative to an intact baseline.

$$ SBP_t = \alpha H_{norm} + \beta \Sigma^0_{norm} + \gamma TCW_{norm} $$

**Variables:**
- $H_{norm}$: Normalized Canopy Height (e.g., from GEDI/ALS), $H_{norm} \in [0, 1]$.
- $\Sigma^0_{norm}$: Normalized SAR backscatter (e.g., L-band ALOS PALSAR), capturing branch and trunk volume.
- $TCW_{norm}$: Tasseled Cap Wetness, a proxy for canopy moisture and density.
- $\alpha, \beta, \gamma$: Empirical weighting coefficients ($\alpha+\beta+\gamma=1$).

**Rationale:** No single sensor perfectly captures 3D forest structure. By fusing LiDAR (height), SAR (volume), and optical (moisture), the SBP becomes robust against saturation limits in high-biomass tropical forests.

## 2. Spectral Recovery Trajectory (SRT)
The SRT assesses the degree to which a pixel has recovered its spectral baseline post-disturbance.

$$ SRT_t = \frac{VI_{current, t} - VI_{pre}}{VI_{ref} - VI_{pre}} $$

**Variables:**
- $VI_{current, t}$: Vegetation Index (e.g., NBR, EVI) at time $t$.
- $VI_{pre}$: The VI value immediately prior to the detected disturbance.
- $VI_{ref}$: The VI value of a completely intact, undisturbed reference pixel of the same eco-class.

**Rationale:** Spectral indices often recover faster than physical biomass. The SRT isolates the *relative* return to a pre-disturbance state. An $SRT \approx 1$ implies full spectral recovery, while $SRT \approx 0$ implies a persistent degraded state.

## 3. Disturbance Memory Function (DMF)
Forest ecosystems possess an ecological "memory" where historical, compounded disturbances hinder current resilience. 

$$ DMF_t = \sum_{i} m_i \cdot e^{-\lambda (t - t_i)}, \quad \forall t_i \leq t $$

**Variables:**
- $m_i$: The magnitude of the disturbance event at time $t_i$.
- $\lambda$: An ecological decay constant representing natural recovery rates (e.g., $\lambda=0.08$ corresponds to a ~12-year recovery half-life).

**Rationale:** The exponential decay models the gradual dissipation of a disturbance's impact over time. Multiple minor, high-frequency disturbances (e.g., repeated selective logging) will compound, elevating the DMF and penalizing the final index.

## 4. Ecosystem Resilience Score (ERS)
The ERS measures the multivariate statistical divergence of the disturbed pixel's features from the intact reference distribution.

$$ ERS_t = 1 - e^{-D_B(f_t, P_{ref})} $$

Where $D_B$ is the Bhattacharyya distance, adapted with Ridge Regularization for numerical stability:
$$ D_B = \frac{1}{8} (f_t - \mu_{ref})^T (\Sigma_{ref} + \lambda_{ridge} I)^{-1} (f_t - \mu_{ref}) $$

**Rationale:** True resilience is not just mean-reversion, but a return to the natural variance envelope of the ecosystem. The Mahalanobis-like term strictly penalizes pixels whose feature vectors ($f_t$) drift outside the covariance structure ($\Sigma_{ref}$) of intact forests.

## 5. False Recovery Penalty (FRP)
A critical innovation of the SFII is identifying "green desert" phenomena—where fast-growing pioneer species or invasive weeds cause spectral indices to recover rapidly, while the actual structural biomass remains depleted.

$$ FRP_t = \max(0, SRT_t - SBP_t) $$

**Rationale:** If $SRT$ is high (the pixel looks green from space) but $SBP$ is low (the 3D structure is missing), the $FRP$ activates. The $\max$ function mathematically ensures that the penalty is *only* applied when spectral recovery is deceptive.

## 6. The Final SFII Integration
The sub-components are linearly integrated to form the final index:

$$ SFII_t = w_1(1 - SBP_t) + w_2\left(\frac{DMF_t}{DMF_{max}}\right) + w_3 ERS_t + w_4 FRP_t $$

**Weighting & Normalization:**
- $(1 - SBP_t)$: Inverts the structural proxy so that loss of structure increases the degradation score.
- $DMF_{max}$: Normalization scalar based on the theoretical maximum disturbance capacity of the landscape.
- $w_1, w_2, w_3, w_4$: Ecological weights summing to $1.0$, typically calibrated via sensitivity analysis to the specific biome.

**Boundary Enforcement:**
Due to transient anomalies (e.g., severe localized flooding skewing $FRP$), the final computational output is strictly clamped:
$$ \text{SFII}_{final} = \max(0, \min(1, SFII_t)) $$
