# Structural Forest Integrity Index (SFII)
## Research Implementation Roadmap
**Role:** Principal Research Scientist — Remote Sensing & Forest Ecology
**Document:** Forest Degradation Assessment Using Satellite Remote Sensing (March 2026)
**Status:** Pre-Implementation Planning

---

## Part I — Extracted Methodology

### 1. Core Problem Statement
NDVI recovers within 1–5 years post-disturbance, yet underlying biomass, canopy structure, and carbon stocks remain depleted for **decades**. This produces a **false-recovery signal** that causes standard degradation indices (notably CEI) to systematically underestimate persistent degradation in secondary-succession landscapes.

### 2. Index Architecture: Four Pillars

| Component | Symbol | Purpose |
|---|---|---|
| Spectral Recovery Trajectory | SRT | Rate & completeness of optical recovery vs. pre-disturbance baseline |
| Structural Biomass Proxy | SBP | Canopy height + SAR backscatter + Tasseled Cap Wetness |
| Disturbance Memory Function | DMF | Exponentially-decaying penalty for prior disturbances |
| Ecosystem Resilience Score | ERS | Bhattacharyya divergence from reference undisturbed forest |

### 3. Mathematical Formulations

**SRT:**
```
SRT(p,t) = [VI(p,t) - VI_pre(p)] / [VI_ref(p) - VI_pre(p)]
```
- VI_pre = 5-year median prior to disturbance year t₀
- VI_ref = median of spectrally similar intact pixels within 5 km
- Clipped: [0, 2]; values > 1 indicate spectral overshoot

**SBP:**
```
SBP(p,t) = 0.5·H_norm + 0.3·σ⁰_norm + 0.2·TCW_norm
```
- H_norm = normalised GEDI RH98 canopy height
- σ⁰_norm = normalised Sentinel-1 VV+VH backscatter
- TCW_norm = normalised Tasseled Cap Wetness
- Coefficients derived from regression against field AGB

**DMF:**
```
DMF(p,t) = Σᵢ [ mᵢ · exp(−λ · (t − tᵢ)) ]   for tᵢ ≤ t
```
- λ = 0.05–0.15 yr⁻¹ (biome-specific)
- mᵢ = disturbance magnitude (fraction of pre-disturbance VI lost)
- Multiple disturbance events summed, capped at 1.0

**ERS:**
```
ERS(p,t) = 1 − exp(−D_B(F_disturbed, F_reference))
```
- D_B = Bhattacharyya distance
- F = multivariate feature vector: [NDVI, NBR, TCW, H, σ⁰]
- Ranges [0,1]; 0 = identical to reference, 1 = maximally divergent

**SFII (Final Index):**
```
SFII(p,t) = 0.30·(1−SBP) + 0.25·DMF/DMF_max + 0.25·ERS + 0.20·max(0, SRT−SBP)
```
- Fourth term is the **False Recovery Penalty (FRP)**
- FRP > 0 only when spectral recovery exceeds structural recovery
- SFII: 0 (intact) → 1 (severely degraded)

**Degradation Classes:**
| SFII Range | Class |
|---|---|
| < 0.2 | Intact forest |
| 0.2–0.4 | Low degradation |
| 0.4–0.6 | Moderate degradation |
| 0.6–0.8 | High degradation |
| > 0.8 | Severe / recently disturbed |

**Carbon Emission Estimate:**
```
CE(p) = ∫[t₀→T] SFII(p,t) · AGB_ref(p) · 0.47 · dt
```

### 4. Processing Pipeline (7 Steps)

1. **Satellite Data Acquisition** — Sentinel-2 SR, Landsat C2L2, Sentinel-1 GRD, GEDI L2A, ICESat-2, MODIS MOD44B, ESA CCI LC
2. **Preprocessing** — Atmospheric correction, cloud masking, radiometric normalisation, monthly medoid compositing, coregistration, spectral index computation, SAR texture (GLCM)
3. **Index Calculation** — SRT, SBP, ERS per pixel per year
4. **Disturbance Detection** — LandTrendr on NBR (1985–present), MODIS MOD14 fire, Hansen GFC deforestation; construct per-pixel disturbance history D = {(tᵢ, mᵢ)}
5. **Recovery Modelling** — DMF computation; Chapman-Richards / logistic curve fitting; Time to Spectral Recovery (TSR) and Time to Structural Recovery (TSTR); Recovery Lag = TSTR − TSR
6. **Degradation Scoring** — SFII computation and annual maps
7. **Prediction** — LSTM + Random Forest + XGBoost stacked model; 5/10/20/50-year SFII forecasts; carbon emission estimates

### 5. Machine Learning Architecture

```
Satellite Time Series
        │
        ▼
[Feature Engineering]  ── 20+ features per pixel per year ──▶
        │
        ├──▶ LSTM (2 layers × 128 units, T=20 years)
        │         └── temporal embedding h_T
        │
        ├──▶ Random Forest (500 trees) → class probabilities P_RF
        │
        └──▶ XGBoost Regressor (stacked: P_RF + h_T + SRT + SBP + DMF + ERS)
                    └── SFII_pred (continuous, 0–1)
```

**Feature vector per pixel-year:**
- NDVI stats (mean, min, max, amplitude, CV) from 12 monthly composites
- NBR LandTrendr segment slope, magnitude, duration
- SRT at t, t−1, t−2, t−5
- SBP, DMF, ERS at t
- Sentinel-1 VV/VH seasonal amplitude
- GEDI RH50 and RH98 (or optical regression interpolation)
- Forest age proxy (years since last disturbance)
- Climate covariates: precipitation anomaly, mean temperature, SPEI-12 drought index

**Training data sources:**
- Field AGB plots: RAINFOR and CTFS network
- Airborne LiDAR transects: Amazonia, Borneo, Congo Basin
- VHR imagery interpretation: Planet, WorldView (expert-validated)

---

## Part II — Identified Assumptions

> [!WARNING]
> These assumptions are embedded in the paper but are not empirically validated within the document. Each must be explicitly tested during implementation.

### Mathematical Assumptions
1. **SBP coefficients (α=0.5, β=0.3, γ=0.2)** are stated to be "based on regression against field AGB" but no regression statistics, dataset, or geographic scope are cited. These values may not transfer across biomes.
2. **Lambda (λ = 0.05–0.15 yr⁻¹)** is listed as biome-specific but no lookup table or calibration method is provided. A tropical moist forest and a boreal forest will have vastly different λ values.
3. **SFII weights (0.30, 0.25, 0.25, 0.20)** are presented as defaults without a sensitivity analysis. The paper does not explore whether equal weighting or optimised weights improve performance.
4. **DMF_max normalisation denominator** is undefined — the paper uses `DMF/DMF_max` but does not define how DMF_max is computed (per pixel? per biome? theoretical maximum?).
5. **ERS Bhattacharyya distance** assumes the reference forest feature distribution is Gaussian and stationary across years and seasons — a strong assumption in heterogeneous tropical forests.
6. **SRT reference value VI_ref** assumes that "spectrally similar intact pixels within 5 km" are available and identifiable — may fail in heavily fragmented landscapes.
7. **Carbon fraction of 0.47** (biomass-to-carbon conversion) is applied uniformly; in reality this ranges from 0.44–0.50 across species and biomes (IPCC Tier 1 default).
8. **Recovery curves** (Chapman-Richards or logistic) are assumed to be sufficient to model post-disturbance SRT trajectories — no model selection criterion is described.

### Operational Assumptions
9. **GEDI availability** is assumed to be continuous and spatially complete — GEDI has irregular temporal sampling and is decommissioned post-2023. Gap-filling via optical regression is mentioned but not detailed.
10. **LandTrendr NBR threshold of 0.1** for flagging disturbances is fixed — no justification or regional calibration is presented.
11. **Google Earth Engine** is assumed as the sole processing environment. Computational costs at global scale are not addressed.
12. **Monthly composites with >30% cloud gap-fill** via linear interpolation — this may introduce significant artefacts in persistently cloudy regions (Borneo, Congo Basin, Amazon).

---

## Part III — Missing Implementation Details

> [!IMPORTANT]
> These are critical gaps that must be resolved before a reproducible, peer-reviewable implementation is possible.

### Missing Technical Specifications

| # | Gap | Impact |
|---|---|---|
| 1 | **DMF_max definition** — no formula or lookup provided | SFII formula is incomplete |
| 2 | **λ lookup table** by biome — only a range (0.05–0.15) given | DMF not reproducible across regions |
| 3 | **SBP coefficient calibration dataset** — regression data unidentified | SBP weights are unverified |
| 4 | **Reference forest pixel selection protocol** — no spatial or spectral criteria for selecting "undisturbed" reference pixels | SRT and ERS undefined operationally |
| 5 | **ERS feature normalisation** — are features z-scored, min-max scaled, or unstandardised before Bhattacharyya distance? | Numerical instability risk |
| 6 | **Covariance matrix estimation for ERS** — how many reference pixels needed? Are per-pixel or per-stratum covariances used? | ERS computation ambiguous |
| 7 | **GEDI gap-filling regression** — "optical regression" mentioned but no equation or model type given | SBP pipeline is incomplete |
| 8 | **LSTM training labels** — what constitutes the ground truth y for the LSTM? AGB class? SFII class? Both are mentioned in different sections | Training objective undefined |
| 9 | **Train/validation/test split strategy** — no temporal or spatial cross-validation scheme described (spatial autocorrelation risk) | Model evaluation not reproducible |
| 10 | **Recovery curve model selection** — Chapman-Richards vs. logistic: when to use which? No AIC/BIC criterion given | TSR/TSTR estimates may vary arbitrarily |
| 11 | **Disturbance history initialisation** — how to handle pixels with no detected LandTrendr disturbance (pristine forest)? DMF = 0? | Edge case unhandled |
| 12 | **SAR preprocessing protocol** — orbit direction, speckle filter (kernel size?), terrain correction DEM source unspecified | Sentinel-1 SBP not reproducible |
| 13 | **Cloud gap-fill thresholding** — exactly how is linear temporal interpolation applied for >30% cloud coverage months? | Data quality not defined |
| 14 | **Medoid compositing definition** — "median" and "medoid" are used interchangeably in the text; these are mathematically distinct | Preprocessing inconsistency |
| 15 | **SFII temporal resolution** — is SFII computed annually or monthly? The ML feature vector implies annual but pseudocode implies per-year | Ambiguous temporal cadence |

### Missing Validation Information
- No field plot dataset is named for the "n > 500 sites" component validation target
- No geographic study area is defined — the GEE code uses Borneo as an example but the framework claims global applicability
- Accuracy targets (r² > 0.75, OA > 85%) have no baseline comparison to existing methods in the same study area

---

## Part IV — Datasets Required

### Primary Satellite Data

| Dataset | Source | Resolution | Coverage | Access |
|---|---|---|---|---|
| Sentinel-2 L2A SR | ESA Copernicus / GEE | 10–20 m | 2017–present | Free (GEE) |
| Landsat Collection 2 L2 | USGS / GEE | 30 m | 1985–present | Free (GEE) |
| Sentinel-1 GRD (VV+VH) | ESA Copernicus / GEE | 10 m | 2014–present | Free (GEE) |
| GEDI Level-2A | NASA / LP DAAC | 25 m footprint | 2019–2023 | Free (LPDAAC) |
| ICESat-2 ATL08 | NASA / NSIDC | ~11 m | 2018–present | Free (NSIDC) |
| MODIS MOD44B VCF | NASA / GEE | 250 m | 2000–present | Free (GEE) |
| MODIS MOD14 Fire | NASA / GEE | 1 km | 2000–present | Free (GEE) |
| Hansen GFC Annual Loss | Hansen et al. / GEE | 30 m | 2000–present | Free (GEE) |
| ESA CCI Land Cover | ESA / GEE | 300 m | 1992–2020 | Free (GEE) |

### Auxiliary / Climate Data

| Dataset | Source | Variable | Access |
|---|---|---|---|
| CHIRPS / ERA5 | UCSB / ECMWF | Precipitation anomaly | Free (GEE / CDS) |
| ERA5-Land | ECMWF | Mean temperature | Free (CDS) |
| SPEI Global Drought Monitor | CSIC | SPEI-12 drought index | Free (download) |
| SRTM / Copernicus DEM | NASA / ESA | Elevation / terrain correction | Free (GEE) |

### Ground Truth / Training Data

| Dataset | Source | Use |
|---|---|---|
| RAINFOR Amazon forest plots | RAINFOR network | AGB ground truth, disturbance history |
| CTFS ForestGEO plots | Smithsonian | AGB ground truth, multi-biome |
| TropiSAR / AfriSAR airborne LiDAR | ESA | Structural ground truth (Amazonia, Congo) |
| Planet NICFI basemaps | Planet Labs | VHR imagery for degradation interpretation |
| WorldView / GeoEye archive | Maxar | VHR validation sites |
| Global Biomass (Avitabile/Saatchi) | ESA CCI Biomass | AGB_ref prior for carbon computation |

---

## Part V — Milestone Roadmap

---

## Phase 1 — Literature

**Objective:** Establish theoretical foundation, identify peer-reviewed methodological precedents, and scope out state-of-the-art for each SFII component.

**Duration: 6 weeks**

### Milestones
- [ ] Systematically review all 11 key references cited in the paper (Kennedy 2010, Verbesselt 2010, Venter 2016, Dubayah 2020, Souza 2020, Zhu 2017, Hansen 2013, Hochreiter 1997, Chen 2016, Breiman 2001, Xu 2020)
- [ ] Extend literature review to 40–60 additional papers covering: GEDI biomass modelling, Bhattacharyya distance in RS, recovery lag quantification, SAR forest biomass, LSTM in time-series RS
- [ ] Review LandTrendr parameter sensitivity studies — identify biome-specific NBR thresholds
- [ ] Review Chapman-Richards and logistic recovery curve applications in forest ecology
- [ ] Review REDD+ MRV methodological requirements (e.g., UNFCCC Good Practice Guidance)
- [ ] Produce annotated bibliography (≥60 sources) in Zotero or Mendeley
- [ ] Write a 5-page critical synthesis: where SFII sits relative to state-of-the-art, and which gaps this study fills

**Key Questions to Resolve:**
- What are published λ values (DMF decay constant) for tropical, temperate, and boreal biomes?
- Are there published SBP-type formulations using GEDI + SAR + TCW with reported regression coefficients?
- What spatial cross-validation strategies are used in large-scale RS classification studies to avoid autocorrelation?

---

## Phase 2 — Data Collection (Automated via GEE Pipeline)

**Objective:** Automatically generate, compute, and export all primary satellite and ancillary datasets required for the study area using Google Earth Engine (GEE).

**Duration: 1 week** (Fully Automated)

### Study Area Decision
> [!IMPORTANT]
> The Nilgiris district (India) has been selected as the primary Region of Interest (ROI) due to the availability of the existing 2018-2024 Sentinel-2 dataset. 

### Milestones (Automated Workflow)
- [x] Implement GEE Python API authentication and configuration module (`config.py`).
- [x] Implement automated DEM & Terrain generation (`01_dem_terrain.py`) for Copernicus DEM, SRTM, Elevation, Slope, Aspect, and Curvature.
- [x] Implement automated Sentinel-1 SAR processing (`02_sentinel1_sar.py`) for VV, VH, and VV/VH composites.
- [x] Implement automated LandTrendr processing (`03_landtrendr.py`) to extract Disturbance and Recovery metrics.
- [x] Implement automated distance metrics computation (`04_distance_metrics.py`) for Distance to Roads and Settlements.
- [x] Implement automated climate data retrieval (`05_climate_data.py`) for Rainfall, SPI, and VPD.
- [x] Orchestrate full data generation via a unified master script (`main.py`).

**Documentation of Limitations:**
> [!NOTE]
> All 20 required variables were successfully categorized as either Computable (Category B) or Downloadable (Category C) via GEE. Manual collection (Category D) was entirely avoided. GEDI and ICESat-2 (if required for structural priors) can also be fetched via GEE's community datasets or equivalent proxies, but the core 20 variables are fully implemented in the pipeline.

**Compute Infrastructure Decision:**
- Data extraction, feature generation, and export handled entirely by Google Earth Engine.
- Final outputs are exported to Google Drive / local storage for subsequent ML training in Python.

---

## Phase 3 — Preprocessing

**Objective:** Produce a clean, co-registered, cloud-free, multi-sensor analysis-ready dataset.

**Duration: 6 weeks**

### Milestones

**Optical (Sentinel-2 / Landsat):**
- [ ] Implement s2cloudless probabilistic cloud masking for Sentinel-2 in GEE
- [ ] Implement Fmask for Landsat in GEE; apply C2 QA_PIXEL band masking
- [ ] Build monthly medoid composites (resolve text ambiguity — implement true spectral medoid, not band-wise median) using 30-day windows
- [ ] Apply linear temporal interpolation for pixels with >30% monthly cloud gap; document gap-fill percentage map
- [ ] Implement relative radiometric normalisation (histogram matching to pseudo-invariant targets for inter-sensor consistency between Landsat and Sentinel-2)
- [ ] Compute spectral indices per composite: NDVI, NBR, EVI2, TCW (Sentinel-2 coefficients from Crist 1985), NDWI, NDBSI

**SAR (Sentinel-1):**
- [ ] Apply thermal noise removal, radiometric calibration, terrain flattening using Copernicus DEM (30 m)
- [ ] Apply speckle filtering — Lee Sigma filter (5×5 kernel) in SNAP or GEE
- [ ] Convert to dB: σ⁰(dB) = 10·log₁₀(σ⁰)
- [ ] Separate ascending / descending orbits; create seasonal composites
- [ ] Compute GLCM texture metrics (contrast, homogeneity, entropy, correlation) in 5×5 kernel using GEE `glcmTexture()`

**LiDAR (GEDI / ICESat-2):**
- [ ] Filter GEDI L2A: quality_flag=1, degrade_flag=0, beam_sensitivity threshold
- [ ] Reproject GEDI footprints to study area CRS (WGS84 UTM)
- [ ] Interpolate GEDI RH98 to continuous raster using kriging or RF regression from Sentinel-2 predictors (document this step carefully — this is a critical gap in the paper)
- [ ] Validate interpolated canopy height against independent ICESat-2 ATL08 samples

**Coregistration:**
- [ ] Coregister all datasets to 10 m WGS84 UTM grid using GDAL (gdalwarp with bilinear resampling for continuous, nearest-neighbour for categorical)
- [ ] Verify coregistration accuracy against stable reference points (< 0.5 pixel RMSE target)

**Quality Assurance:**
- [ ] Generate data availability summary maps per sensor per year (% valid pixels)
- [ ] Histogram inspection for all composited indices; flag anomalous distributions
- [ ] Archive preprocessed rasters in cloud-optimised GeoTIFF (COG) format

---

## Phase 4 — Feature Engineering

**Objective:** Construct the complete per-pixel, per-year feature matrix for SFII computation and ML training.

**Duration: 5 weeks**

### Milestones

**Baseline Construction:**
- [ ] Compute per-pixel pre-disturbance NDVI/NBR baseline: 5-year median prior to first detected disturbance (or prior to study start for undisturbed pixels)
- [ ] Define reference forest pixel protocol: intact pixels within 5 km radius, matching biome/climate stratum, meeting minimum NDVI/canopy height threshold — document selection criteria explicitly
- [ ] Compute VI_ref (median of reference forest pixels) per pixel

**Disturbance Detection:**
- [ ] Run LandTrendr algorithm on annual NBR composites (1985–2024) in GEE
- [ ] Apply disturbance flagging: NBR drop > 0.1 within single LandTrendr segment
- [ ] Record (tᵢ, mᵢ) per disturbance event per pixel; construct disturbance history DB
- [ ] Overlay MODIS MOD14 fire detections → attribute fire-origin disturbances
- [ ] Overlay Hansen GFC annual loss → attribute deforestation-origin disturbances
- [ ] Validate a random sample of 200 disturbance detections against Landsat visual inspection and Planet NICFI imagery

**SFII Component Computation:**
- [ ] Compute SRT(p,t) for all pixels, all years
- [ ] Compute SBP(p,t): normalise GEDI RH98, Sentinel-1 σ⁰, TCW per pixel; apply α=0.5, β=0.3, γ=0.2 (flag these coefficients for Phase 7 sensitivity analysis)
- [ ] Compute DMF(p,t): **establish λ per biome** from literature review (Phase 1); resolve DMF_max definition — propose: DMF_max = theoretical maximum for a single disturbance of magnitude 1.0 at t=0
- [ ] Compute ERS(p,t): construct reference feature distribution [NDVI, NBR, TCW, H, σ⁰] per pixel; compute Bhattacharyya distance; normalise features (z-score recommended); document covariance matrix estimation strategy

**ML Feature Matrix:**
- [ ] Assemble 20+ feature vector per pixel per year (see Section 7.1 of paper)
- [ ] Compute NDVI annual statistics: mean, min, max, amplitude, CV from 12 monthly composites
- [ ] Extract LandTrendr NBR segment parameters: slope, magnitude, duration
- [ ] Compute forest age proxy: years since last LandTrendr disturbance
- [ ] Merge ERA5/CHIRPS climate covariates (precipitation anomaly, temperature, SPEI-12)
- [ ] Construct 3D array: [pixels × years × features] for LSTM input
- [ ] Generate degradation class labels (5-class) using rule-based SFII thresholds and expert VHR interpretation

---

## Phase 5 — SFII Development

**Objective:** Implement the full analytical SFII computation pipeline, resolve all identified mathematical gaps, and produce baseline annual SFII maps.

**Duration: 8 weeks**

### Milestones

**Mathematical Resolution (Critical):**
- [ ] Define and justify **DMF_max**: test three formulations — (a) per-pixel maximum observed DMF, (b) theoretical maximum (λ·e⁻¹)⁻¹, (c) fixed biome-level constant — select based on sensitivity analysis
- [ ] Establish **λ lookup table** by biome (tropical moist, tropical dry, temperate broadleaf, boreal) from Phase 1 literature review; if unavailable, calibrate against AGB recovery curves from field data
- [ ] Validate **SBP coefficients** against field AGB measurements from RAINFOR/CTFS: run OLS regression of H_norm, σ⁰_norm, TCW_norm against measured AGB/ha; report adjusted R², RMSE, and 95% CI for each coefficient
- [ ] Define **ERS normalisation protocol**: implement z-score normalisation of [NDVI, NBR, TCW, H, σ⁰] using reference forest distribution as the standardisation basis
- [ ] Resolve **medoid vs. median** compositing: implement true spectral-angle-based medoid
- [ ] Resolve **SAR preprocessing orbit selection**: test ascending vs. descending vs. mosaic; select based on speckle variance comparison

**SFII Implementation:**
- [ ] Implement full SRT → SBP → DMF → ERS → FRP → SFII pipeline in Python (NumPy/Xarray) on local data
- [ ] Implement parallel GEE version for scalable cloud processing
- [ ] Produce annual SFII maps for study area (2018–2024)
- [ ] Produce annual FRP maps to identify false-recovery hotspots
- [ ] Produce Recovery Lag maps (TSTR − TSR) for all disturbed pixels
- [ ] Fit Chapman-Richards recovery curves to per-pixel SRT time series; extract TSR and TSTR
- [ ] Compute carbon emission estimates per pixel using AGB_ref from ESA CCI Biomass

**CEI Baseline Comparison:**
- [ ] Implement CEI for the same study area and period
- [ ] Produce side-by-side SFII vs. CEI maps for 3 representative disturbed sites
- [ ] Quantify false-recovery magnitude: percentage of pixels where CEI < 0.2 but SFII > 0.5

**Weight Sensitivity Analysis:**
- [ ] Run SFII with ±0.05 perturbations to all four weights
- [ ] Report SFII rank-order stability across weight perturbations (Spearman ρ)
- [ ] Recommend revised weights if default weights show poor correlation with AGB ground truth

---

## Phase 6 — Machine Learning

**Objective:** Train, tune, and evaluate the LSTM + Random Forest + XGBoost stacked ML pipeline for automated SFII prediction.

**Duration: 8 weeks**

### Milestones

**Training Data Preparation:**
- [ ] Assemble ground-truth dataset from RAINFOR + CTFS plots + airborne LiDAR + VHR expert interpretation
- [ ] Ensure stratified sampling: tropical/temperate/boreal × disturbance type × forest age class
- [ ] Define LSTM labels explicitly: binary (degraded/intact) or 5-class degradation category
- [ ] Implement spatial block cross-validation (minimum block size = 2× spatial autocorrelation range) to prevent information leakage

**LSTM Training:**
- [ ] Implement ForestLSTM: 2-layer LSTM, 128 hidden units, dropout=0.3, input_dim determined by feature count
- [ ] Determine sequence length T: test T=10, T=20 years; select based on validation AUC
- [ ] Train with Adam (lr=1e-3), BCELoss; implement early stopping (patience=10 epochs)
- [ ] Log training curves; report final validation AUC and F1 per class
- [ ] Extract final hidden state h_T as 128-dimensional temporal embedding

**Random Forest Training:**
- [ ] Train RandomForestClassifier (500 trees, max_depth=12) on [F_static + h_T]
- [ ] Compute feature importance: identify top-10 most discriminative variables
- [ ] Report OOB accuracy and confusion matrix

**XGBoost Stacking:**
- [ ] Stack RF class probabilities P_RF with LSTM embedding h_T and SFII component scores [SRT, SBP, DMF, ERS]
- [ ] Train XGBRegressor (n_estimators=400, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8)
- [ ] Tune hyperparameters via Bayesian optimisation (Optuna or Hyperopt)
- [ ] Report final SFII_pred: RMSE, MAE, R² vs. analytical SFII

**Forecast Models:**
- [ ] Implement 5/10/20/50-year SFII forecasting under three disturbance scenarios: business-as-usual, reduced disturbance (30% reduction), intensified disturbance (+30%)
- [ ] Report forecast uncertainty (prediction intervals from XGBoost quantile regression)

---

## Phase 7 — Validation

**Objective:** Rigorously validate all SFII components and the composite index against independent field measurements and VHR imagery.

**Duration: 6 weeks**

### Milestones

**Component Validation (n > 500 sites target):**
- [ ] **SBP validation**: regress computed SBP against field-measured AGB (t C/ha) from RAINFOR/CTFS plots; report R², RMSE, bias
- [ ] **SRT validation**: compare SRT recovery trajectory shape against airborne LiDAR AGB recovery trajectories from known-age secondary forests
- [ ] **ERS validation**: compare ERS divergence with expert-assigned degradation severity scores from VHR interpretation (Spearman ρ target > 0.70)
- [ ] **DMF validation**: test whether DMF correctly retains high values at sites with documented recent disturbance events; compute false-negative rate

**SFII Index Validation:**
- [ ] Correlate annual SFII with independently estimated AGB change (R² target > 0.75)
- [ ] Compare SFII degradation classes against expert VHR interpretation at n ≥ 200 test sites (target overall accuracy > 85%, Cohen's κ > 0.75)
- [ ] Compute producer's and user's accuracies per class (confusion matrix)
- [ ] Compare SFII vs. CEI false-recovery detection: compute precision and recall for identifying "spectral recovery without structural recovery" (FRP > 0.2 and field-confirmed degraded)

**Temporal Prediction Validation:**
- [ ] Use 2018–2021 as training period; holdout 2022–2024 as temporal test set
- [ ] Forecast SFII for 2022–2024 using ML model trained on 2018–2021
- [ ] Report RMSE and bias for forecasted vs. observed SFII at t+1, t+2, t+3

**Weight Sensitivity Revisit:**
- [ ] Apply optimised weights from Phase 5 sensitivity analysis
- [ ] Re-run validation with optimised weights; compare R² and OA improvements
- [ ] Report final recommended weight set

**Uncertainty Quantification:**
- [ ] Propagate input data uncertainty (cloud gap-fill, GEDI interpolation error) through to SFII confidence intervals
- [ ] Map per-pixel SFII uncertainty (coefficient of variation) for the study area

---

## Phase 8 — Scientific Paper

**Objective:** Produce a submission-ready manuscript for a Q1 remote sensing journal.

**Duration: 12 weeks**

### Target Journals (in order of preference)

| Journal | IF | Scope fit |
|---|---|---|
| Remote Sensing of Environment | ~13 | Highest — primary RS methods journal |
| ISPRS Journal of Photogrammetry | ~12 | Strong — methods-forward |
| Global Change Biology | ~11 | If emphasis is ecological |
| Remote Sensing (MDPI) | ~5 | Faster review; open access |

### Manuscript Structure

- [ ] **Title:** *The Structural Forest Integrity Index (SFII): A Multi-Source Time-Aware Framework for Detecting Persistent Forest Degradation Beyond Spectral Recovery*
- [ ] **Abstract** (250 words): problem, method, key results, implications
- [ ] **Section 1 — Introduction:** false-recovery problem, gap in existing methods, contributions
- [ ] **Section 2 — Study Area and Data:** justified AOI selection, all datasets with specifications
- [ ] **Section 3 — Methodology:** SRT, SBP, DMF, ERS, SFII formula derivation; ML pipeline
- [ ] **Section 4 — Results:**
  - Annual SFII maps (3+ years)
  - SFII vs. CEI false-recovery comparison maps
  - Component validation statistics (R², RMSE, OA)
  - Feature importance from Random Forest
  - Forecast SFII maps
- [ ] **Section 5 — Discussion:**
  - Superiority over CEI for false-recovery detection
  - GEDI gap-filling limitations
  - λ biome sensitivity
  - Weight uncertainty
  - Transferability to global application
  - REDD+ MRV integration pathway
- [ ] **Section 6 — Conclusion**
- [ ] **Supplementary Material:** Full Python code (GitHub), all preprocessing scripts, extended validation tables
- [ ] **Data Availability Statement:** GEE scripts, all open datasets cited with DOIs
- [ ] **Code Repository:** Publish full pipeline on GitHub with DOI (Zenodo) before submission

### Publication Milestones
- [ ] Internal draft completed and circulated to co-authors
- [ ] Peer review by external colleague (senior RS scientist) before submission
- [ ] Address all reviewer comments; produce response letter
- [ ] Submit to primary journal; track under review
- [ ] Revise and resubmit if required
- [ ] Acceptance and publication

---

## Summary Timeline

```
Month  1–2   Phase 1: Literature Review
Month  2–3   Phase 2: Data Collection          (parallel with Phase 1)
Month  3–5   Phase 3: Preprocessing
Month  5–7   Phase 4: Feature Engineering
Month  6–9   Phase 5: SFII Development
Month  8–11  Phase 6: Machine Learning
Month  11–13 Phase 7: Validation
Month  13–18 Phase 8: Scientific Paper
─────────────────────────────────────────
Total:        ~18 months to submission
```

---

## Critical Path Items

> [!CAUTION]
> The following issues must be resolved before Phase 5 can proceed. Failure to resolve them will produce a non-reproducible SFII implementation.

1. **DMF_max definition** — resolve via Phase 1 literature review or first-principles derivation
2. **λ biome lookup table** — derive from published AGB recovery curves or field calibration
3. **SBP coefficient validation** — run regression against named field dataset before committing weights
4. **GEDI gap-filling regression** — implement and validate before using SBP in SFII
5. **Reference forest pixel selection protocol** — must be explicit, reproducible, and documented
6. **Spatial cross-validation design** — implement before ML training to prevent inflated accuracy estimates
7. **Study area selection** — drives all subsequent data acquisition decisions; decide at Phase 2 start
