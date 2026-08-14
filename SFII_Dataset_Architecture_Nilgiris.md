# SFII Dataset Architecture
## Nilgiri Biosphere Reserve — Implementation Blueprint
**Prepared by:** Principal Research Scientist
**Study Area:** Nilgiri Biosphere Reserve, Western Ghats, India
**AOI Bounding Box:** 76.0°E – 77.5°E, 10.5°N – 12.0°N
**Area:** ~5,520 km² (core + buffer + transition zones)
**Forest Types:** Tropical wet evergreen, Semi-evergreen, Moist deciduous, Shola grassland-forest mosaic, Dry deciduous
**Temporal Scope:** Historical archive 1985–2000 (Landsat only); Main analysis 2018–2024; Forecast horizon 2025–2035
**Coordinate Reference System:** WGS84 / UTM Zone 43N (EPSG:32643)
**Target Resolution:** 10 m (analysis grid); 30 m (historical archive)

---

## Section 1 — Dataset Inventory

> Complete specification of every satellite and ancillary dataset required for SFII computation over the Nilgiris.

### 1A — Primary Satellite Datasets

| # | Dataset | Full Name | Sensor Type | Spatial Resolution | Temporal Resolution | Temporal Coverage | Spectral Bands / Channels Used | GEE Collection ID | Primary SFII Use |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Sentinel-2 L2A | Sentinel-2 MSI Surface Reflectance (Harmonised) | Passive Optical (MSI) | 10 m (B2,B3,B4,B8); 20 m (B5,B6,B7,B8A,B11,B12) | 5-day revisit | 2017-present | B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B8A (RedEdge), B11 (SWIR-1), B12 (SWIR-2), QA60 (cloud mask) | `COPERNICUS/S2_SR_HARMONIZED` | NDVI, NBR, TCW, EVI2, SRT computation |
| 2 | Sentinel-1 GRD | Sentinel-1 C-SAR Ground Range Detected | Active SAR (C-band, 5.6 cm) | 10 m | 6–12 day revisit (IW mode) | 2014-present | VV polarisation, VH polarisation | `COPERNICUS/S1_GRD` | SBP (canopy backscatter σ⁰), GLCM texture metrics |
| 3 | Landsat 8 OLI/TIRS C2L2 | Landsat 8 Collection 2 Tier 1 Level-2 | Passive Optical + Thermal | 30 m | 16-day revisit | 2013-present | SR_B2–B7, ST_B10 (Thermal) | `LANDSAT/LC08/C02/T1_L2` | LST, historical NBR for LandTrendr, CEI baseline |
| 4 | Landsat 9 OLI-2/TIRS-2 C2L2 | Landsat 9 Collection 2 Tier 1 Level-2 | Passive Optical + Thermal | 30 m | 16-day revisit | 2021-present | SR_B2–B7, ST_B10 | `LANDSAT/LC09/C02/T1_L2` | Continuity of L8 record post-2021 |
| 5 | Landsat 5 TM C2L2 | Landsat 5 Thematic Mapper Collection 2 | Passive Optical | 30 m | 16-day revisit | 1984–2013 | SR_B1–B7 | `LANDSAT/LT05/C02/T1_L2` | Historical LandTrendr archive 1985–2000 |
| 6 | Landsat 7 ETM+ C2L2 | Landsat 7 Enhanced Thematic Mapper Plus | Passive Optical | 30 m | 16-day revisit | 1999–2022 | SR_B1–B7 (SLC-off post-2003) | `LANDSAT/LE07/C02/T1_L2` | Gap-fill 1999–2013; use pre-SLC-off where possible |
| 7 | GEDI L2A | Global Ecosystem Dynamics Investigation Level-2A | Spaceborne LiDAR (1064 nm) | 25 m footprint (60 m along-track spacing) | Non-continuous orbital sampling | 2019–2023 | RH10, RH25, RH50, RH75, RH98, cover, pai | `LARSE/GEDI/GEDI02_A_002_MONTHLY` | SBP (H_norm = normalised RH98), canopy height rasterisation |
| 8 | MODIS MOD44B | MODIS Vegetation Continuous Fields (VCF) | Passive Optical (Terra MODIS) | 250 m | Annual | 2000-present | Tree cover %, Non-tree %, Bare % | `MODIS/006/MOD44B` | Reference canopy cover baseline, GEDI gap-fill prior |
| 9 | MODIS MOD13Q1 | MODIS Terra Vegetation Indices 16-Day | Passive Optical | 250 m | 16-day | 2000-present | NDVI, EVI, pixel reliability | `MODIS/006/MOD13Q1` | Long-term NDVI trend analysis, cloud-gap fill reference |
| 10 | MODIS MOD14A1 | MODIS Terra Thermal Anomalies / Fire | Thermal | 1 km | Daily | 2000-present | FireMask, MaxFRP | `MODIS/006/MOD14A1` | Fire disturbance attribution for DMF |
| 11 | Hansen GFC | Global Forest Change (Global Forest Watch) | Derived from Landsat | 30 m | Annual | 2000–2023 | treecover2000, lossyear, gain, datamask | `UMD/hansen/global_forest_change_2023_v1_11` | Deforestation events for disturbance history D={tᵢ,mᵢ} |

---

### 1B — Auxiliary and Ancillary Datasets

| # | Dataset | Source | Resolution | Coverage | Variables Used | GEE / Download Source | SFII Role |
|---|---|---|---|---|---|---|---|
| 12 | ESA CCI Biomass | ESA Climate Change Initiative | 100 m | Global | AGB (t/ha), AGB uncertainty | Zenodo (CEDA) | AGB_ref prior for carbon emission computation CE(p) |
| 13 | ESA CCI Land Cover | ESA CCI-LC S2 Prototype | 10 m (S2 prototype) / 300 m (standard) | Global | Land cover class (22 classes) | `ESA/WorldCover/v200` (GEE) | LULC_change term, forest/non-forest masking, reference pixel selection |
| 14 | SRTM DEM | NASA Shuttle Radar Topography Mission | 30 m | 60°N–56°S | Elevation (m), Slope, Aspect | `USGS/SRTMGL1_003` (GEE) | SAR terrain flattening, topographic correction of NDVI, slope-stratified analysis |
| 15 | Copernicus DEM GLO-30 | Copernicus / ESA | 30 m | Global | Elevation (m) | `COPERNICUS/DEM/GLO30` (GEE) | Preferred DEM for Sentinel-1 SAR terrain correction (more recent than SRTM) |
| 16 | CHIRPS Precipitation | UCSB Climate Hazards Group | 5.5 km | Global | Monthly precipitation (mm) | `UCSB-CHG/CHIRPS/MONTHLY` (GEE) | Climate covariate: precipitation anomaly for ML feature vector |
| 17 | ERA5-Land Monthly | ECMWF Reanalysis | 9 km (0.1°) | Global | 2m air temperature, surface runoff | `ECMWF/ERA5_LAND/MONTHLY_AGGR` (GEE) | Climate covariate: mean temperature for ML feature vector |
| 18 | SPEI Global Drought | CSIC (Beguería & Vicente-Serrano) | 0.5° (~55 km) | Global | SPEI at 12-month scale | csic.es / download | Climate covariate: drought stress index for ML feature vector |
| 19 | TropiSAR / ALOS PALSAR-2 | JAXA / ESA | 25 m | Pan-tropical | L-band HH/HV SAR backscatter | `JAXA/ALOS/PALSAR/YEARLY/SAR` (GEE) | Supplementary SAR biomass proxy (L-band penetrates deeper canopy than C-band) |
| 20 | India State Forest Report | FSI (Forest Survey of India) | District / range level | India | Forest type, degradation class, stocking density | FSI report download | Ground truth reference classes, validation stratum definition |

---

### 1C — Ground Truth / Validation Datasets

| # | Dataset | Institution | Spatial Coverage | Forest Type | Variables | Access Route | SFII Validation Role |
|---|---|---|---|---|---|---|---|
| 21 | Mudumalai / Sathyamangalam forest plots | WCS India / NCBS | Nilgiris core zone | Moist deciduous, wet evergreen | AGB, basal area, species composition, plot age | Collaboration / MOU | Component-level validation: SBP vs. field AGB |
| 22 | IISC Agumbe Rainforest plots | Indian Institute of Science | Western Ghats | Tropical wet evergreen | LAI, canopy height, AGB | Academic collaboration | ERS reference distribution construction |
| 23 | Planet NICFI Basemaps | Planet Labs (NICFI Programme) | Tropics ≤30°N/S | All forest types | True-colour + NIR, 4.77 m | Free (academic registration at planet.com/nicfi) | VHR visual interpretation; expert degradation labelling; disturbance validation |
| 24 | Cartosat-2 / ResourceSAT-2 | ISRO / NRSC | India | All types | Multispectral, 1–5 m | NRSC Bhuvan portal | India-specific VHR validation (supplements Planet) |

---

## Section 2 — Download Strategy

### 2A — Google Earth Engine (GEE) Processing Strategy

| Dataset | GEE Collection | Filter: Date | Filter: Bounds | Filter: Quality | Composite Method | Export Resolution | Export Format | Priority |
|---|---|---|---|---|---|---|---|---|
| Sentinel-2 L2A | `COPERNICUS/S2_SR_HARMONIZED` | 2018-01-01 to 2024-12-31 | AOI = `ee.Geometry.Rectangle([76.0, 10.5, 77.5, 12.0])` | `CLOUDY_PIXEL_PERCENTAGE < 20`; QA60 bitwiseAnd cloud mask | Monthly medoid (spectral angle); annual median fallback | 10 m | Cloud-Optimised GeoTIFF (.tif) | **P1 — Critical** |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | 2018-01-01 to 2024-12-31 | AOI | `instrumentMode = 'IW'`; `orbitProperties_pass = 'DESCENDING'` (preferred for Nilgiris topography) | Quarterly σ⁰ dB mean composite (VV and VH separate) | 10 m | COG GeoTIFF | **P1 — Critical** |
| Landsat 8+9 C2L2 | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` | 2013-01-01 to 2024-12-31 | AOI | `CLOUD_COVER < 30`; QA_PIXEL bit masking (cloud, shadow, saturation) | Annual medoid | 30 m | COG GeoTIFF | **P1 — Critical** |
| Landsat 5 C2L2 | `LANDSAT/LT05/C02/T1_L2` | 1985-01-01 to 2013-05-05 | AOI | `CLOUD_COVER < 30`; QA_PIXEL masking | Annual medoid | 30 m | COG GeoTIFF | **P2 — High** |
| Landsat 7 C2L2 | `LANDSAT/LE07/C02/T1_L2` | 1999-01-01 to 2003-05-31 | AOI | Pre-SLC-off only; `CLOUD_COVER < 30` | Annual medoid | 30 m | COG GeoTIFF | **P3 — Medium** |
| GEDI L2A Monthly | `LARSE/GEDI/GEDI02_A_002_MONTHLY` | 2019-01-01 to 2023-03-31 | AOI | `quality_flag = 1`; `degrade_flag = 0`; `beam_sensitivity > 0.95` | All valid shots → point shapefile → rasterise to 25 m | 25 m → resample to 10 m | COG GeoTIFF (interpolated raster) + GeoPackage (raw footprints) | **P1 — Critical** |
| MODIS MOD44B | `MODIS/006/MOD44B` | 2000-01-01 to 2023-12-31 | AOI | `Percent_Tree_Cover_SD < 20` | Annual (native product) | 250 m | COG GeoTIFF | **P2 — High** |
| MODIS MOD13Q1 | `MODIS/006/MOD13Q1` | 2000-01-01 to 2024-12-31 | AOI | `SummaryQA = 0 or 1` (good or marginal) | 16-day native composites | 250 m | COG GeoTIFF | **P2 — High** |
| MODIS MOD14A1 | `MODIS/006/MOD14A1` | 2001-01-01 to 2024-12-31 | AOI | `FireMask >= 7` (high confidence fire) | Daily → annual fire occurrence map | 1 km | COG GeoTIFF | **P2 — High** |
| Hansen GFC | `UMD/hansen/global_forest_change_2023_v1_11` | Static (2000–2023) | AOI | No filter needed | Static product | 30 m | COG GeoTIFF | **P1 — Critical** |
| ESA WorldCover | `ESA/WorldCover/v200` | 2021 (static) | AOI | No filter | Static | 10 m | COG GeoTIFF | **P2 — High** |
| SRTM / Copernicus DEM | `USGS/SRTMGL1_003` or `COPERNICUS/DEM/GLO30` | Static | AOI + 10 km buffer | No filter | Static mosaic | 30 m | COG GeoTIFF | **P1 — Critical** |
| CHIRPS Monthly | `UCSB-CHG/CHIRPS/MONTHLY` | 2001-01-01 to 2024-12-31 | AOI + 50 km buffer | No filter | Monthly native | 5.5 km → resample to 1 km | COG GeoTIFF | **P3 — Medium** |
| ERA5-Land Monthly | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | 2001-01-01 to 2024-12-31 | AOI + 50 km buffer | No filter | Monthly native | 9 km → resample to 1 km | COG GeoTIFF | **P3 — Medium** |
| ALOS PALSAR-2 | `JAXA/ALOS/PALSAR/YEARLY/SAR` | 2015–2023 | AOI | No filter | Annual mosaic | 25 m | COG GeoTIFF | **P3 — Medium** |

---

### 2B — Non-GEE Download Strategy

| Dataset | Source Portal | URL / API | Format | Download Method | File Size Estimate | Notes |
|---|---|---|---|---|---|---|
| ESA CCI Biomass (2018, 2020) | CEDA Archive / Zenodo | `https://data.ceda.ac.uk/neodc/esacci/biomass` | NetCDF / GeoTIFF tile | Manual tile download; clip to AOI + 10 km buffer | ~2 GB per year (clipped) | Select tiles: N10E075, N10E076, N12E076, N12E077 |
| SPEI-12 | CSIC / SPEIbase v2.9 | `https://digital.csic.es/handle/10261/332007` | NetCDF | Bulk download via browser or `wget` | ~500 MB | Extract AOI grid cells; monthly 1901–2023 |
| Planet NICFI Basemaps | Planet API v1 | `https://api.planet.com/basemaps/v1/mosaics` | GeoTIFF (16-bit) | Planet Python SDK (`planet` CLI); authenticate with NICFI API key | ~5–10 GB per quarter per tile | Nilgiris falls in PSScene tiles; download 2017-Q1 onward |
| Cartosat / ResourceSAT | NRSC Bhuvan | `https://bhuvan.nrsc.gov.in` | GeoTIFF | Web portal request; free for academic users with NRSC account | Variable | Apply for specific dates matching disturbance events |
| GEDI L2A (raw HDF5 backup) | NASA EarthData LP DAAC | `https://lpdaac.usgs.gov/products/gedi02_av002/` | HDF5 | `earthdata` Python client or `wget` with `.netrc` credentials | ~50 GB (raw HDF5, full archive) | Use GEE version preferentially; raw HDF5 only for custom QA analysis |
| India Forest Survey Reports | FSI India | `http://fsi.nic.in` | PDF / Shapefile | Manual download | <100 MB | ISFR 2021 and 2023 editions; extract Nilgiris district statistics |

---

### 2C — GEE Export Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Export destination | Google Drive → `/SFII_Nilgiris/raw/` | Primary; use GEE asset storage as secondary cache |
| CRS | `EPSG:32643` (WGS84 / UTM Zone 43N) | Metric coordinate system for area calculations; standard for South India |
| CRS transform | `[10, 0, 166021, 0, -10, 1328547]` | Aligns to 10 m snap grid anchored at zone origin |
| Region | AOI + 5 km buffer | Buffer prevents edge effects in texture computation and compositing |
| Max pixels | `1e13` | Prevent GEE export truncation |
| File dimensions | `5000 × 5000` | ~50 MB per GeoTIFF tile; avoids GEE 32 MB limit per file |
| shardSize | 256 | Optimised for COG internal tiling |
| Pyramid resample | `bilinear` (continuous); `mode` (categorical) | Preserves radiometric continuity for indices; prevents class blending |
| Batch export naming | Automated via Python `ee.batch.Export.image.toDrive(...)` loop | Script loops over sensors × years × months |

---

## Section 3 — Storage Structure

### 3A — Master Directory Tree

```
SFII_Nilgiris/
│
├── 00_aoi/                                  # Study area boundary files
│   ├── nilgiris_br_boundary.gpkg            # Official NBR boundary (GeoPackage)
│   ├── nilgiris_br_buffer5km.gpkg           # Processing buffer
│   ├── nilgiris_zones.gpkg                  # Core / buffer / transition zones
│   └── gee_aoi_geometry.json               # GEE-compatible rectangle geometry
│
├── 01_raw/                                  # Immutable raw downloads — NEVER modify
│   │
│   ├── sentinel2/
│   │   ├── monthly_composites/
│   │   │   ├── 2018/
│   │   │   │   ├── NBR_S2L2A_NIL_10m_20180101_20180131_UTM43N.tif
│   │   │   │   ├── NDVI_S2L2A_NIL_10m_20180101_20180131_UTM43N.tif
│   │   │   │   ├── TCW_S2L2A_NIL_10m_20180101_20180131_UTM43N.tif
│   │   │   │   └── ...
│   │   │   └── .../
│   │   ├── annual_composites/
│   │   │   ├── NDVI_S2L2A_NIL_10m_2018_annual_UTM43N.tif
│   │   │   └── .../
│   │   └── metadata/
│   │       └── s2_scene_inventory.csv
│   │
│   ├── sentinel1/
│   │   ├── quarterly_composites/
│   │   │   ├── VV_S1GRD_NIL_10m_2018Q1_DESC_UTM43N.tif
│   │   │   ├── VH_S1GRD_NIL_10m_2018Q1_DESC_UTM43N.tif
│   │   │   └── .../
│   │   ├── texture_glcm/
│   │   │   ├── CONTRAST_S1GRD_NIL_10m_2018Q1_DESC_UTM43N.tif
│   │   │   ├── ENTROPY_S1GRD_NIL_10m_2018Q1_DESC_UTM43N.tif
│   │   │   └── .../
│   │   └── metadata/
│   │
│   ├── landsat/
│   │   ├── l5_tm/
│   │   │   └── annual_composites/
│   │   │       └── NBR_L5TM_NIL_30m_1985_annual_UTM43N.tif
│   │   ├── l7_etm/
│   │   │   └── annual_composites/
│   │   ├── l8_oli/
│   │   │   └── annual_composites/
│   │   │       ├── NBR_L8OLI_NIL_30m_2018_annual_UTM43N.tif
│   │   │       ├── LST_L8OLI_NIL_30m_2018_annual_UTM43N.tif
│   │   │       └── .../
│   │   ├── l9_oli2/
│   │   │   └── annual_composites/
│   │   └── harmonised/                      # Cross-sensor harmonised NBR stack
│   │       └── NBR_LSAT_NIL_30m_1985_2024_harmonised_UTM43N.tif
│   │
│   ├── gedi/
│   │   ├── footprints_raw/
│   │   │   └── GEDI_L2A_NIL_25m_footprints_2019_2023.gpkg
│   │   ├── rasters_interpolated/
│   │   │   ├── RH98_GEDI_NIL_25m_2020_interpolated_UTM43N.tif
│   │   │   ├── RH50_GEDI_NIL_25m_2020_interpolated_UTM43N.tif
│   │   │   └── RH98_GEDI_NIL_10m_2020_regressed_UTM43N.tif    # Optical-regression gap-fill
│   │   └── metadata/
│   │       └── gedi_shot_qa_summary.csv
│   │
│   ├── modis/
│   │   ├── mod44b_vcf/
│   │   │   └── TREECOV_MOD44B_NIL_250m_2001_2023_UTM43N.tif
│   │   ├── mod13q1_vi/
│   │   │   ├── NDVI_MOD13Q1_NIL_250m_2001_2024_UTM43N.tif
│   │   │   └── EVI_MOD13Q1_NIL_250m_2001_2024_UTM43N.tif
│   │   ├── mod14a1_fire/
│   │   │   └── FIRE_MOD14A1_NIL_1km_2001_2024_annual_UTM43N.tif
│   │   └── metadata/
│   │
│   ├── hansen_gfc/
│   │   ├── TREECOVER2000_HANSEN_NIL_30m_UTM43N.tif
│   │   ├── LOSSYEAR_HANSEN_NIL_30m_2001_2023_UTM43N.tif
│   │   ├── GAIN_HANSEN_NIL_30m_UTM43N.tif
│   │   └── DATAMASK_HANSEN_NIL_30m_UTM43N.tif
│   │
│   ├── auxiliary/
│   │   ├── dem/
│   │   │   ├── DEM_SRTM_NIL_30m_UTM43N.tif
│   │   │   ├── DEM_COP30_NIL_30m_UTM43N.tif
│   │   │   ├── SLOPE_NIL_30m_UTM43N.tif
│   │   │   └── ASPECT_NIL_30m_UTM43N.tif
│   │   ├── land_cover/
│   │   │   └── LC_WORLDCOVER_NIL_10m_2021_UTM43N.tif
│   │   ├── biomass/
│   │   │   ├── AGB_CCIBIO_NIL_100m_2018_UTM43N.tif
│   │   │   └── AGB_CCIBIO_NIL_100m_2020_UTM43N.tif
│   │   ├── climate/
│   │   │   ├── PRECIP_CHIRPS_NIL_5km_2001_2024_monthly_UTM43N.tif
│   │   │   ├── TEMP_ERA5_NIL_9km_2001_2024_monthly_UTM43N.tif
│   │   │   └── SPEI12_CSIC_NIL_55km_1901_2023_monthly_UTM43N.tif
│   │   ├── palsar/
│   │   │   └── HV_PALSAR2_NIL_25m_2015_2023_annual_UTM43N.tif
│   │   └── planet_nicfi/
│   │       ├── 2017H1/
│   │       │   └── NICFI_PS_NIL_5m_2017H1_UTM43N.tif
│   │       └── .../
│   │
│   └── ground_truth/
│       ├── field_plots/
│       │   ├── nilgiris_field_plots_raw.csv
│       │   └── nilgiris_field_plots.gpkg
│       ├── vhr_interpretation/
│       │   └── expert_degradation_labels_v1.gpkg
│       └── fsi_reports/
│           └── ISFR_2023_Nilgiris_extract.csv
│
├── 02_processed/                             # Derived, preprocessed data
│   │
│   ├── indices/
│   │   ├── ndvi/
│   │   ├── nbr/
│   │   ├── evi2/
│   │   ├── tcw/
│   │   ├── ndwi/
│   │   └── ndbsi/
│   │
│   ├── composites_clean/                     # Cloud-masked, gap-filled, coregistered
│   │   ├── s2_monthly_clean/
│   │   └── landsat_annual_clean/
│   │
│   ├── sar_processed/
│   │   ├── sigma0_db/                        # Calibrated, terrain-corrected σ⁰
│   │   └── glcm_texture/
│   │
│   ├── gedi_processed/
│   │   ├── filtered_footprints/
│   │   └── rh98_raster_10m/
│   │
│   └── coregistration_report/
│       └── coregistration_accuracy_log.csv
│
├── 03_sfii_components/                       # Per-year SFII sub-index rasters
│   ├── srt/
│   │   ├── SRT_NIL_10m_2018_UTM43N.tif
│   │   └── .../
│   ├── sbp/
│   │   ├── SBP_NIL_10m_2018_UTM43N.tif
│   │   └── .../
│   ├── dmf/
│   │   ├── DMF_NIL_10m_2018_UTM43N.tif
│   │   └── .../
│   ├── ers/
│   │   ├── ERS_NIL_10m_2018_UTM43N.tif
│   │   └── .../
│   └── frp/
│       ├── FRP_NIL_10m_2018_UTM43N.tif
│       └── .../
│
├── 04_sfii_outputs/                          # Final index and derived products
│   ├── sfii_annual/
│   │   ├── SFII_NIL_10m_2018_UTM43N.tif
│   │   └── .../
│   ├── sfii_classified/                      # 5-class degradation maps
│   │   └── SFII_CLASS_NIL_10m_2018_UTM43N.tif
│   ├── cei_baseline/                         # CEI for comparison
│   │   └── CEI_NIL_30m_2018_UTM43N.tif
│   ├── disturbance_history/
│   │   ├── DISTYR_LANDTRENDR_NIL_30m_1985_2024_UTM43N.tif
│   │   ├── DISTMAG_LANDTRENDR_NIL_30m_1985_2024_UTM43N.tif
│   │   └── disturbance_history_db.parquet    # Per-pixel disturbance DB
│   ├── recovery_metrics/
│   │   ├── TSR_NIL_30m_UTM43N.tif           # Time to Spectral Recovery
│   │   ├── TSTR_NIL_30m_UTM43N.tif          # Time to Structural Recovery
│   │   └── RECLAG_NIL_30m_UTM43N.tif        # Recovery Lag = TSTR − TSR
│   └── carbon_estimates/
│       └── CE_NIL_10m_2018_2024_tCha_UTM43N.tif
│
├── 05_ml/                                    # Machine learning pipeline
│   ├── features/
│   │   ├── feature_matrix_2018_2024.zarr     # 3D array: pixels × years × features
│   │   └── feature_metadata.json            # Feature names, units, scaling params
│   ├── labels/
│   │   ├── training_labels_v1.gpkg
│   │   └── training_labels_v1.parquet
│   ├── models/
│   │   ├── lstm_forestmodel_v1.pt
│   │   ├── rf_classifier_v1.pkl
│   │   └── xgb_regressor_v1.ubj
│   ├── predictions/
│   │   ├── SFII_PRED_NIL_10m_2022_UTM43N.tif
│   │   └── SFII_FORECAST_NIL_10m_2030_UTM43N.tif
│   └── evaluation/
│       ├── validation_statistics.csv
│       ├── confusion_matrix.csv
│       └── feature_importance.csv
│
├── 06_validation/                            # All validation materials
│   ├── component_validation/
│   ├── sfii_accuracy_assessment/
│   └── temporal_prediction_validation/
│
├── 07_outputs_publication/                   # Final figures, tables, supplementary
│   ├── figures/
│   ├── tables/
│   └── supplementary/
│
├── 08_scripts/                               # All processing scripts (versioned)
│   ├── gee/
│   ├── preprocessing/
│   ├── sfii_computation/
│   ├── ml_pipeline/
│   └── validation/
│
└── 09_metadata/                              # Provenance and documentation
    ├── data_provenance.csv
    ├── processing_log.csv
    └── dataset_inventory_master.csv
```

---

## Section 4 — Naming Convention

### 4A — File Naming Token Structure

Every file follows a **7-token structure** separated by underscores (`_`):

```
[INDEX/VARIABLE]_[SENSOR]_[SITE]_[RESOLUTION]_[DATE/PERIOD]_[COMPOSITE]_[CRS].ext
```

| Token | Position | Purpose | Allowed Values / Format |
|---|---|---|---|
| `INDEX/VARIABLE` | 1 | What the file contains | See Table 4B |
| `SENSOR` | 2 | Source satellite/dataset | See Table 4C |
| `SITE` | 3 | Study area abbreviation | `NIL` (Nilgiris), `NIL_CORE`, `NIL_BUF` |
| `RESOLUTION` | 4 | Native or resampled pixel size | `10m`, `25m`, `30m`, `100m`, `250m`, `1km` |
| `DATE/PERIOD` | 5 | Date or temporal range | `YYYYMMDD`, `YYYYMMDD_YYYYMMDD`, `YYYY`, `YYYYQ#`, `YYYYH#` |
| `COMPOSITE` | 6 | Compositing method applied | `monthly`, `quarterly`, `annual`, `static`, `interpolated`, `regressed` |
| `CRS` | 7 | Coordinate reference system | `UTM43N` (EPSG:32643), `GEO` (WGS84 geographic) |
| Extension | — | File format | `.tif` (COG GeoTIFF), `.gpkg` (vector), `.parquet`, `.zarr`, `.csv`, `.json` |

---

### 4B — Allowed Variable / Index Names

| Code | Full Name | Unit | Data Type |
|---|---|---|---|
| `NDVI` | Normalised Difference Vegetation Index | Dimensionless [−1, 1] | Float32 |
| `NBR` | Normalised Burn Ratio | Dimensionless [−1, 1] | Float32 |
| `EVI2` | Enhanced Vegetation Index 2 | Dimensionless | Float32 |
| `TCW` | Tasseled Cap Wetness | Dimensionless | Float32 |
| `NDWI` | Normalised Difference Water Index | Dimensionless | Float32 |
| `NDBSI` | Normalised Difference Bare Soil Index | Dimensionless | Float32 |
| `LST` | Land Surface Temperature | Kelvin | Float32 |
| `VV` | Sentinel-1 VV polarisation backscatter | dB | Float32 |
| `VH` | Sentinel-1 VH polarisation backscatter | dB | Float32 |
| `CONTRAST` | GLCM Contrast texture | Dimensionless | Float32 |
| `ENTROPY` | GLCM Entropy texture | Dimensionless | Float32 |
| `HOMOGEN` | GLCM Homogeneity texture | Dimensionless | Float32 |
| `RH50` | GEDI Relative Height at 50th percentile | Metres | Float32 |
| `RH98` | GEDI Relative Height at 98th percentile (proxy for canopy top) | Metres | Float32 |
| `TREECOV` | MODIS VCF Tree Cover percentage | % | UInt8 |
| `LOSSYEAR` | Hansen GFC year of first forest loss | Year (0=no loss) | UInt8 |
| `TREECOVER2000` | Hansen GFC tree cover in year 2000 | % | UInt8 |
| `FIRE` | MODIS MOD14 annual fire occurrence | Binary (0/1) | UInt8 |
| `AGB` | Above-ground Biomass | t ha⁻¹ | Float32 |
| `DEM` | Digital Elevation Model | Metres | Float32 |
| `SLOPE` | Terrain slope | Degrees | Float32 |
| `ASPECT` | Terrain aspect | Degrees | Float32 |
| `LC` | Land cover class | Class code | UInt8 |
| `PRECIP` | Monthly precipitation | mm | Float32 |
| `TEMP` | Mean monthly temperature | °C | Float32 |
| `SPEI12` | Standardised Precipitation-Evapotranspiration Index (12-month) | Z-score | Float32 |
| `SRT` | Spectral Recovery Trajectory | Dimensionless [0, 2] | Float32 |
| `SBP` | Structural Biomass Proxy | Dimensionless [0, 1] | Float32 |
| `DMF` | Disturbance Memory Function | Dimensionless [0, 1] | Float32 |
| `ERS` | Ecosystem Resilience Score | Dimensionless [0, 1] | Float32 |
| `FRP` | False Recovery Penalty | Dimensionless [0, 1] | Float32 |
| `SFII` | Structural Forest Integrity Index | Dimensionless [0, 1] | Float32 |
| `SFII_CLASS` | SFII 5-class degradation map | Class 1–5 | UInt8 |
| `CEI` | Carbon Emission Index (baseline) | Dimensionless [0, 1] | Float32 |
| `DISTYR` | Year of disturbance event (LandTrendr) | Year | UInt16 |
| `DISTMAG` | Magnitude of disturbance event | NBR units | Float32 |
| `TSR` | Time to Spectral Recovery | Years | Float32 |
| `TSTR` | Time to Structural Recovery | Years | Float32 |
| `RECLAG` | Recovery Lag (TSTR − TSR) | Years | Float32 |
| `CE` | Cumulative Carbon Emission estimate | t C ha⁻¹ | Float32 |
| `SFII_PRED` | ML-predicted SFII | Dimensionless [0, 1] | Float32 |
| `SFII_FORECAST` | ML-forecasted future SFII | Dimensionless [0, 1] | Float32 |

---

### 4C — Allowed Sensor Codes

| Code | Dataset | Full Name |
|---|---|---|
| `S2L2A` | Sentinel-2 L2A | Sentinel-2 MSI Surface Reflectance Harmonised |
| `S1GRD` | Sentinel-1 GRD | Sentinel-1 C-SAR Ground Range Detected |
| `L5TM` | Landsat 5 TM | Landsat 5 Thematic Mapper Collection 2 |
| `L7ETM` | Landsat 7 ETM+ | Landsat 7 Enhanced Thematic Mapper Plus Collection 2 |
| `L8OLI` | Landsat 8 OLI/TIRS | Landsat 8 Collection 2 |
| `L9OLI2` | Landsat 9 OLI-2 | Landsat 9 Collection 2 |
| `LSAT` | Landsat harmonised | Cross-sensor harmonised Landsat stack (L5+L7+L8+L9) |
| `GEDI` | GEDI L2A | Global Ecosystem Dynamics Investigation Level-2A |
| `MOD44B` | MODIS VCF | MODIS MOD44B Vegetation Continuous Fields |
| `MOD13Q1` | MODIS NDVI | MODIS MOD13Q1 Vegetation Indices 16-Day |
| `MOD14A1` | MODIS Fire | MODIS MOD14A1 Thermal Anomalies |
| `HANSEN` | Hansen GFC | Global Forest Change v1.11 |
| `CCIBIO` | ESA CCI Biomass | ESA CCI Above-Ground Biomass |
| `WORLDCOV` | ESA WorldCover | ESA WorldCover v2.0 |
| `SRTM` | SRTM DEM | USGS SRTM 30m |
| `COP30` | Copernicus DEM | Copernicus GLO-30 DEM |
| `CHIRPS` | CHIRPS Precip | UCSB CHIRPS Monthly |
| `ERA5` | ERA5-Land | ECMWF ERA5-Land Monthly |
| `CSIC` | SPEI | CSIC SPEIbase v2.9 |
| `PALSAR2` | ALOS PALSAR-2 | JAXA ALOS PALSAR-2 Annual Mosaic |
| `NICFI_PS` | Planet NICFI | Planet PlanetScope NICFI Basemaps |
| `SFII` | Derived | SFII pipeline output |

---

### 4D — Naming Examples

| Correct Filename | Meaning |
|---|---|
| `NDVI_S2L2A_NIL_10m_20180601_20180630_UTM43N.tif` | Sentinel-2 NDVI monthly medoid composite for Nilgiris, June 2018, 10 m, UTM Zone 43N |
| `VH_S1GRD_NIL_10m_2020Q3_DESC_UTM43N.tif` | Sentinel-1 VH backscatter quarterly composite, Q3 2020, descending orbit, 10 m |
| `NBR_LSAT_NIL_30m_1985_annual_UTM43N.tif` | Harmonised Landsat NBR annual composite, 1985, 30 m |
| `RH98_GEDI_NIL_10m_2020_regressed_UTM43N.tif` | GEDI RH98 gap-filled via optical regression, 10 m, 2020 |
| `SBP_SFII_NIL_10m_2022_annual_UTM43N.tif` | SFII Structural Biomass Proxy layer, 10 m, 2022 |
| `SFII_SFII_NIL_10m_2022_annual_UTM43N.tif` | Final SFII composite index, 10 m, 2022 |
| `FIRE_MOD14A1_NIL_1km_2005_annual_UTM43N.tif` | Annual fire occurrence from MODIS MOD14A1, 1 km, 2005 |
| `LOSSYEAR_HANSEN_NIL_30m_static_UTM43N.tif` | Hansen GFC forest loss year, 30 m, static product |

---

### 4E — Version Control Convention

| Scenario | Convention | Example |
|---|---|---|
| First version of any file | No version suffix | `SFII_SFII_NIL_10m_2022_annual_UTM43N.tif` |
| Revised parameters / reprocessed | Append `_v2`, `_v3` before extension | `SFII_SFII_NIL_10m_2022_annual_UTM43N_v2.tif` |
| Work-in-progress draft | Append `_WIP` before extension | `SFII_SFII_NIL_10m_2022_annual_UTM43N_WIP.tif` |
| Deprecated file | Move to `_deprecated/` subfolder | Do not delete until pipeline validation complete |
| Processing scripts | Semantic versioning in header comment + Git tag | `sfii_compute.py` → git tag `v1.2.0` |

---

## Section 5 — Metadata Schema

> Every dataset file must be accompanied by a metadata record. Two schemas are defined: a **File-Level Schema** (stored in `data_provenance.csv`) and a **Raster-Level Schema** (embedded in GeoTIFF tags and/or a sidecar `.json`).

### 5A — Master Dataset Inventory Metadata Schema

> Stored in: `09_metadata/dataset_inventory_master.csv`
> One row per unique dataset (sensor × temporal extent).

| Field Name | Data Type | Required | Description | Example Value |
|---|---|---|---|---|
| `dataset_id` | String | ✅ | Unique internal identifier | `S2L2A_001` |
| `dataset_name` | String | ✅ | Human-readable name | `Sentinel-2 L2A Monthly NDVI Composites` |
| `sensor_code` | String | ✅ | From Table 4C | `S2L2A` |
| `variable_code` | String | ✅ | From Table 4B | `NDVI` |
| `source_collection` | String | ✅ | GEE collection ID or download URL | `COPERNICUS/S2_SR_HARMONIZED` |
| `temporal_start` | Date (YYYY-MM-DD) | ✅ | Start of temporal coverage used | `2018-01-01` |
| `temporal_end` | Date (YYYY-MM-DD) | ✅ | End of temporal coverage used | `2024-12-31` |
| `temporal_resolution` | String | ✅ | Native temporal step | `5-day revisit; monthly composite` |
| `spatial_resolution_m` | Integer | ✅ | Native pixel size (metres) | `10` |
| `output_resolution_m` | Integer | ✅ | Exported pixel size (metres) | `10` |
| `crs_epsg` | Integer | ✅ | EPSG code of output CRS | `32643` |
| `aoi_name` | String | ✅ | Study area name | `Nilgiri Biosphere Reserve` |
| `aoi_bbox_wgs84` | String | ✅ | [west, south, east, north] | `[76.0, 10.5, 77.5, 12.0]` |
| `cloud_threshold_pct` | Float | Conditional | Applied cloud filter (optical only) | `20.0` |
| `quality_filters` | String | ✅ | All QA/QC filters applied | `QA60 bitwiseAnd cloud mask; CLOUDY_PIXEL_PERCENTAGE < 20` |
| `composite_method` | String | ✅ | Compositing algorithm | `Monthly medoid (spectral angle distance)` |
| `atmospheric_correction` | String | Conditional | Correction applied | `Sen2Cor (embedded in L2A product)` |
| `download_date` | Date (YYYY-MM-DD) | ✅ | Date of download/export | `2025-03-15` |
| `download_platform` | String | ✅ | How data was obtained | `Google Earth Engine → Google Drive` |
| `gee_script_id` | String | Conditional | GEE script filename | `01_s2_monthly_composites.py` |
| `file_count` | Integer | ✅ | Number of files in this dataset | `84` |
| `total_size_gb` | Float | ✅ | Combined file size (GB) | `12.4` |
| `format` | String | ✅ | File format | `Cloud-Optimised GeoTIFF (COG)` |
| `nodata_value` | Float | ✅ | NoData / fill value | `-9999.0` |
| `scale_factor` | Float | ✅ | Scale factor applied to stored values | `0.0001` (if stored as Int16) |
| `offset` | Float | ✅ | Offset applied to stored values | `0.0` |
| `valid_range_min` | Float | ✅ | Physical minimum valid value | `-1.0` |
| `valid_range_max` | Float | ✅ | Physical maximum valid value | `1.0` |
| `doi` | String | Conditional | Dataset DOI if available | `10.5270/S2_-742ikth` |
| `citation` | String | ✅ | Full citation string | `ESA (2021). Sentinel-2 MSI Level-2A. https://doi.org/10.5270/S2_-742ikth` |
| `licence` | String | ✅ | Data use licence | `Copernicus Sentinel Open Data Licence v1.1` |
| `processing_notes` | String | Optional | Any non-standard processing steps | `Medoid replaced by median for months with <5 clear scenes` |
| `sfii_role` | String | ✅ | Which SFII component this dataset feeds | `SRT (NDVI, NBR), SBP (TCW)` |
| `status` | Enum | ✅ | Acquisition status | `Downloaded`, `In Progress`, `Pending`, `Failed` |

---

### 5B — File-Level Raster Sidecar Metadata Schema

> Stored as: `[filename].json` alongside every GeoTIFF.
> Also embedded in GeoTIFF GDAL metadata tags (`gdal.SetMetadata`).

| Field Name | Required | Description | Example |
|---|---|---|---|
| `filename` | ✅ | Exact filename with extension | `NDVI_S2L2A_NIL_10m_20180601_20180630_UTM43N.tif` |
| `variable` | ✅ | Variable name | `NDVI` |
| `sensor` | ✅ | Sensor code | `S2L2A` |
| `site` | ✅ | Site code | `NIL` |
| `date_start` | ✅ | Composite start date | `2018-06-01` |
| `date_end` | ✅ | Composite end date | `2018-06-30` |
| `composite_type` | ✅ | Monthly / Annual / Static | `monthly` |
| `scenes_used` | ✅ | Number of scenes in composite | `7` |
| `cloud_free_pct` | ✅ | % valid pixels after cloud masking | `78.3` |
| `gap_fill_applied` | ✅ | Was gap-filling applied? | `false` |
| `gap_fill_method` | Conditional | Gap-fill algorithm if applied | `linear_temporal_interpolation` |
| `crs_wkt` | ✅ | Full WKT string of CRS | `PROJCS["WGS 84 / UTM zone 43N"...]` |
| `pixel_size_m` | ✅ | Pixel size in metres | `10` |
| `extent_west` | ✅ | West bounding coordinate (map units) | `166021.0` |
| `extent_east` | ✅ | East bounding coordinate | `332021.0` |
| `extent_north` | ✅ | North bounding coordinate | `1328547.0` |
| `extent_south` | ✅ | South bounding coordinate | `1160547.0` |
| `rows` | ✅ | Number of rows | `16800` |
| `cols` | ✅ | Number of columns | `16600` |
| `nodata_value` | ✅ | NoData fill value | `-9999.0` |
| `data_type` | ✅ | GeoTIFF data type | `Float32` |
| `compression` | ✅ | Internal compression | `DEFLATE` |
| `tiling` | ✅ | Internal tile dimensions | `256×256` |
| `checksum_md5` | ✅ | MD5 hash of file for integrity | `a3f1b2c4...` |
| `file_size_mb` | ✅ | File size in MB | `48.2` |
| `created_by` | ✅ | Script that produced this file | `01_s2_monthly_composites.py` |
| `created_date` | ✅ | ISO 8601 creation timestamp | `2025-03-15T14:32:00+05:30` |
| `gee_image_id` | Conditional | GEE image ID if exported from GEE | `COPERNICUS/S2_SR_HARMONIZED/...` |
| `processing_version` | ✅ | Script version / git commit hash | `v1.0 / 3a7f9bc` |
| `processing_notes` | Optional | Non-standard steps | `Medoid failed: fell back to median for June 2018` |
| `sfii_role` | ✅ | Which SFII component uses this layer | `SRT` |

---

### 5C — SFII Component-Specific Metadata Fields

> Additional fields required for SFII computation outputs (stored in `03_sfii_components/` and `04_sfii_outputs/`).

| Field Name | Applies To | Required | Description | Example |
|---|---|---|---|---|
| `sfii_formula_version` | All SFII layers | ✅ | Version of SFII formula used | `SFII_v1.2` |
| `sfii_weights` | SFII final | ✅ | Weights used: w1, w2, w3, w4 | `[0.30, 0.25, 0.25, 0.20]` |
| `srt_vi_type` | SRT | ✅ | VI used for SRT calculation | `NBR` |
| `srt_baseline_years` | SRT | ✅ | Pre-disturbance baseline period | `5-year median (t₀−5 to t₀−1)` |
| `srt_reference_radius_km` | SRT | ✅ | Reference pixel search radius | `5.0` |
| `sbp_alpha` | SBP | ✅ | GEDI RH98 weight | `0.5` |
| `sbp_beta` | SBP | ✅ | Sentinel-1 σ⁰ weight | `0.3` |
| `sbp_gamma` | SBP | ✅ | TCW weight | `0.2` |
| `sbp_gedi_source` | SBP | ✅ | GEDI: raw / interpolated / regressed | `RF_regression_from_S2` |
| `dmf_lambda` | DMF | ✅ | Biome-specific decay constant (yr⁻¹) | `0.08` |
| `dmf_lambda_source` | DMF | ✅ | Source of λ value | `Literature (Poorter et al. 2016)` |
| `dmf_max_definition` | DMF | ✅ | How DMF_max was defined | `Theoretical: 1/(lambda·e)` |
| `dmf_disturbance_source` | DMF | ✅ | Source of disturbance history | `LandTrendr_NBR + MOD14A1_fire + Hansen_GFC` |
| `ers_feature_vector` | ERS | ✅ | Features used in Bhattacharyya distance | `[NDVI, NBR, TCW, RH98, VH_dB]` |
| `ers_normalisation` | ERS | ✅ | Normalisation method | `z-score (reference forest mean and std)` |
| `ers_reference_pixel_source` | ERS | ✅ | How reference pixels were selected | `WorldCover class 10 (tree cover) + Hansen treecover2000 ≥ 70 + no Hansen loss` |
| `ers_n_reference_pixels` | ERS | ✅ | Number of reference pixels used | `14,820` |
| `frp_threshold` | FRP | ✅ | Threshold below which FRP = 0 | `max(0, SRT−SBP)` |
| `recovery_curve_model` | TSR / TSTR | ✅ | Fitted recovery model | `Chapman-Richards` |
| `carbon_fraction` | CE | ✅ | Biomass-to-carbon conversion factor | `0.47` |
| `agb_ref_source` | CE | ✅ | AGB reference dataset | `ESA CCI Biomass 2020` |

---

### 5D — Processing Log Schema

> Stored in: `09_metadata/processing_log.csv`
> One row per processing step executed.

| Field | Description | Example |
|---|---|---|
| `log_id` | Auto-increment integer | `00142` |
| `timestamp` | ISO 8601 | `2025-04-01T09:15:00+05:30` |
| `step_name` | Descriptive step name | `Sentinel-2 monthly medoid composite` |
| `script_name` | Script filename | `01_s2_monthly_composites.py` |
| `script_version` | Git commit hash or version tag | `3a7f9bc` |
| `input_files` | Comma-separated input filenames | `GEE collection: COPERNICUS/S2_SR_HARMONIZED` |
| `output_files` | Comma-separated output filenames | `NDVI_S2L2A_NIL_10m_20180601_20180630_UTM43N.tif` |
| `parameters` | JSON string of key parameters | `{"cloud_thresh": 20, "composite": "medoid"}` |
| `status` | Outcome | `SUCCESS`, `FAILED`, `PARTIAL` |
| `runtime_min` | Processing time (minutes) | `12.3` |
| `errors` | Error messages if any | `None` |
| `operator` | Person / system that ran the step | `researcher_1` |
| `notes` | Any deviation from standard | `Medoid failed for Jun 2018; fell back to median` |

---

## Summary Reference Card

| Component | Key Decision | Value for Nilgiris |
|---|---|---|
| AOI | Bounding Box | 76.0–77.5°E, 10.5–12.0°N (+5 km processing buffer) |
| Output CRS | EPSG | 32643 (WGS84 / UTM Zone 43N) |
| Output resolution | Analysis grid | 10 m (Sentinel-2 native) |
| Historical archive | Landsat | 1985–2024 (L5+L7+L8+L9 harmonised) |
| Primary optical | Sensor | Sentinel-2 L2A Harmonised |
| SAR orbit | Sentinel-1 direction | Descending (preferred for Western Ghats terrain) |
| Cloud filter | Optical threshold | < 20% for scene-level; < 30% monthly composite |
| Composite method | Algorithm | True spectral medoid (not band-wise median) |
| LiDAR | Primary source | GEDI L2A (2019–2023); gap-fill via RF regression from S2 |
| Forest loss | Primary source | Hansen GFC v1.11 (2000–2023) |
| Fire disturbance | Primary source | MODIS MOD14A1 (high confidence mask ≥ 7) |
| Biomass reference | AGB_ref | ESA CCI Biomass 2020 (100 m, resampled to 10 m) |
| Total datasets | Count | 24 primary + auxiliary datasets |
| Estimated raw storage | Total | ~120–150 GB (Nilgiris AOI only) |
| File format | Standard | Cloud-Optimised GeoTIFF (COG), DEFLATE compression |
| NoData value | Standard | `-9999.0` (Float32 layers) |
| Version control | Strategy | Git (scripts) + MD5 checksum (rasters) + processing log |
