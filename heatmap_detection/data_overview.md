# Data Overview

## Dataset: Marine Heatwave (MHW) 2016 JFM

Naming pattern: `mhw2016JFM_{variable}_{year}.nc`

### Variables (4 types)

| Variable | File prefix | Description |
|----------|------------|-------------|
| `sst` | `mhw2016JFM_sst_` | Analysed sea surface temperature (SST) |
| `tcc` | `mhw2016JFM_tcc_` | Total cloud cover |
| `uwind` | `mhw2016JFM_uwind_` | U-component (east-west) of wind |
| `vwind` | `mhw2016JFM_vwind_` | V-component (north-south) of wind |

### Years covered
2013, 2014, 2015, 2016, 2017, 2018, 2019 (7 files per variable, 28 total)

### Additional dataset
`mhw2023JJA_sst_{year}.nc` — SST for years 2018–2024 (7 files), likely a separate JJA (June-July-August) marine heatwave event.

---

## SST File Structure (inspected)

### Dimensions
| Dimension | Size | Notes |
|-----------|------|-------|
| `time` | 90 (91 for leap year 2016) | Daily, Jan 1 – Mar 31 (JFM) |
| `latitude` | 400 | ~35.0°N to ~55.0°N |
| `longitude` | 700 | ~-75.0°W to ~-40.0°W |

### Spatial extent
- **Latitude:** 35.025°N → 54.975°N (0.05° resolution)
- **Longitude:** 74.975°W → 40.025°W (0.05° resolution)
- **Region:** Northwest Atlantic Ocean (US/Canada east coast, Gulf Stream area)

### Variables
| Variable | Shape | Dtype | Units | Description |
|----------|-------|-------|-------|-------------|
| `analysed_sst` | (time, latitude, longitude) | float64 | Kelvin | Analysed sea & under-ice surface temperature |
| `latitude` | (400,) | float32 | degrees_north | Latitude coordinates |
| `longitude` | (700,) | float32 | degrees_east | Longitude coordinates |
| `time` | (90,) | datetime64[ns] | — | Daily timestamps |

### SST metadata
- **Source:** AASTI v2 SST/IST, ESA CCI SST and C3S SST L2P products
- **Valid range:** [-6000, 4500] (likely scaled integers)
- **Size per file:** ~202 MB (25.2M values × 8 bytes)
- **Total SST data:** ~1.4 GB for 7 years
