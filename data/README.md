# Input data

The input files are not part of this release. Place them here with the
layout below, or point `data_dir` in the case file to another folder.

```text
data/
├── seine/                          Seine estuary (cases/seine.yaml)
│   ├── river_profiles_raster.csv   cross sections (bathymetry)
│   ├── Q_in_vernon.csv(.gz)        upstream discharge
│   ├── boundary_downstream.csv(.gz) downstream stage at Tancarville
│   ├── tributaries.csv(.gz)        tributary discharges
│   └── gage_stations/              stage records, one file per station
│       ├── index.csv               station list (metadata, not read)
│       ├── H3270001.csv.gz
│       └── ...
└── synthetic/                      synthetic estuary (cases/synthetic.yaml)
    └── made by: python tools/make_synthetic_case.py --out data/synthetic
```

A compressed copy (`.gz`, `.bz2`, `.xz`, `.zip`) is found automatically when the
case names the plain `.csv`.

## Conventions

| | |
|---|---|
| `timestamp` | `YYYY-MM-DD HH:MM:SS`, one clock for all files (UT+1 for the Seine, no daylight saving) |
| stage | m, in the vertical datum of the survey (m NGF / IGN69 for the Seine) |
| discharge | m3/s |
| chainage `X_m` | m from the upstream end of the reach, increasing downstream |
| `imputed` (optional) | 0 where the value was measured, 1 where it was filled |

Gaps are allowed: the model fills them (harmonic analysis for the tide,
log-space interpolation towards the seasonal norm for discharge) and keeps a
mask, so filled values never count as measurements. If a file has an
`imputed` column (or `<column>_imputed`), the values flagged 1 are treated as
filled too.

## 1. Cross sections: `river_profiles_raster.csv`

One row per cross section, ordered by chainage.

| column | meaning |
|---|---|
| `X_m` | chainage of the section (m) |
| `d_0 ... d_N` | transverse offset of each point across the river (m) |
| `h_0 ... h_N` | bed elevation at that offset (m, survey datum) |

Rows may have fewer valid points than columns; empty cells are skipped. For
the Seine the file is cut from a DEM along the river centreline, every 100 m,
over 500 m on each side (1378 sections x 168 points):

```bash
python tools/raster_to_transects.py \
    --raster data/seine/flood_plain_seine_compressed_DEFLATE2.tif \
    --centerline data/seine/seine_centerline.shp \
    --out data/seine/river_profiles_raster.csv --reverse
```

## 2. Boundary conditions and tributaries

| file | columns | role |
|---|---|---|
| `Q_in_vernon.csv` | `timestamp`, `discharge` (m3/s) | discharge entering the reach at x = 0 (hourly) |
| `boundary_downstream.csv` | `timestamp`, `stage` (m) | stage at x = L, imposed exactly (5 min, reduced to hourly) |
| `tributaries.csv` | `timestamp`, one column per tributary (m3/s) | lateral inflows (Andelle, Eure, Cailly) |

The column names are set in the case file (`value_col`, `tributaries[].column`).

## 3. Gauges: `gage_stations/`

One file per station with the columns `timestamp`, `stage_m` and `X_m`, at
the native sampling of the station (5 or 10 min). The case keeps one reading
per station and hour (`gauge_step_h: 1.0`). A single CSV with all stations in
the same columns also works.

| station code | chainage (km) | station |
|---|---|---|
| H3270001 | 6.2 | Pont-de-l'Arche |
| H5010110 | 16.5 | Elbeuf |
| H5010120 | 27.7 | Oissel |
| H5030110 | 42.1 | Rouen |
| H5050110 | 51.0 | Petit-Couronne |
| H5050120 | 59.0 | La Bouille |
| H5050130 | 64.5 | Mauny |
| H5051040 | 77.5 | Duclair |
| H5100110 | 85.4 | Mesnil-sous-Jumièges |
| H5100120 | 97.0 | Heurteauville |
| H5120110 | 110.1 | Caudebec-en-Caux |
| H5120120 | 116.7 | Vatteville-la-Rue |
| H5120130 | 123.3 | Aizier |
| H5120140 | 130.4 | Petiville |

## Sources

- Stage records and discharges: French hydrometric services (HydroPortail,
  Hub'Eau), Voies navigables de France (VNF) and the port of Rouen (GPMR).
- Downstream boundary (Tancarville): to be completed.
- Bathymetry and floodplain DEM: to be completed.

These data are subject to the terms of their providers and are not
redistributed here.

## Synthetic case

`tools/make_synthetic_case.py` writes everything the synthetic case needs
into `data/synthetic/`: the bathymetry, the forcing, five gauges (stage with
noise, plus the true velocity, only used to score the model) and
`truth_grid.npz`, the full true field used by `run_evaluate.py`.
