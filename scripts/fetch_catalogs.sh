#!/bin/bash
# Fetch every catalogue the pipeline needs from the NASA Exoplanet Archive.
#
# These files are not committed: they are large, and the archive is the
# authoritative source.  Re-running this script reproduces them exactly.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/catalogs
mkdir -p "$OUT"

TAP="https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

fetch () {  # fetch <output-name> <ADQL query>
    local name="$1" query="$2"
    echo "fetching ${name} ..."
    curl -sS -G "$TAP" --data-urlencode "query=${query}" --data "format=csv" \
         -o "${OUT}/${name}"
    echo "  $(wc -l < "${OUT}/${name}") lines"
}

# Kepler DR24 threshold-crossing events: the labelled training benchmark.
fetch dr24_tce.csv "select * from q1_q17_dr24_tce"

# Kepler DR25 threshold-crossing events: used for reference parameters.
fetch dr25_tce.csv "select * from q1_q17_dr25_tce"

# Cumulative KOI table: cross-identification between KIC and Kepler planet names.
fetch koi_cumulative.csv "select kepid,kepoi_name,kepler_name,koi_disposition,\
koi_pdisposition,koi_score,koi_tce_plnt_num,koi_period,koi_period_err1,koi_duration,\
koi_duration_err1,koi_depth,koi_depth_err1,koi_prad,koi_model_snr,koi_fpflag_nt,\
koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec from cumulative"

# Published parameters of confirmed Kepler planets: the independent reference
# against which recovered transit parameters are assessed.
fetch confirmed_pscomppars.csv "select pl_name,hostname,disc_facility,pl_orbper,\
pl_orbpererr1,pl_trandur,pl_trandurerr1,pl_trandep,pl_trandeperr1,pl_rade,st_rad,\
st_teff from pscomppars where disc_facility like '%Kepler%'"

# TESS Objects of Interest: the cross-mission evaluation set.
fetch toi.csv "select toi,tid,tfopwg_disp,pl_tranmid,pl_tranmiderr1,pl_orbper,\
pl_orbpererr1,pl_trandurh,pl_trandurherr1,pl_trandep,pl_rade,st_tmag,st_teff,\
st_rad,st_logg,sectors from toi"

echo "done -> ${OUT}"
