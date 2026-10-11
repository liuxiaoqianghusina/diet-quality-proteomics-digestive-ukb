#!/bin/bash
# Full pipeline. Set DIET_PROJECT_DIR, UKB_EXTRACT_DIR, UKB_DATA_DIR and UKB_OLINK_DIR first (see README).
# The NOVA classification (data_prep/nova_classification.csv) is the source of truth for UPF: after changing it,
# re-run 04_nova_items.py (all three scenarios) and the analysis scripts from F01 onwards.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="${DIET_PROJECT_DIR:-.}/results"; mkdir -p "$LOG"

cd "$HERE/data_prep"
python3 -u 01_extract.py                       > "$LOG/log_01_extract.txt" 2>&1
python3 -u 02_build.py                         > "$LOG/log_02_build.txt" 2>&1
python3 -u 03_biochem_olink_qc_extra.py        > "$LOG/log_03_biochem.txt" 2>&1
for sc in main upper lower; do
  python3 -u 04_nova_items.py $sc              > "$LOG/log_04_nova_$sc.txt" 2>&1
done

cd "$HERE/analysis"
for s in F01_prepare F02_diet_proteome F03_signatures F04_cox F05_sensitivity F06_mediation F06b_protein_outcomes \
         F10a_extract_extra F10_tier1 F11_residual_repeat F14_tier3; do
  echo "### $s $(date +%H:%M)"; python3 -u $s.py > "$LOG/log_$s.txt" 2>&1
done
Rscript F14b_finegray.R                         > "$LOG/log_F14b.txt" 2>&1
python3 -u F15_nova_sensitivity.py              > "$LOG/log_F15.txt" 2>&1
# Mendelian randomisation and colocalisation: needs FinnGen R12 and UKB-PPP files in $DIET_PROJECT_DIR/data/mr (README)
python3 -u F12a_ukbppp_instruments.py           > "$LOG/log_F12a.txt" 2>&1
python3 -u F12_mr.py                            > "$LOG/log_F12.txt" 2>&1
python3 -u F13_coloc_mr_full.py                 > "$LOG/log_F13.txt" 2>&1
# figures and tables
python3 F07_figures.py fig1 fig2 fig3 fig4 figS1 > "$LOG/log_F07.txt" 2>&1
python3 FF_figures.py                           > "$LOG/log_FF_figures.txt" 2>&1
python3 FF_toc_graphic.py                       >> "$LOG/log_FF_figures.txt" 2>&1
python3 F08_tables.py                           > "$LOG/log_F08.txt" 2>&1
echo "ALLDONE $(date +%H:%M)"
