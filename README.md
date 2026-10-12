# Plasma proteomic signatures of diet quality and digestive disease (UK Biobank)

Analysis code for the manuscript:

> **Plasma proteomic signatures of diet quality and risk of gallstone disease, inflammatory bowel disease and acute pancreatitis: a prospective cohort study.**

The study relates four dimensions of diet quality, assessed with Oxford WebQ 24-h dietary recalls, to 2,919 Olink plasma proteins. The four dimensions are ultra-processed food (UPF, classified item by item with NOVA), the healthful and unhealthful plant-based diet indices (hPDI, uPDI) and energy-adjusted dietary fibre. The code:

- builds an elastic-net proteomic signature for each dimension and a competing BMI signature;
- tests the signatures against incident inflammatory bowel disease, gallstone disease and acute pancreatitis in participants not used for training. The tests include sequential adjustment, negative-control outcomes, a pre-specified specificity rule, diet-specific (residual) signatures, repeat proteomic measurements, dose–response, subgroup, prediction, substitution and competing-risk analyses;
- examines intestinal epithelial proteins (GUCA2A, DDC, FABP2, FABP6, CDH17) with mediation analysis, cis-pQTL Mendelian randomisation and colocalisation (UKB-PPP, FinnGen R12).

**No individual-level data are included in this repository.** UK Biobank data are available to approved researchers on application (https://www.ukbiobank.ac.uk); this work used application number 1099827. FinnGen and UKB-PPP summary statistics are publicly available. The only data file in the repository is `data_prep/nova_classification.csv`, the item-level NOVA classification used to compute UPF intake (also provided as Data S1 of the paper).

## Repository layout

| Folder | Script | Purpose |
|---|---|---|
| `data_prep/` | `01_extract.py` | Extract 24-h recall food groups and nutrients, population characteristics, baseline covariates and outcome dates from the UK Biobank tables |
| | `02_build.py` | Valid recalls (not flagged by field 100026; total energy 500–3,500 kcal for women and 500–4,200 kcal for men), hPDI, uPDI, energy-adjusted fibre, covariates and outcomes |
| | `03_biochem_olink_qc_extra.py` | Blood biochemistry, Olink sample metadata (plate, consortium selection), waist, GGT, triglycerides and negative-control outcome dates |
| | `04_nova_items.py`, `nova_classification.csv` | Item-level NOVA classification, energy per serving by non-negative least squares, UPF (% energy); `upper`/`lower` scenarios for the uncertain items |
| `analysis/` | `common.py` | Paths, covariate sets (Models 1–3), outcome definitions, Cox helper |
| | `F01_prepare.py` | Person-level table; Olink quality control, plate-mean correction and standardisation |
| | `F02_diet_proteome.py` | Diet–protein associations (2,919 proteins), food-group drivers |
| | `F03_signatures.py` | Elastic-net diet and BMI signatures (70/30 split) |
| | `F04_cox.py` | Signatures and self-reported scores versus outcomes (Models 1–3); pre-specified specificity check |
| | `F05_sensitivity.py` | Sensitivity analyses and proportional-hazards tests |
| | `F06_mediation.py`, `F06b_protein_outcomes.py` | Mediation by intestinal epithelial proteins; proteins versus outcomes |
| | `F10a_extract_extra.py`, `F10_tier1.py` | Gallstone-related confounders, alternative gallstone definitions, quintiles, splines, complete-case and multiple-imputation analyses, E-values |
| | `F11_residual_repeat.py` | Diet-specific (residual) signatures; within-person stability from repeat proteomics |
| | `F12a_ukbppp_instruments.py`, `F12_mr.py`, `F13_coloc_mr_full.py` | cis-pQTL Mendelian randomisation and GUCA2A colocalisation |
| | `F14_tier3.py`, `F14b_finegray.R` | Subgroups, C-index, isocaloric substitution; Fine–Gray competing-risk models |
| | `F15_nova_sensitivity.py` | UPF results under the extreme NOVA scenarios |
| | `F07_figures.py`, `FF_figures.py`, `FF_toc_graphic.py`, `F08_tables.py` | Figures (journal versions at ≤17.1 cm width), table-of-contents graphic, main and supplementary tables |

## Configuration

Directories are set with environment variables (defaults in brackets):

- `UKB_DATA_DIR`: UK Biobank dataset CSV extracts, one file per Research Analysis Platform table (`../ukb_data`). The scripts expect the file names used in our project (`UKB619657_online_followup_2025Nov.csv`, `UKB619657_population_characteristics_2025Nov.csv`, `UKB619657_assessment_center_2025Nov.csv`, `UKB619657_biological_samples_2025Nov.csv`, `UKB619657_health_outcomes_2025Nov.csv`); rename to match your own extracts. Columns follow the RAP convention `p<field>_i<instance>` / `p<field>_a<array>`.
- `UKB_OLINK_DIR`: Olink NPX tables, wide format with `Participant.ID` and one column per protein, named `olink_instance_0.csv`, `olink_instance_2.csv` and `olink_instance_3.csv` (`$UKB_DATA_DIR/olink`).
- `UKB_EXTRACT_DIR`: cache written by `data_prep/` and read by `analysis/` (`../ukb_extracts`).
- `DIET_PROJECT_DIR`: output root; `cache/`, `results/`, `figures/` and `manuscript/` are created inside (`.`). The companion repository [ssb-biliary-pancreatic-proteomics-ukb](https://github.com/liuxiaoqianghusina/ssb-biliary-pancreatic-proteomics-ukb) reads the person table, quality-controlled Olink matrix and signature split from this directory.

Mendelian randomisation inputs go in `$DIET_PROJECT_DIR/data/mr/`:

- `finngen_r12/finngen_R12_<code>.gz`: FinnGen R12 summary statistics for K11_CHOLELITH, K11_IBD_STRICT, K11_CD_STRICT2, K11_UC_STRICT2 and K11_ACUTPANC (https://www.finngen.fi/en/access_results);
- `ukbppp/Sun2023_SupplementaryTables.xlsx`: Supplementary Information workbook of Sun et al., *Nature* 2023 (UKB-PPP). `F12a_ukbppp_instruments.py` extracts Supplementary Tables 9 and 16 for the five proteins;
- `ukbppp/GUCA2A_chr1/`: the extracted GUCA2A chromosome 1 file from the UKB-PPP European discovery summary statistics on Synapse (syn51365303; GUCA2A file syn51470098), used for colocalisation.

## Running order

`run_all.sh` runs the whole pipeline in this order:

```
data_prep/01_extract.py
data_prep/02_build.py
data_prep/03_biochem_olink_qc_extra.py
data_prep/04_nova_items.py main | upper | lower
analysis/F01_prepare.py ... F06b_protein_outcomes.py
analysis/F10a_extract_extra.py, F10_tier1.py, F11_residual_repeat.py, F14_tier3.py
Rscript analysis/F14b_finegray.R
analysis/F15_nova_sensitivity.py
analysis/F12a_ukbppp_instruments.py, F12_mr.py, F13_coloc_mr_full.py
analysis/F07_figures.py, FF_figures.py, FF_toc_graphic.py, F08_tables.py
```

Run the `analysis/` scripts from inside `analysis/` (they import `common.py`). The random seed is fixed (2026) so the signature training split is reproducible.

## Software

Python 3.14 (pandas 2.3, numpy 2.4, scipy 1.17, lifelines 0.30, scikit-learn 1.9, statsmodels 0.14, patsy 1.0, matplotlib 3.11, joblib 1.5, openpyxl 3.1, Pillow 12, python-docx) and R 4.2.2 (survival, data.table). See `requirements.txt`. The MR scripts call `gzip`, `awk` and `grep`.

## License

MIT (see `LICENSE`).
