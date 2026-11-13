# Decomposing Double Standards in International Law: replication package

Yair Listokin and Matanel Bareli. Revised manuscript submitted to the *Journal of Law and Empirical Analysis* (JLEA-25-0056), November 2026.

This repository contains the code, the derived data, the classifier outputs and the results behind every number in the revised paper. It does not contain the texts of the General Assembly resolutions or the third-party datasets, which are available from their owners (see "Data not included").

## What is here

```
code/                      Python scripts, one per task, numbered in the order they were written
data/processed/            data produced by the scripts (counts, panels, content codes, classifier labels)
Israel and the UN/Data/    the paper's country-year panel and the per-resolution GPT-4o mini labels
                           (the two classifier files are reduced copies without the resolution texts)
output/tables/             every result table and run log, as produced
output/figures/            Figures 1 and 2
MANIFEST.csv               every file, its size, sha256 checksum and role
requirements.txt           Python packages
```

The folder names follow the layout the scripts expect, so each script runs unchanged from the repository root, for example `python3 code/19_corrected_tables.py`.

## Where each result comes from

| In the paper | Script | Output |
|---|---|---|
| Classification against UN Watch: Figure 1, F1 scores, counts in Section III.A | `41_unwatch_validation_fixed.py` | `output/tables/41_*`, `output/figures/41_figure1_confusion_matrix.png` |
| Corrected country-year counts | `19_corrected_tables.py` | `data/processed/19_counts_corrected.csv` |
| Tables 1, 2 and 3; Appendix Table B | `19_corrected_tables.py`, `19_run_bootstrap.sh` | `output/tables/19_*` |
| Standard error of the unexplained part | `36_unexplained_se.py` | `output/tables/36_*` |
| Resampling schemes (Section III.D); hurdle model (Appendix Table F) | `20_hurdle_and_bootstrap.py`, `20_run_bootstrap.sh` | `output/tables/20_*` |
| Added controls, OECD and democracy splits (Section III.E; Appendix Tables A, A2, D) | `21_referee1_controls_and_democracy.py`, `21_run_bootstrap.sh` | `output/tables/21_*` |
| EPR discrimination (Section III.F) | `05_epr_discrimination_check.py` | `output/tables/05_*` |
| First use of force (Section III.F) | `10_first_use_of_force_check.py` | `output/tables/10_*` |
| Content of the criticism (Table 5) | `12_build_reading_set.py`, `13_reading_packets.py`, `14_classification_summary.py` | `data/processed/14_row_codes.csv`, `output/tables/14_*` |
| Gravity variables in the decomposition, comparison with South Africa, case list (Table 4, Figure 2, Appendix Table C) | `22_referee2_package.py`, `22_run_bootstrap.sh` | `output/tables/22_*`, `output/figures/22_*` |
| Second classifier, DeepSeek V4 Pro (Section III.A; Appendix Table E) | `24_deepseek_reclassify.py`, `28_deepseek_decomposition.py`, `39_deepseek_f1.py` | `data/processed/24_deepseek_labels.csv`, `output/tables/24_*`, `28_*`, `39_*` |
| Blind second reading of the content codes | `29_deepseek_content.py` | `data/processed/29_deepseek_content.csv`, `output/tables/29_*` |

Bootstrap results use 500 replications with numpy seeds 0 to 499; the `*_run_bootstrap.sh` files run them in chunks and then summarise.

## The classifiers

Each resolution's full text was sent to GPT-4o mini (OpenAI) through the API in August 2024, temperature 0.5, one call per resolution, with this prompt:

> You are an AI assistant with expertise in international relations, law, and political science. Your task is to analyze the text of resolutions issued by the United Nations General Assembly and identify if a specific country is being condemned or criticized by the UN resolution. For each General Assembly Resolution I provide, please return only the name of the specific country being criticized. If no specific country is being criticized, return "None": {text}

The same prompt and texts were later sent to DeepSeek V4 Pro. `data/processed/24_deepseek_labels.csv` gives both labels for every resolution (`gpt_label`, `deepseek_label`; an empty `gpt_label` means "None"). The content codes for the 1,682 resolution-country pairs read in full are in `data/processed/14_row_codes.csv`, and the second, blind reading in `data/processed/29_deepseek_content.csv`.

## Data not included

Place these files at the paths shown to re-run the scripts that need them.

| File (path the scripts expect) | Source | Needed by |
|---|---|---|
| `data/raw/EPR-2021.csv`, `data/raw/EPR-2023.csv` | Ethnic Power Relations, EPR Core 2021 and 2023 (ETH Zurich) | 05, 22 |
| `data/raw/MID-5-Data-and-Supporting-Materials.zip` | Correlates of War, Militarized Interstate Disputes 5.0 | 10 |
| `data/raw/vdem_v16.RData` | V-Dem Country-Year Dataset v16 | 07, 21 |
| `data/raw/KOFGI_2025_public.xlsx` | KOF Globalisation Index, 2025 release | 21 |
| `data/raw/dpi2023/` | Database of Political Institutions 2023 (IDB) | 21 |
| `data/raw/WRP_national.csv` | World Religion Project, national data (Correlates of War) | 21 |
| `data/raw/ML indices.xlsx` | Gründler and Krieger machine-learning democracy indices | 21 |
| `Israel and the UN/Data/UN Watch GA Resolutions 2015-August 2024.xlsx` | UN Watch, General Assembly resolutions database (shared with researchers on request) | 41 |
| Resolution texts, 1981-1993 | Mesquita and Pires (2024) | 12, 13, 24 |
| Resolution texts, 1994-2023 | Voelsen, Bochtler and Majewski (2024) | 12, 13, 24 |

Scripts 12, 13 and 24 need the resolution texts, and 24 also an API key; their outputs are included, so every later script runs without them. The paper's panel, `Israel and the UN/Data/final_data.dta`, was built by the Stata files of the submitted version from the sources listed in the paper.

## Requirements

Python 3.10 or later; `pip install -r requirements.txt`. The core estimators are implemented in `code/panel_tools.py` with numpy only.

## Citation

Listokin, Yair, and Matanel Bareli. "Decomposing Double Standards in International Law." *Journal of Law and Empirical Analysis* (forthcoming).

## License

[To be decided before the repository is made public.]
