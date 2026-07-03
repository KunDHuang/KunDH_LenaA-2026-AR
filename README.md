# KunDH_LenaA-2026-AR
This repo hosts codes used in a Rheuma-Vor microbiome study by Kun D.H. and Lena A. et al.,

### Random Forest Machine Learning with Leave-One-Out Cross Validation
The modeling wrapper script `rf_mtx_response_pipeline.py` can be found in the `scripts` folder and the command line usage is shown as below: 
```bash
usage: rf_mtx_response_pipeline.py [-h] [--data-dir DATA_DIR] [--out-dir OUT_DIR] [--groups {RA,PsA} [{RA,PsA} ...]]
                                   [--feature-sets {pathway_diversity,pathway_only,species_diversity,combined_pathway_species_diversity} [{pathway_diversity,pathway_only,species_diversity,combined_pathway_species_diversity} ...]]
                                   [--cutoffs [CUTOFFS ...]] [--n-permutations N_PERMUTATIONS] [--n-cores N_CORES] [--threshold THRESHOLD] [--top-n-plot TOP_N_PLOT] [--boruta-perc BORUTA_PERC]
                                   [--boruta-max-iter BORUTA_MAX_ITER] [--boruta-estimator-n BORUTA_ESTIMATOR_N] [--fallback-max-features FALLBACK_MAX_FEATURES] [--no-boruta] [--fast-grid]
                                   [--dry-run] [--overwrite] [--no-resume]

Random Forest MTX-response pipeline for RA/PsA microbiome feature sets.

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Directory containing input TSV files.
  --out-dir OUT_DIR     Output directory for results.
  --groups {RA,PsA} [{RA,PsA} ...]
                        Disease groups to run.
  --feature-sets {pathway_diversity,pathway_only,species_diversity,combined_pathway_species_diversity} [{pathway_diversity,pathway_only,species_diversity,combined_pathway_species_diversity} ...]
                        Feature sets to run.
  --cutoffs [CUTOFFS ...]
                        Top percentile cutoffs to re-run after all-feature ranking.
  --n-permutations N_PERMUTATIONS
                        Number of label permutations for each run.
  --n-cores N_CORES     Parallel workers. In SLURM, set this to $SLURM_CPUS_PER_TASK.
  --threshold THRESHOLD
                        Probability threshold for confusion matrix.
  --top-n-plot TOP_N_PLOT
                        Number of features in important-score plot.
  --boruta-perc BORUTA_PERC
                        Boruta perc parameter.
  --boruta-max-iter BORUTA_MAX_ITER
                        Boruta max_iter parameter.
  --boruta-estimator-n BORUTA_ESTIMATOR_N
                        RF trees for Boruta/fallback screening.
  --fallback-max-features FALLBACK_MAX_FEATURES
                        Maximum features retained by fallback RF screening when Boruta is unavailable/rejects all.
  --no-boruta           Disable Boruta and use all input features inside each fold.
  --fast-grid           Use a smaller RF tuning grid; useful for testing.
  --dry-run             Load inputs and print shapes without running models.
  --overwrite           Overwrite completed runs.
  --no-resume           Do not skip completed runs.
```

### PERMANOVA Test with Adjustment for Covariables 
The PERMANOVA test was conducted based on the core formula of function `adonis2` from R package `vegan`:

*To adjust the effects from covariables, covariables were first sequencially placed before main variable.*

```adonis2(matrix ~ co_var1 + co_var2 + main_var, data = metadata, permutations = 999)```
`co_var1`: The 1st covariable to be controlled, i.e., age.
`co_var2`: The 2nd covariable to be controlled, i.e., enthesitis.
`main_var`: The main variable to be assessed, i.e., MTX response.

The wrapper R script `permanova_pcoa.R` can be found in the folder `example_data` 
