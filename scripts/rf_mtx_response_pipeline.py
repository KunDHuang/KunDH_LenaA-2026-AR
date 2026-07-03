#!/usr/bin/env python3
"""
Random Forest MTX-response pipeline for RA and PsA
==================================================
"""

from __future__ import annotations

# ---- XXXXXXXXX ----
import os
for _thread_var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")

import argparse
import json
import math
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold

try:
    from boruta import BorutaPy
    HAS_BORUTA = True
except Exception:  # pragma: no cover - only used when Boruta is unavailable
    BorutaPy = None
    HAS_BORUTA = False

warnings.filterwarnings("ignore")

RNG = 42

PROJECT_DIR = Path(
    "/vol/projects/psivapor/PMIG_project/BioinfoHelper/"
    "RheumaVOR_revise_Arthritis_Rheumatology_2026"
)
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUT_DIR = PROJECT_DIR / "result" / "rf_mtx_response_v3"

LABEL_COL = "MTX_response"
POSITIVE_CLASS = "inefficiency"
NEGATIVE_CLASS = "remission"
CLASS_LABELS = [NEGATIVE_CLASS, POSITIVE_CLASS]
DIVERSITY_FEATURES = {"observed", "shannon"}

# Metadata rows found in the uploaded RA/PsA tables.  The loader uses names,
# not a fixed column count, because RA contains MTX_response_subclustering and
# PsA may not.
METADATA_ROWS = {
    "patient-a_partner-b",
    "time_point",
    "diagnose",
    "collection_date",
    "difference_to_first_sample (in months)",
    "RF",
    "ACPA",
    "anti-CD74_qualitative",
    "HLA-B27",
    "gender",
    "age",
    "MTX_treatment",
    "MTX_response",
    "MTX_response_subclustering",
    "other_therapy",
    "clustering_other_therapy",
}

DEFAULT_CUTOFFS = [5, 10, 15, 25, 50, 75, 80, 90, 95, 99]

# Kept close to the previous scripts.
DEFAULT_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 4, 6],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", 0.3],
}


@dataclass(frozen=True)
class InputFiles:
    group: str
    pathway: Path
    species: Path


@dataclass
class FeatureSetData:
    group: str
    feature_set: str
    X: pd.DataFrame
    y: pd.Series
    feature_types: Dict[str, str]
    source_files: Dict[str, str]


@dataclass
class RunResult:
    summary: pd.DataFrame
    feature_importance: pd.DataFrame
    run_dir: Path


# =============================================================================
# Generic helpers
# =============================================================================

def tsv(df: pd.DataFrame, path: Path) -> None:
    """Write TSV atomically where possible.

    This prevents partially written global summary files if several SLURM array
    tasks finish at nearly the same time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def safe_name(x: object) -> str:
    s = str(x)
    for ch in ["/", "\\", " ", ":", "|", "(", ")", "[", "]", "{", "}", ",", ";"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def make_unique(names: Sequence[object]) -> List[str]:
    """Return unique strings while preserving the original order."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for value in names:
        base = str(value)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__dup{seen[base]}")
    return out


def zero_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else float("nan")


def parse_cutoffs(values: Optional[Sequence[str]]) -> List[float]:
    if values is None:
        values = [str(v) for v in DEFAULT_CUTOFFS]
    out: List[float] = []
    for v in values:
        val = float(v)
        if val <= 0 or val > 100:
            raise ValueError(f"Cutoff percentile must be >0 and <=100, got {v}")
        out.append(val)
    return sorted(set(out))


def cutoff_label(cutoff: float) -> str:
    if abs(cutoff - int(cutoff)) < 1e-9:
        return f"top{int(cutoff):03d}pct"
    return f"top{str(cutoff).replace('.', 'p')}pct"


def get_input_files(data_dir: Path) -> Dict[str, InputFiles]:
    return {
        "RA": InputFiles(
            group="RA",
            pathway=data_dir / "humann3_aggre_pathways_relab_initial_RA_md_excluding_noMTX_3samples_with_diversity.tsv",
            species=data_dir / "mpa4_merged_relab_RA_sgb_md_excluding_noMTX_final_samples_with_diversity.tsv",
        ),
        "PsA": InputFiles(
            group="PsA",
            pathway=data_dir / "humann3_aggre_pathways_relab_initial_PsA_md_excluding_noMTX_with_diversity.tsv",
            species=data_dir / "mpa4_merged_relab_PsA_sgb_md_excluding_noMTX_final_samples_with_diversity.tsv",
        ),
    }


# =============================================================================
# Data loading and feature-set construction
# =============================================================================

def load_one_transposed_matrix(
    path: Path,
    feature_family: str,
    include_diversity: bool,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Load one feature table and return X, y, feature type map.

    Input format expected:
        first column named 'sample'; rows are metadata/features; columns are samples.

    The old scripts used:
        pd.read_csv(path, sep='\t').set_index('sample').T

    This loader keeps that structure, but drops metadata by row name rather than
    by a hard-coded number of metadata rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    raw = pd.read_csv(path, sep="\t", low_memory=False)
    if "sample" not in raw.columns:
        raise ValueError(f"Missing first/ID column 'sample' in {path}")

    raw = raw.copy()
    raw["sample"] = make_unique(raw["sample"].tolist())
    df = raw.set_index("sample").T

    if LABEL_COL not in df.columns:
        raise ValueError(f"{LABEL_COL!r} not found in {path}")

    y = df[LABEL_COL].astype(str).str.strip()

    feature_cols: List[str] = []
    feature_types: Dict[str, str] = {}
    for col in df.columns:
        if col in METADATA_ROWS:
            continue
        is_div = col in DIVERSITY_FEATURES
        if is_div and not include_diversity:
            continue
        feature_cols.append(col)
        feature_types[col] = "diversity" if is_div else feature_family

    if not feature_cols:
        raise ValueError(f"No feature columns found in {path}")

    X = df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")

    # Keep only valid MTX response labels.
    valid = y.isin([POSITIVE_CLASS, NEGATIVE_CLASS])
    X = X.loc[valid].fillna(0.0)
    y = y.loc[valid]

    # Drop features with no variation after filtering samples.
    nunique = X.nunique(dropna=False)
    keep = nunique > 1
    X = X.loc[:, keep]
    feature_types = {c: feature_types[c] for c in X.columns}

    # RandomForest does not care about sample order, but reproducible outputs do.
    X = X.sort_index()
    y = y.loc[X.index]

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Only one valid class found in {path}; counts={y.value_counts().to_dict()}"
        )

    return X, y, feature_types


def assert_same_labels(y1: pd.Series, y2: pd.Series, context: str) -> None:
    common = y1.index.intersection(y2.index)
    if common.empty:
        raise ValueError(f"No common samples found for {context}")
    mismatched = common[y1.loc[common].astype(str).values != y2.loc[common].astype(str).values]
    if len(mismatched):
        preview = ", ".join(map(str, mismatched[:5]))
        raise ValueError(f"MTX_response mismatch in {context}; examples: {preview}")


def load_feature_set(group: str, feature_set: str, data_dir: Path) -> FeatureSetData:
    inputs = get_input_files(data_dir)[group]

    if feature_set == "pathway_diversity":
        X, y, feature_types = load_one_transposed_matrix(
            inputs.pathway, feature_family="pathway", include_diversity=True
        )
        sources = {"pathway": str(inputs.pathway)}

    elif feature_set == "pathway_only":
        X, y, feature_types = load_one_transposed_matrix(
            inputs.pathway, feature_family="pathway", include_diversity=False
        )
        sources = {"pathway": str(inputs.pathway)}

    elif feature_set == "species_diversity":
        X, y, feature_types = load_one_transposed_matrix(
            inputs.species, feature_family="species", include_diversity=True
        )
        sources = {"species": str(inputs.species)}

    elif feature_set == "combined_pathway_species_diversity":
        # Optional feature set, not used by default.  It combines pathway,
        # species, and one copy of observed/shannon.  Feature prefixes avoid name
        # collisions across modalities.
        Xp, yp, tp = load_one_transposed_matrix(
            inputs.pathway, feature_family="pathway", include_diversity=False
        )
        Xs, ys, ts = load_one_transposed_matrix(
            inputs.species, feature_family="species", include_diversity=False
        )
        Xd, yd, td = load_one_transposed_matrix(
            inputs.pathway, feature_family="pathway", include_diversity=True
        )
        div_cols = [c for c in Xd.columns if c in DIVERSITY_FEATURES]
        Xd = Xd.loc[:, div_cols]
        td = {c: "diversity" for c in div_cols}

        assert_same_labels(yp, ys, f"{group} combined pathway/species")
        assert_same_labels(yp, yd, f"{group} combined pathway/diversity")
        common = yp.index.intersection(ys.index).intersection(yd.index)
        yp = yp.loc[common]
        Xp = Xp.loc[common].add_prefix("pathway|")
        Xs = Xs.loc[common].add_prefix("species|")
        Xd = Xd.loc[common].add_prefix("diversity|")
        X = pd.concat([Xp, Xs, Xd], axis=1)
        y = yp
        feature_types = {}
        feature_types.update({f"pathway|{k}": "pathway" for k in tp if k in Xp.columns.str.replace("pathway|", "", regex=False)})
        feature_types.update({c: "pathway" for c in Xp.columns})
        feature_types.update({c: "species" for c in Xs.columns})
        feature_types.update({c: "diversity" for c in Xd.columns})
        sources = {"pathway": str(inputs.pathway), "species": str(inputs.species)}

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # Final guard against duplicate column labels.
    if X.columns.duplicated().any():
        old_cols = X.columns.tolist()
        new_cols = make_unique(old_cols)
        rename_map = dict(zip(old_cols, new_cols))
        X.columns = new_cols
        feature_types = {rename_map.get(k, k): v for k, v in feature_types.items()}

    # Drop constants again after any merge/prefix step.
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]
    feature_types = {c: feature_types.get(c, "unknown") for c in X.columns}

    print(
        f"[load] {group} {feature_set}: X={X.shape}, "
        f"y={y.value_counts().to_dict()}, sources={sources}"
    )
    return FeatureSetData(group, feature_set, X, y, feature_types, sources)


# =============================================================================
# Core ML pipeline
# =============================================================================

def build_rf_for_boruta(n_estimators: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=1,
        class_weight="balanced",
        max_depth=5,
        random_state=random_state,
    )


def fallback_feature_screen(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    max_features: int,
    n_estimators: int,
) -> np.ndarray:
    """RF-importance fallback when Boruta is unavailable or rejects everything."""
    p = X.shape[1]
    n_keep = min(max_features, p)
    rf = build_rf_for_boruta(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    mask = np.zeros(p, dtype=bool)
    mask[order[:n_keep]] = True
    return mask


def run_preselector(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    use_boruta: bool,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
) -> Tuple[np.ndarray, str]:
    p = X.shape[1]
    if p == 0:
        raise ValueError("No features were supplied to preselector")

    if not use_boruta:
        return np.ones(p, dtype=bool), "none_all_features"

    if HAS_BORUTA and BorutaPy is not None:
        try:
            rf_for_boruta = build_rf_for_boruta(
                n_estimators=boruta_estimator_n,
                random_state=random_state,
            )
            boruta = BorutaPy(
                estimator=rf_for_boruta,
                n_estimators="auto",
                perc=boruta_perc,
                max_iter=boruta_max_iter,
                random_state=random_state,
                verbose=0,
            )
            boruta.fit(X, y)
            mask = boruta.support_ | boruta.support_weak_
            if int(mask.sum()) > 0:
                return mask, "boruta_confirmed_or_tentative"
        except Exception:
            # Fall through to RF fallback, but keep the whole run alive.
            pass

    mask = fallback_feature_screen(
        X=X,
        y=y,
        random_state=random_state,
        max_features=fallback_max_features,
        n_estimators=boruta_estimator_n,
    )
    return mask, "rf_importance_fallback"


def tune_rf(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    param_grid: Dict[str, Sequence[object]],
) -> Tuple[RandomForestClassifier, Dict[str, object]]:
    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )

    counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(counts[counts > 0].min())
    n_splits = min(3, min_class)

    if n_splits < 2:
        # This should not happen for the current RA/PsA data, but it keeps the
        # script robust to very small future subsets.
        params = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
        model = RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(X, y)
        return model, params

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    gs.fit(X, y)
    return gs.best_estimator_, dict(gs.best_params_)


def _one_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_arr: np.ndarray,
    y_enc: np.ndarray,
    random_state: int,
    use_boruta: bool,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
    collect_details: bool,
) -> Dict[str, object]:
    X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
    y_tr = y_enc[train_idx]

    fold_seed = random_state + fold_idx
    mask, preselector = run_preselector(
        X_tr,
        y_tr,
        random_state=fold_seed,
        use_boruta=use_boruta,
        boruta_perc=boruta_perc,
        boruta_max_iter=boruta_max_iter,
        boruta_estimator_n=boruta_estimator_n,
        fallback_max_features=fallback_max_features,
    )

    X_tr_sel = X_tr[:, mask]
    X_te_sel = X_te[:, mask]

    best_rf, best_params = tune_rf(
        X_tr_sel,
        y_tr,
        random_state=fold_seed,
        param_grid=param_grid,
    )

    pos_col = list(best_rf.classes_).index(1)
    proba = float(best_rf.predict_proba(X_te_sel)[0, pos_col])

    out: Dict[str, object] = {
        "fold_idx": int(fold_idx),
        "test_i": int(test_idx[0]),
        "proba": proba,
    }
    if collect_details:
        out.update(
            {
                "mask": mask,
                "importances": best_rf.feature_importances_.astype(float),
                "params": best_params,
                "preselector": preselector,
            }
        )
    return out


def nested_loo(
    X: pd.DataFrame,
    y_enc: np.ndarray,
    random_state: int,
    n_jobs: int,
    use_boruta: bool,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
    collect_details: bool,
) -> Tuple[np.ndarray, Optional[Dict[str, object]]]:
    loo = LeaveOneOut()
    X_arr = X.to_numpy(dtype=float)
    splits = list(loo.split(X_arr))
    n = X.shape[0]

    worker = delayed(_one_fold)
    if n_jobs == 1:
        results = [
            _one_fold(
                i,
                tr,
                te,
                X_arr,
                y_enc,
                random_state,
                use_boruta,
                boruta_perc,
                boruta_max_iter,
                boruta_estimator_n,
                fallback_max_features,
                param_grid,
                collect_details,
            )
            for i, (tr, te) in enumerate(splits)
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            worker(
                i,
                tr,
                te,
                X_arr,
                y_enc,
                random_state,
                use_boruta,
                boruta_perc,
                boruta_max_iter,
                boruta_estimator_n,
                fallback_max_features,
                param_grid,
                collect_details,
            )
            for i, (tr, te) in enumerate(splits)
        )

    oof_proba = np.zeros(n, dtype=float)
    fold_masks: List[np.ndarray] = []
    fold_importances: List[np.ndarray] = []
    fold_params: List[Dict[str, object]] = []
    fold_preselectors: List[str] = []
    fold_test_indices: List[int] = []

    for r in sorted(results, key=lambda d: int(d["fold_idx"])):
        test_i = int(r["test_i"])
        oof_proba[test_i] = float(r["proba"])
        if collect_details:
            fold_test_indices.append(test_i)
            mask = np.asarray(r["mask"], dtype=bool)
            fold_masks.append(mask)
            imp_full = np.full(X.shape[1], np.nan, dtype=float)
            imp_full[mask] = np.asarray(r["importances"], dtype=float)
            fold_importances.append(imp_full)
            fold_params.append(dict(r["params"]))
            fold_preselectors.append(str(r["preselector"]))

    if not collect_details:
        return oof_proba, None

    details = {
        "feature_names": X.columns.to_numpy(dtype=object),
        "sample_names": X.index.to_numpy(dtype=object),
        "fold_masks": np.vstack(fold_masks),
        "fold_importances": np.vstack(fold_importances),
        "fold_params": fold_params,
        "fold_preselectors": fold_preselectors,
        "fold_test_indices": fold_test_indices,
    }
    return oof_proba, details


def _one_permutation(
    perm_seed: int,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    use_boruta: bool,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
) -> float:
    rng = np.random.RandomState(perm_seed)
    y_shuf = rng.permutation(y_enc)
    oof, _ = nested_loo(
        X,
        y_shuf,
        random_state=perm_seed,
        n_jobs=1,
        use_boruta=use_boruta,
        boruta_perc=boruta_perc,
        boruta_max_iter=boruta_max_iter,
        boruta_estimator_n=boruta_estimator_n,
        fallback_max_features=fallback_max_features,
        param_grid=param_grid,
        collect_details=False,
    )
    return float(roc_auc_score(y_shuf, oof))


def permutation_pvalue(
    X: pd.DataFrame,
    y_enc: np.ndarray,
    observed_auc: float,
    n_perm: int,
    n_jobs: int,
    use_boruta: bool,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
) -> Tuple[float, np.ndarray, int]:
    if n_perm <= 0:
        return float("nan"), np.array([], dtype=float), 0

    seeds = np.arange(n_perm, dtype=int) + 10_000
    print(f"  permutations: n={n_perm}, n_jobs={n_jobs}", flush=True)
    null_aucs = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_one_permutation)(
            int(s),
            X,
            y_enc,
            use_boruta,
            boruta_perc,
            boruta_max_iter,
            boruta_estimator_n,
            fallback_max_features,
            param_grid,
        )
        for s in seeds
    )
    null = np.array([a for a in null_aucs if not np.isnan(a)], dtype=float)
    n_ge = int(np.sum(null >= observed_auc))
    p = float((n_ge + 1) / (len(null) + 1))
    return p, null, n_ge


# =============================================================================
# Tables and plots
# =============================================================================

def build_feature_importance_table(
    details: Dict[str, object],
    feature_types: Dict[str, str],
) -> pd.DataFrame:
    fnames = np.asarray(details["feature_names"], dtype=object)
    imps = np.asarray(details["fold_importances"], dtype=float)
    masks = np.asarray(details["fold_masks"], dtype=bool)
    n_folds = imps.shape[0]

    selected_counts = masks.sum(axis=0).astype(int)
    selection_frequency = selected_counts / float(n_folds)

    mean_when_selected = np.nanmean(imps, axis=0)
    mean_when_selected = np.where(np.isnan(mean_when_selected), 0.0, mean_when_selected)

    # Stability-weighted importance = importance averaged over all outer folds,
    # treating unselected folds as 0.  This is the ranking score used for the
    # percentile cutoff re-runs.
    importance_score = np.nan_to_num(imps, nan=0.0).mean(axis=0)

    out = pd.DataFrame(
        {
            "feature": fnames,
            "feature_type": [feature_types.get(str(f), "unknown") for f in fnames],
            "importance_score": importance_score,
            "mean_importance_when_selected": mean_when_selected,
            "selection_frequency": selection_frequency,
            "selected_in_n_folds": selected_counts,
            "n_outer_folds": n_folds,
        }
    )
    out = out.sort_values(
        [
            "importance_score",
            "mean_importance_when_selected",
            "selection_frequency",
            "feature",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out.insert(0, "rank_within_run", np.arange(1, len(out) + 1, dtype=int))
    return out


def prediction_and_confusion_tables(
    samples: Sequence[object],
    y_enc: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_enc, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    pred_df = pd.DataFrame(
        {
            "sample": list(samples),
            "true_label_encoded": y_enc.astype(int),
            "true_label": [POSITIVE_CLASS if v == 1 else NEGATIVE_CLASS for v in y_enc],
            "predicted_probability_inefficiency": y_proba.astype(float),
            "predicted_label_encoded": y_pred.astype(int),
            "predicted_label": [POSITIVE_CLASS if v == 1 else NEGATIVE_CLASS for v in y_pred],
            "classification_threshold": threshold,
        }
    )

    cm_df = pd.DataFrame(
        [
            {"true_label": NEGATIVE_CLASS, "predicted_label": NEGATIVE_CLASS, "count": tn},
            {"true_label": NEGATIVE_CLASS, "predicted_label": POSITIVE_CLASS, "count": fp},
            {"true_label": POSITIVE_CLASS, "predicted_label": NEGATIVE_CLASS, "count": fn},
            {"true_label": POSITIVE_CLASS, "predicted_label": POSITIVE_CLASS, "count": tp},
        ]
    )

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_enc, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_enc, y_pred)),
        "sensitivity_recall_inefficiency": zero_div(tp, tp + fn),
        "specificity_remission": zero_div(tn, tn + fp),
        "precision_ppv_inefficiency": float(precision_score(y_enc, y_pred, zero_division=0)),
        "npv_remission": zero_div(tn, tn + fn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return pred_df, cm_df, metrics


def set_plot_grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="grey", linewidth=0.6, alpha=0.45)
    ax.grid(True, which="minor", color="grey", linewidth=0.35, alpha=0.25)
    try:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    except Exception:
        pass
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)


def save_roc_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    auc_val: float,
    p_val: float,
    fig_path: Path,
    plot_data_path: Path,
    title: str,
) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    plot_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    tsv(plot_df, plot_data_path)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    label = f"AUROC = {auc_val:.3f}"
    if not np.isnan(p_val):
        label += f"\npermutation p = {p_val:.4f}"
    ax.plot(fpr, tpr, linewidth=2.3, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="grey")
    ax.set_xlabel("False positive rate", fontsize=14)
    ax.set_ylabel("True positive rate", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    set_plot_grid(ax)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_plot(
    importance_df: pd.DataFrame,
    fig_path: Path,
    plot_data_path: Path,
    title: str,
    top_n: int,
) -> None:
    plot_df = importance_df.sort_values("rank_within_run").head(top_n).copy()
    # Reverse so the highest ranked feature appears at the top after barh.
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    tsv(plot_df, plot_data_path)

    height = max(5.2, 0.34 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, height))
    ax.barh(plot_df["feature"], plot_df["importance_score"])
    ax.set_xlabel("Stability-weighted RF importance score", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    ax.set_title(title, fontsize=14)
    set_plot_grid(ax)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(
    cm_df: pd.DataFrame,
    fig_path: Path,
    plot_data_path: Path,
    title: str,
) -> None:
    tsv(cm_df, plot_data_path)
    matrix = np.array(
        [
            [
                int(cm_df.query("true_label == @NEGATIVE_CLASS and predicted_label == @NEGATIVE_CLASS")["count"].iloc[0]),
                int(cm_df.query("true_label == @NEGATIVE_CLASS and predicted_label == @POSITIVE_CLASS")["count"].iloc[0]),
            ],
            [
                int(cm_df.query("true_label == @POSITIVE_CLASS and predicted_label == @NEGATIVE_CLASS")["count"].iloc[0]),
                int(cm_df.query("true_label == @POSITIVE_CLASS and predicted_label == @POSITIVE_CLASS")["count"].iloc[0]),
            ],
        ],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(matrix, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Count", fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_LABELS, fontsize=12, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_LABELS, fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)
    ax.set_title(title, fontsize=14)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)

    # Grid lines at cell boundaries plus minor grid.
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(True, which="minor", color="grey", linewidth=0.8, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_null_distribution_plot(
    null_aucs: np.ndarray,
    observed_auc: float,
    p_val: float,
    fig_path: Path,
    plot_data_path: Path,
    title: str,
) -> None:
    if len(null_aucs) == 0:
        tsv(pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count"]), plot_data_path)
        return
    counts, edges = np.histogram(null_aucs, bins=40)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plot_df = pd.DataFrame(
        {"bin_left": edges[:-1], "bin_right": edges[1:], "bin_center": centers, "count": counts}
    )
    tsv(plot_df, plot_data_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(null_aucs, bins=40)
    label = f"Observed AUROC = {observed_auc:.3f}"
    if not np.isnan(p_val):
        label += f"\np = {p_val:.4f}"
    ax.axvline(observed_auc, linewidth=2.0, linestyle="--", label=label)
    ax.set_xlabel("AUROC under permuted labels", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(frameon=False, fontsize=12)
    set_plot_grid(ax)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fold_tables(
    details: Dict[str, object],
    samples: Sequence[object],
    feature_names: Sequence[object],
    feature_types: Dict[str, str],
    run_id: str,
    table_dir: Path,
) -> None:
    rows = []
    for fold_idx, (test_i, params, preselector) in enumerate(
        zip(details["fold_test_indices"], details["fold_params"], details["fold_preselectors"])
    ):
        row = {
            "run_id": run_id,
            "fold_idx": fold_idx,
            "test_sample": samples[int(test_i)],
            "preselector": preselector,
        }
        row.update({f"param_{k}": v for k, v in params.items()})
        rows.append(row)
    tsv(pd.DataFrame(rows), table_dir / f"{run_id}_FoldParameters.tsv")

    selected_rows = []
    masks = np.asarray(details["fold_masks"], dtype=bool)
    imps = np.asarray(details["fold_importances"], dtype=float)
    fnames = np.asarray(feature_names, dtype=object)
    for fold_idx in range(masks.shape[0]):
        selected_idx = np.where(masks[fold_idx])[0]
        for idx in selected_idx:
            feat = str(fnames[idx])
            selected_rows.append(
                {
                    "run_id": run_id,
                    "fold_idx": fold_idx,
                    "feature": feat,
                    "feature_type": feature_types.get(feat, "unknown"),
                    "fold_importance": imps[fold_idx, idx],
                }
            )
    tsv(pd.DataFrame(selected_rows), table_dir / f"{run_id}_SelectedFeaturesByFold.tsv")


# =============================================================================
# Run orchestration
# =============================================================================

def completed_run_paths(run_dir: Path, run_id: str) -> Tuple[Path, Path, Path]:
    table_dir = run_dir / "tables"
    summary_path = table_dir / f"{run_id}_Summary.tsv"
    importance_path = table_dir / f"{run_id}_FeatureImportance.tsv"
    complete_path = run_dir / "RUN_COMPLETE.ok"
    return summary_path, importance_path, complete_path


def load_completed_run(run_dir: Path, run_id: str) -> Optional[RunResult]:
    summary_path, importance_path, complete_path = completed_run_paths(run_dir, run_id)
    if complete_path.exists() and summary_path.exists() and importance_path.exists():
        return RunResult(read_tsv(summary_path), read_tsv(importance_path), run_dir)
    return None


def run_one_model(
    data: FeatureSetData,
    X_run: pd.DataFrame,
    feature_types_run: Dict[str, str],
    run_name: str,
    run_type: str,
    percentile_cutoff: Optional[float],
    global_rank_source: Optional[pd.DataFrame],
    out_dir: Path,
    n_permutations: int,
    n_jobs: int,
    threshold: float,
    top_n_plot: int,
    use_boruta: bool,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
    resume: bool,
    overwrite: bool,
) -> RunResult:
    run_id = f"{data.group}_{data.feature_set}_{run_name}"
    run_dir = out_dir / data.group / data.feature_set / run_name
    table_dir = run_dir / "tables"
    fig_dir = run_dir / "figures"
    plot_data_dir = run_dir / "plot_data"
    log_dir = run_dir / "logs"
    for d in [table_dir, fig_dir, plot_data_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if resume and not overwrite:
        completed = load_completed_run(run_dir, run_id)
        if completed is not None:
            print(f"[resume] skipping completed run: {run_id}", flush=True)
            return completed

    print("\n" + "=" * 90, flush=True)
    print(f"RUN: {run_id}", flush=True)
    print(f"  group={data.group}; feature_set={data.feature_set}; run_type={run_type}; cutoff={percentile_cutoff}", flush=True)
    print(f"  X={X_run.shape}; labels={data.y.value_counts().to_dict()}", flush=True)
    print("=" * 90, flush=True)

    # Log basic settings for reproducibility.
    settings = {
        "run_id": run_id,
        "group": data.group,
        "feature_set": data.feature_set,
        "run_type": run_type,
        "percentile_cutoff": percentile_cutoff,
        "source_files": data.source_files,
        "label_col": LABEL_COL,
        "positive_class_encoded_1": POSITIVE_CLASS,
        "negative_class_encoded_0": NEGATIVE_CLASS,
        "n_permutations": n_permutations,
        "n_jobs": n_jobs,
        "threshold": threshold,
        "use_boruta": use_boruta,
        "boruta_available": HAS_BORUTA,
        "boruta_perc": boruta_perc,
        "boruta_max_iter": boruta_max_iter,
        "boruta_estimator_n": boruta_estimator_n,
        "fallback_max_features": fallback_max_features,
        "param_grid": param_grid,
    }
    (table_dir / f"{run_id}_Settings.json").write_text(json.dumps(settings, indent=2, default=str))

    y_enc = (data.y.loc[X_run.index].values == POSITIVE_CLASS).astype(int)
    n_pos = int(y_enc.sum())
    n_neg = int(len(y_enc) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"{run_id}: need both classes; got pos={n_pos}, neg={n_neg}")

    t0 = time.time()
    print(f"  observed nested LOO-CV: n_jobs={n_jobs}", flush=True)
    oof_proba, details = nested_loo(
        X_run,
        y_enc,
        random_state=RNG,
        n_jobs=n_jobs,
        use_boruta=use_boruta,
        boruta_perc=boruta_perc,
        boruta_max_iter=boruta_max_iter,
        boruta_estimator_n=boruta_estimator_n,
        fallback_max_features=fallback_max_features,
        param_grid=param_grid,
        collect_details=True,
    )
    assert details is not None
    observed_auc = float(roc_auc_score(y_enc, oof_proba))
    print(f"  observed AUROC={observed_auc:.4f}", flush=True)

    p_val, null_aucs, n_null_ge = permutation_pvalue(
        X_run,
        y_enc,
        observed_auc=observed_auc,
        n_perm=n_permutations,
        n_jobs=n_jobs,
        use_boruta=use_boruta,
        boruta_perc=boruta_perc,
        boruta_max_iter=boruta_max_iter,
        boruta_estimator_n=boruta_estimator_n,
        fallback_max_features=fallback_max_features,
        param_grid=param_grid,
    )
    runtime_sec = float(time.time() - t0)
    print(f"  permutation p={p_val if not np.isnan(p_val) else 'NA'}; runtime={runtime_sec:.1f}s", flush=True)

    # Core output tables.
    pred_df, cm_df, cm_metrics = prediction_and_confusion_tables(
        samples=X_run.index,
        y_enc=y_enc,
        y_proba=oof_proba,
        threshold=threshold,
    )
    pred_df.insert(0, "run_id", run_id)
    pred_df.insert(1, "disease_group", data.group)
    pred_df.insert(2, "feature_set", data.feature_set)
    pred_df.insert(3, "run_type", run_type)
    pred_df.insert(4, "percentile_cutoff", percentile_cutoff)
    tsv(pred_df, table_dir / f"{run_id}_Predictions.tsv")

    cm_df.insert(0, "run_id", run_id)
    cm_df.insert(1, "disease_group", data.group)
    cm_df.insert(2, "feature_set", data.feature_set)
    cm_df.insert(3, "run_type", run_type)
    cm_df.insert(4, "percentile_cutoff", percentile_cutoff)
    tsv(cm_df, table_dir / f"{run_id}_ConfusionMatrix.tsv")

    perm_df = pd.DataFrame(
        {
            "run_id": run_id,
            "disease_group": data.group,
            "feature_set": data.feature_set,
            "run_type": run_type,
            "percentile_cutoff": percentile_cutoff,
            "permutation_id": np.arange(1, len(null_aucs) + 1, dtype=int),
            "null_auc": null_aucs,
            "observed_auc": observed_auc,
            "p_value": p_val,
        }
    )
    tsv(perm_df, table_dir / f"{run_id}_Permutation.tsv")

    pval_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "disease_group": data.group,
                "feature_set": data.feature_set,
                "run_type": run_type,
                "percentile_cutoff": percentile_cutoff,
                "observed_auc": observed_auc,
                "n_permutations_requested": n_permutations,
                "n_permutations_completed": int(len(null_aucs)),
                "n_null_auc_ge_observed": int(n_null_ge),
                "p_value": p_val,
            }
        ]
    )
    tsv(pval_df, table_dir / f"{run_id}_PValue.tsv")

    importance_df = build_feature_importance_table(details, feature_types_run)
    importance_df.insert(0, "run_id", run_id)
    importance_df.insert(1, "disease_group", data.group)
    importance_df.insert(2, "feature_set", data.feature_set)
    importance_df.insert(3, "run_type", run_type)
    importance_df.insert(4, "percentile_cutoff", percentile_cutoff)
    importance_df.insert(5, "n_features_input_to_run", X_run.shape[1])

    if global_rank_source is not None:
        gr = global_rank_source.loc[:, ["feature", "rank_within_run", "importance_score"]].copy()
        gr = gr.rename(
            columns={
                "rank_within_run": "rank_from_all_feature_run",
                "importance_score": "importance_score_from_all_feature_run",
            }
        )
        importance_df = importance_df.merge(gr, on="feature", how="left")
    else:
        importance_df["rank_from_all_feature_run"] = importance_df["rank_within_run"]
        importance_df["importance_score_from_all_feature_run"] = importance_df["importance_score"]

    tsv(importance_df, table_dir / f"{run_id}_FeatureImportance.tsv")
    save_fold_tables(details, X_run.index, X_run.columns, feature_types_run, run_id, table_dir)

    # Plot data and 300 dpi figures.  Plot-data file stems match figure stems.
    roc_stem = f"{run_id}_AUROC"
    save_roc_plot(
        y_true=y_enc,
        y_proba=oof_proba,
        auc_val=observed_auc,
        p_val=p_val,
        fig_path=fig_dir / f"{roc_stem}.png",
        plot_data_path=plot_data_dir / f"{roc_stem}.tsv",
        title=f"{data.group} {data.feature_set} {run_name}: AUROC",
    )

    imp_stem = f"{run_id}_ImportantScore"
    save_feature_importance_plot(
        importance_df=importance_df,
        fig_path=fig_dir / f"{imp_stem}.png",
        plot_data_path=plot_data_dir / f"{imp_stem}.tsv",
        title=f"{data.group} {data.feature_set} {run_name}: important score",
        top_n=top_n_plot,
    )

    cm_stem = f"{run_id}_ConfusionMatrix"
    save_confusion_matrix_plot(
        cm_df=cm_df,
        fig_path=fig_dir / f"{cm_stem}.png",
        plot_data_path=plot_data_dir / f"{cm_stem}.tsv",
        title=f"{data.group} {data.feature_set} {run_name}: confusion matrix",
    )

    perm_stem = f"{run_id}_PermutationNullAUROC"
    save_null_distribution_plot(
        null_aucs=null_aucs,
        observed_auc=observed_auc,
        p_val=p_val,
        fig_path=fig_dir / f"{perm_stem}.png",
        plot_data_path=plot_data_dir / f"{perm_stem}.tsv",
        title=f"{data.group} {data.feature_set} {run_name}: permutation null AUROC",
    )

    summary_row = {
        "run_id": run_id,
        "disease_group": data.group,
        "feature_set": data.feature_set,
        "run_name": run_name,
        "run_type": run_type,
        "percentile_cutoff": percentile_cutoff,
        "n_samples": int(X_run.shape[0]),
        "n_features_input_to_run": int(X_run.shape[1]),
        "n_inefficiency_positive": n_pos,
        "n_remission_negative": n_neg,
        "observed_auc": observed_auc,
        "p_value": p_val,
        "n_permutations_requested": int(n_permutations),
        "n_permutations_completed": int(len(null_aucs)),
        "n_null_auc_ge_observed": int(n_null_ge),
        "runtime_sec": runtime_sec,
        "boruta_available": HAS_BORUTA,
        "use_boruta_requested": use_boruta,
    }
    summary_row.update(cm_metrics)
    summary_df = pd.DataFrame([summary_row])
    tsv(summary_df, table_dir / f"{run_id}_Summary.tsv")

    (run_dir / "RUN_COMPLETE.ok").write_text(f"completed\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return RunResult(summary_df, importance_df, run_dir)


def select_top_percentile_features(
    all_feature_importance: pd.DataFrame,
    cutoff: float,
) -> List[str]:
    ranked = all_feature_importance.sort_values("rank_within_run")
    n_features = len(ranked)
    n_keep = max(1, int(math.ceil(n_features * cutoff / 100.0)))
    return ranked.head(n_keep)["feature"].astype(str).tolist()


def write_selected_features_table(
    all_feature_importance: pd.DataFrame,
    selected_features: Sequence[str],
    data: FeatureSetData,
    run_name: str,
    cutoff: float,
    out_dir: Path,
) -> None:
    run_id = f"{data.group}_{data.feature_set}_{run_name}"
    run_dir = out_dir / data.group / data.feature_set / run_name
    selected = set(selected_features)
    df = all_feature_importance.copy()
    df = df.sort_values("rank_within_run")
    # `all_feature_importance` is the importance table from the all-feature run, so
    # it already carries per-run scalar columns (run_id, run_type, percentile_cutoff
    # = None) describing THAT run.  This table instead describes the cutoff run that
    # is about to use these rankings, so drop the carried-over columns before adding
    # fresh ones.  Without this drop, `insert("percentile_cutoff", ...)` raises
    # "cannot insert percentile_cutoff, already exists" and every cutoff run fails.
    df = df.drop(columns=[c for c in ["run_id", "run_type", "percentile_cutoff"] if c in df.columns])
    df.insert(0, "selected_run_id", run_id)
    df.insert(1, "selected_by_percentile_cutoff", df["feature"].astype(str).isin(selected))
    df.insert(2, "percentile_cutoff", cutoff)
    tsv(df, run_dir / "tables" / f"{run_id}_SelectedFeaturesFromAllFeatureRank.tsv")


def collect_global_outputs(out_dir: Path) -> None:
    table_patterns = {
        "all_runs_summary.tsv": "*_Summary.tsv",
        "all_runs_feature_importance.tsv": "*_FeatureImportance.tsv",
        "all_runs_permutation.tsv": "*_Permutation.tsv",
        "all_runs_pvalues.tsv": "*_PValue.tsv",
        "all_runs_predictions.tsv": "*_Predictions.tsv",
        "all_runs_confusion_matrices.tsv": "*_ConfusionMatrix.tsv",
    }
    for out_name, pattern in table_patterns.items():
        paths = sorted(out_dir.glob(f"*/*/*/tables/{pattern}"))
        if not paths:
            continue
        frames = []
        for p in paths:
            try:
                frames.append(read_tsv(p))
            except Exception as exc:
                print(f"[collect warning] could not read {p}: {exc}", file=sys.stderr)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            tsv(combined, out_dir / out_name)


def write_manifest(
    out_dir: Path,
    groups: Sequence[str],
    feature_sets: Sequence[str],
    data_dir: Path,
) -> None:
    rows = []
    inputs = get_input_files(data_dir)
    for group in groups:
        for feature_set in feature_sets:
            files = inputs[group]
            rows.append(
                {
                    "disease_group": group,
                    "feature_set": feature_set,
                    "pathway_file": str(files.pathway),
                    "pathway_file_exists": files.pathway.exists(),
                    "species_file": str(files.species),
                    "species_file_exists": files.species.exists(),
                }
            )
    tsv(pd.DataFrame(rows), out_dir / "input_manifest.tsv")


def run_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = args.groups
    feature_sets = args.feature_sets
    cutoffs = parse_cutoffs(args.cutoffs)
    n_jobs = int(args.n_cores)
    if n_jobs < 1:
        n_jobs = 1

    param_grid = DEFAULT_PARAM_GRID
    if args.fast_grid:
        param_grid = {
            "n_estimators": [200],
            "max_depth": [None, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
        }

    use_boruta = not args.no_boruta
    resume = not args.no_resume

    print(f"[config] data_dir={data_dir}", flush=True)
    print(f"[config] out_dir={out_dir}", flush=True)
    print(f"[config] groups={groups}", flush=True)
    print(f"[config] feature_sets={feature_sets}", flush=True)
    print(f"[config] cutoffs={cutoffs}", flush=True)
    print(f"[config] n_permutations={args.n_permutations}", flush=True)
    print(f"[config] n_cores={n_jobs}", flush=True)
    print(f"[config] use_boruta={use_boruta}; boruta_available={HAS_BORUTA}", flush=True)
    if use_boruta and not HAS_BORUTA:
        print(
            "[warning] Boruta is not importable in this Python environment. "
            "The script will use the RF-importance fallback pre-screening. "
            "Install package 'Boruta' to reproduce the original Boruta step.",
            flush=True,
        )

    write_manifest(out_dir, groups, feature_sets, data_dir)

    if args.dry_run:
        print("[dry-run] Checking input files and feature shapes only.", flush=True)
        for group in groups:
            for feature_set in feature_sets:
                _ = load_feature_set(group, feature_set, data_dir)
        print("[dry-run] All requested inputs loaded successfully.", flush=True)
        return

    failed_runs: List[Dict[str, str]] = []

    for group in groups:
        for feature_set in feature_sets:
            try:
                data = load_feature_set(group, feature_set, data_dir)

                # Run 1: all available features in this feature set.
                all_result = run_one_model(
                    data=data,
                    X_run=data.X,
                    feature_types_run=data.feature_types,
                    run_name="all_features",
                    run_type="all_features",
                    percentile_cutoff=None,
                    global_rank_source=None,
                    out_dir=out_dir,
                    n_permutations=args.n_permutations,
                    n_jobs=n_jobs,
                    threshold=args.threshold,
                    top_n_plot=args.top_n_plot,
                    use_boruta=use_boruta,
                    boruta_perc=args.boruta_perc,
                    boruta_max_iter=args.boruta_max_iter,
                    boruta_estimator_n=args.boruta_estimator_n,
                    fallback_max_features=args.fallback_max_features,
                    param_grid=param_grid,
                    resume=resume,
                    overwrite=args.overwrite,
                )

                all_importance = all_result.feature_importance.sort_values("rank_within_run")

                # Runs 2-N: top percentile cutoffs based on all-feature ranking.
                for cutoff in cutoffs:
                    run_name = cutoff_label(cutoff)
                    selected_features = select_top_percentile_features(all_importance, cutoff)
                    missing = [f for f in selected_features if f not in data.X.columns]
                    if missing:
                        raise RuntimeError(
                            f"Internal error: selected features missing from data matrix: {missing[:5]}"
                        )
                    X_cutoff = data.X.loc[:, selected_features].copy()
                    feature_types_cutoff = {f: data.feature_types.get(f, "unknown") for f in selected_features}
                    write_selected_features_table(
                        all_feature_importance=all_importance,
                        selected_features=selected_features,
                        data=data,
                        run_name=run_name,
                        cutoff=cutoff,
                        out_dir=out_dir,
                    )
                    run_one_model(
                        data=data,
                        X_run=X_cutoff,
                        feature_types_run=feature_types_cutoff,
                        run_name=run_name,
                        run_type="percentile_cutoff",
                        percentile_cutoff=cutoff,
                        global_rank_source=all_importance,
                        out_dir=out_dir,
                        n_permutations=args.n_permutations,
                        n_jobs=n_jobs,
                        threshold=args.threshold,
                        top_n_plot=args.top_n_plot,
                        use_boruta=use_boruta,
                        boruta_perc=args.boruta_perc,
                        boruta_max_iter=args.boruta_max_iter,
                        boruta_estimator_n=args.boruta_estimator_n,
                        fallback_max_features=args.fallback_max_features,
                        param_grid=param_grid,
                        resume=resume,
                        overwrite=args.overwrite,
                    )
            except Exception as exc:
                failed_runs.append(
                    {
                        "disease_group": group,
                        "feature_set": feature_set,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[ERROR] {group} {feature_set}: {exc}", file=sys.stderr, flush=True)
                print(traceback.format_exc(), file=sys.stderr, flush=True)

    collect_global_outputs(out_dir)
    if failed_runs:
        tsv(pd.DataFrame(failed_runs), out_dir / "FAILED_RUNS.tsv")
        raise SystemExit(f"Pipeline finished with {len(failed_runs)} failed group/feature-set blocks. See FAILED_RUNS.tsv")

    print(f"\nAll requested runs completed. Results: {out_dir.resolve()}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random Forest MTX-response pipeline for RA/PsA microbiome feature sets."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing input TSV files.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for results.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["RA", "PsA"],
        choices=["RA", "PsA"],
        help="Disease groups to run.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["pathway_diversity", "pathway_only", "species_diversity"],
        choices=[
            "pathway_diversity",
            "pathway_only",
            "species_diversity",
            "combined_pathway_species_diversity",
        ],
        help="Feature sets to run.",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="*",
        default=[str(v) for v in DEFAULT_CUTOFFS],
        help="Top percentile cutoffs to re-run after all-feature ranking.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=int(os.environ.get("RF_N_PERMUTATIONS", "1000")),
        help="Number of label permutations for each run.",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=int(os.environ.get("N_CORES", os.cpu_count() or 1)),
        help="Parallel workers. In SLURM, set this to $SLURM_CPUS_PER_TASK.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for confusion matrix.")
    parser.add_argument("--top-n-plot", type=int, default=30, help="Number of features in important-score plot.")
    parser.add_argument("--boruta-perc", type=int, default=90, help="Boruta perc parameter.")
    parser.add_argument("--boruta-max-iter", type=int, default=100, help="Boruta max_iter parameter.")
    parser.add_argument("--boruta-estimator-n", type=int, default=200, help="RF trees for Boruta/fallback screening.")
    parser.add_argument(
        "--fallback-max-features",
        type=int,
        default=50,
        help="Maximum features retained by fallback RF screening when Boruta is unavailable/rejects all.",
    )
    parser.add_argument("--no-boruta", action="store_true", help="Disable Boruta and use all input features inside each fold.")
    parser.add_argument("--fast-grid", action="store_true", help="Use a smaller RF tuning grid; useful for testing.")
    parser.add_argument("--dry-run", action="store_true", help="Load inputs and print shapes without running models.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite completed runs.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed runs.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
