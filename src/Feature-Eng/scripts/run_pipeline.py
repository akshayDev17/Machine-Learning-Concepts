"""Run the feature-engineering pipeline described in config.yml.

    python run_pipeline.py --config config.yml                     # every enabled stage
    python run_pipeline.py --disable impute_missing,scale_numeric  # same, with two stages switched off
    python run_pipeline.py --ablation                              # one run per optional stage switched off

How a run works:
1. load the table and split off a stratified test part (locked away until step 4);
2. cut the training part into cross-validation folds;
3. for every fold: fit each enabled stage on the fold's training rows, apply it to the fold's validation rows,
   train the model, score it. Every stage is re-fitted inside every fold, so nothing learned from validation rows
   (medians, category rates, chosen columns, ...) leaks into the score;
4. fit every stage on the whole training part, apply it to the test part, train, score once on test.
"""
import argparse
import copy
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))               # so stage files can `import column_kinds`

WHY_REQUIRED = {
    "load_data": "there would be no table to work on.",
    "split_data": "without a held-out test part and CV folds there is no honest score; every stage must learn "
                  "from training rows only.",
    "train_and_evaluate": "the pipeline exists to produce this model and its score.",
}


def load_stage(name):
    """Import the stage file `NN_<name>.py` from this folder and return it as a module."""
    matches = sorted(SCRIPTS_DIR.glob(f"[0-9][0-9]_{name}.py"))
    if not matches:
        raise SystemExit(f"config names a stage {name!r}, but there is no file NN_{name}.py in {SCRIPTS_DIR}")
    spec = importlib.util.spec_from_file_location(name, matches[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_config(path):
    """Read the YAML config into a plain dict."""
    with open(path) as handle:
        return yaml.safe_load(handle)


def switch_off(config, stage_names):
    """Return a copy of `config` with the named stages disabled."""
    config = copy.deepcopy(config)
    known = {stage["name"] for stage in config["stages"]}
    for name in stage_names:
        if name not in known:
            raise SystemExit(f"--disable: no stage called {name!r}. Stages: {sorted(known)}")
    for stage in config["stages"]:
        if stage["name"] in stage_names:
            stage["enabled"] = False
    return config


def check_required(config):
    """Stop with an explanation if a required stage was switched off."""
    for stage in config["stages"]:
        if stage.get("required") and not stage.get("enabled", True):
            raise SystemExit(f"Stage {stage['name']!r} is required and can't be disabled: "
                             f"{WHY_REQUIRED[stage['name']]} Set it back to `enabled: true`.")


def run_stages(train, valid, stages, target):
    """Fit each stage on `train`, apply it to `valid`. Return (train, valid, list of (name, state))."""
    states = []
    for name, module, params in stages:
        train, state = module.fit_transform(train, target, params)
        valid = module.transform(valid, state)
        states.append((name, state))
    return train, valid, states


def run_pipeline(config, base_dir, verbose=True):
    """Run the whole pipeline. Return a summary dict with the stage list, CV scores and test scores."""
    check_required(config)
    by_name = {stage["name"]: stage for stage in config["stages"]}
    seed = config["random_state"]
    load, split, evaluate = (load_stage(n) for n in ("load_data", "split_data", "train_and_evaluate"))
    data_params, split_params = by_name["load_data"]["params"], by_name["split_data"]["params"]
    model_params = {**by_name["train_and_evaluate"]["params"], "random_state": seed}
    target = data_params["target"]

    table = load.load_table(base_dir / data_params["path"])
    train, test = split.split_train_test(table, target, split_params["test_size"], seed, split_params["stratify"])
    folds = split.make_cv_folds(train, target, split_params["cv_folds"], seed)
    stages = [(s["name"], load_stage(s["name"]), {**s.get("params", {}), "random_state": seed})
              for s in config["stages"] if s["enabled"] and s["name"] not in WHY_REQUIRED]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fold_scores = []
        for fit_rows, valid_rows in folds:
            fold_train, fold_valid, _ = run_stages(train.iloc[fit_rows], train.iloc[valid_rows], stages, target)
            fold_scores.append(evaluate.fit_and_score(fold_train, fold_valid, target, model_params)[0])
        full_train, full_test, states = run_stages(train, test, stages, target)
        test_scores, notes, n_features = evaluate.fit_and_score(full_train, full_test, target, model_params)
    warning_kinds = sorted({w.category.__name__ for w in caught})

    cv = {metric: (float(np.mean([s[metric] for s in fold_scores])), float(np.std([s[metric] for s in fold_scores])))
          for metric in ("roc_auc", "accuracy")}
    summary = {
        "stages": [{"name": s["name"], "enabled": s["enabled"], "required": bool(s.get("required"))}
                   for s in config["stages"]],
        "stage_reports": {name: state.get("summary", "") for name, state in states},
        "model": model_params["model"], "n_features": n_features, "notes": notes, "warnings": warning_kinds,
        "rows": {"train": len(train), "test": len(test)},
        "cv": {m: {"mean": mean, "std": std} for m, (mean, std) in cv.items()},
        "test": {m: float(v) for m, v in test_scores.items()},
    }
    if verbose:
        print_summary(summary, len(table), table.shape[1], split_params["cv_folds"])
    return summary


def print_summary(summary, n_rows, n_columns, n_folds):
    """Print which stages ran, what each did, and the scores."""
    print("Stages (top to bottom):")
    for stage in summary["stages"]:
        name = stage["name"]
        if stage["required"]:
            report = {"load_data": f"{n_rows} rows x {n_columns} columns",
                      "split_data": f"train {summary['rows']['train']} / test {summary['rows']['test']}, "
                                    f"{n_folds} CV folds on train",
                      "train_and_evaluate": f"{summary['model']} on {summary['n_features']} features"}[name]
            print(f"  [on ] {name:<24} (required) {report}")
        elif stage["enabled"]:
            print(f"  [on ] {name:<24} {summary['stage_reports'][name]}")
        else:
            print(f"  [off] {name:<24} disabled in config")
    for note in summary["notes"]:
        print(f"  note: {note}")
    if summary["warnings"]:
        print(f"  warnings raised while fitting: {summary['warnings']}")
    cv, test = summary["cv"], summary["test"]
    print(f"\nCV on train ({n_folds} folds): ROC-AUC {cv['roc_auc']['mean']:.3f} ± {cv['roc_auc']['std']:.3f}   "
          f"accuracy {cv['accuracy']['mean']:.3f} ± {cv['accuracy']['std']:.3f}")
    print(f"Test (used once)    : ROC-AUC {test['roc_auc']:.3f}           accuracy {test['accuracy']:.3f}")


def run_ablation(config, base_dir):
    """Run once with everything on, then once per optional enabled stage switched off. Return a list of rows."""
    rows = [("(all enabled stages)", run_pipeline(config, base_dir, verbose=False))]
    optional = [s["name"] for s in config["stages"] if s["enabled"] and not s.get("required")]
    for name in optional:
        rows.append((f"without {name}", run_pipeline(switch_off(config, [name]), base_dir, verbose=False)))
    return rows


def print_ablation(rows):
    """Print one line per ablation run: CV and test ROC-AUC, and the change against the full run."""
    full_cv = rows[0][1]["cv"]["roc_auc"]["mean"]
    print(f"{'run':<34} {'CV ROC-AUC':>15} {'change':>8} {'test ROC-AUC':>13}")
    for label, summary in rows:
        cv = summary["cv"]["roc_auc"]
        print(f"{label:<34} {cv['mean']:>8.3f} ± {cv['std']:.3f} {cv['mean'] - full_cv:>+8.3f} "
              f"{summary['test']['roc_auc']:>13.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(SCRIPTS_DIR / "config.yml"), help="path to the YAML config")
    parser.add_argument("--disable", default="", help="comma-separated stage names to switch off for this run")
    parser.add_argument("--ablation", action="store_true", help="also run once per optional stage switched off")
    parser.add_argument("--summary-out", help="write the run summary as JSON to this path")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    if args.disable:
        config = switch_off(config, [name.strip() for name in args.disable.split(",")])
    if args.ablation:
        rows = run_ablation(config, config_path.parent)
        print_ablation(rows)
        result = {label: summary for label, summary in rows}
    else:
        result = run_pipeline(config, config_path.parent)
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(result, indent=2))
        print(f"\nsummary written to {args.summary_out}")


if __name__ == "__main__":
    main()
