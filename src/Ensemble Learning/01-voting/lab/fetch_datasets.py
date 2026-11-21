"""Download the real datasets that voting_lab.ipynb uses from OpenML, once, into openml_data/ (git-ignored).

Run it from this folder, after cloning the repository and installing requirements.txt:

    python fetch_datasets.py            # download what is missing
    python fetch_datasets.py --force    # download everything again

Each dataset is saved as one plain CSV file. The notebook only reads those files and never goes online.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

DATA_HOME = Path(__file__).parent / "openml_data"

DATASETS = {  # name -> (OpenML data id, what it is)
    "titanic": (40945, "Titanic passengers"),
    "house_prices": (42165, "Ames houses: the training data of Kaggle's House Prices competition"),
}


def download(name, data_id, force, attempts=3):
    """Fetch one dataset from OpenML and save it as openml_data/<name>.csv; returns False if it could not be fetched."""
    path = DATA_HOME / f"{name}.csv"
    if path.exists() and not force:
        print(f"{name:13s} already there ({path.name}); --force downloads it again")
        return True
    from sklearn.datasets import fetch_openml   # imported late so that --help works without scikit-learn

    for attempt in range(1, attempts + 1):
        try:
            with tempfile.TemporaryDirectory() as scratch:   # scikit-learn's own cache is discarded; we keep one CSV
                frame = fetch_openml(data_id=data_id, as_frame=True, parser="liac-arff", data_home=scratch).frame
            break
        except Exception as error:   # network hiccups, OpenML being briefly down
            print(f"{name:13s} attempt {attempt}/{attempts} failed: {error}")
            if attempt == attempts:
                return False
            time.sleep(3 * attempt)
    DATA_HOME.mkdir(exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"{name:13s} saved {len(frame):,} rows x {frame.shape[1]} columns to {path.relative_to(Path(__file__).parent)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if the CSV already exists")
    args = parser.parse_args()
    results = [download(name, data_id, args.force) for name, (data_id, _) in DATASETS.items()]
    if not all(results):
        sys.exit("Some downloads failed. Check your internet connection and run the script again.")
    print("Done. You can run voting_lab.ipynb now.")
