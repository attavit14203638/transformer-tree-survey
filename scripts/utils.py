import pandas as pd
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define paths - relative to repository root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# Data directory: where cleaned CSV files are stored
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Output directory: where generated figures will be saved
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Runtime cache configuration (keeps repo root clean)
# -----------------------------------------------------------------------------
# Matplotlib/fontconfig try to write cache files; we redirect them to a hidden
# cache directory at the project root to avoid littering the repo.
RUNTIME_CACHE_DIR = os.path.join(REPO_ROOT, ".cache")
os.makedirs(RUNTIME_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RUNTIME_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RUNTIME_CACHE_DIR, "fontconfig"))

def load_data():
    """Loads all cleaned CSV files and returns them as pandas DataFrames."""
    try:
        geography = pd.read_csv(os.path.join(DATA_DIR, "geography_clean.csv"))
        models = pd.read_csv(os.path.join(DATA_DIR, "models_clean.csv"))
        # Canonical performance file: percentage metrics are normalized to 0-100 scale.
        performance = pd.read_csv(os.path.join(DATA_DIR, "performance_clean.csv"))
        papers = pd.read_csv(os.path.join(DATA_DIR, "papers_clean.csv"))
        training = pd.read_csv(os.path.join(DATA_DIR, "training_clean.csv"))
        datasets = pd.read_csv(os.path.join(DATA_DIR, "datasets_clean.csv"))
        
        # Merge models info into training for size analysis if needed
        if 'model_id' in training.columns and 'model_id' in models.columns:
             training = pd.merge(training, models[['model_id', 'paper_id', 'parameter_count_millions', 'category_clean']], 
                                on=['paper_id', 'model_id'], how='left', suffixes=('', '_model'))
        
        print("✅ Data loaded successfully.")
        return geography, models, performance, papers, training, datasets
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print(f"Checked path: {DATA_DIR}")
        return None, None, None, None, None, None

def save_fig(fig, name):
    """Saves the figure to the output directory."""
    # Ensure extension is .jpg
    base_name = os.path.splitext(name)[0]
    jpg_name = f"{base_name}.jpg"
    path = os.path.join(OUTPUT_DIR, jpg_name)
    fig.savefig(path, bbox_inches='tight', dpi=300, format='jpeg')
    print(f"✅ Saved {jpg_name} to {path}")
