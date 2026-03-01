import sys
from pathlib import Path

# Add parent directory to path (works regardless of where script is run from)
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from data_generator import raw_data
dataset = raw_data
print(dataset.head())