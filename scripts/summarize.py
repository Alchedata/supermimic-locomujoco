"""Print metrics from the trainer's evaluation JSON files."""
import argparse
import csv
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('files', nargs='+', type=Path)
a = p.parse_args()
writer = csv.writer(sys.stdout)
writer.writerow(['Evaluation', 'Return', 'Error (m)', 'Failures', 'Numerical failures', 'Full horizon', 'Episodes'])
for path in a.files:
    r = json.loads(path.read_text())
    writer.writerow([path.as_posix(), r['mean_return'], r['relative_site_position_error_m'],
                     r['absorbing_failures'], r['numerical_failures'],
                     r['completed_horizon_without_failure'], r['episodes']])
