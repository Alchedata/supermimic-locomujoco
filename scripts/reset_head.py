"""Shrink the decoder output layer and clear only its Adam moments."""
import argparse
import copy
import hashlib
import json
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
import pickle
import jax
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--scale', type=float, default=.1)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
payload = a.checkpoint.read_bytes()
original = pickle.loads(payload)
modified = copy.deepcopy(original)
head = modified['pp']['params']['MLP_1']['Dense_4']
for name in ['kernel', 'bias']:
    head[name] = head[name] * a.scale
for moments in [modified['op'][2][0].mu, modified['op'][2][0].nu]:
    for name in ['kernel', 'bias']:
        layer = moments['params']['MLP_1']['Dense_4']
        layer[name] = np.zeros_like(layer[name])
changes = []
left, left_structure = jax.tree_util.tree_flatten_with_path(original)
right, right_structure = jax.tree_util.tree_flatten_with_path(modified)
assert left_structure == right_structure
for (path, before), (_, after) in zip(left, right):
    if not np.array_equal(before, after):
        key = jax.tree_util.keystr(path)
        assert key.startswith(("['pp']", "['op']")) and "['MLP_1']['Dense_4']" in key, key
        changes.append(key)
assert len(changes) == 6, changes
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_bytes(pickle.dumps(modified))
a.output.with_suffix('.json').write_text(json.dumps({
    'parent_sha256': hashlib.sha256(payload).hexdigest(),
    'parent_iteration': original['iteration'], 'output_scale': a.scale,
    'changed_leaves': changes, 'adam_shared_count': 'preserved',
    'other_checkpoint_fields': 'unchanged'
}, indent=2))

