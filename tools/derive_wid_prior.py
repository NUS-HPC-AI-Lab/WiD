"""Convert a saved centered PCA artifact using the exact two-stage WiD convention."""
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from compute_wid_statistics import derive_uncentered, digest

p=argparse.ArgumentParser(__doc__)
p.add_argument("--centered",required=True)
p.add_argument("--output",required=True)
a=p.parse_args()
out=Path(a.output)
if out.exists():
    raise FileExistsError(out)
source=torch.load(a.centered,map_location="cpu",weights_only=True)
if source["flatten_order"]!="chw_unfold_v1":
    raise ValueError("Expected CHW patch statistics")
result=derive_uncentered(source)
result["source_stats_sha256"]=digest(a.centered)
out.parent.mkdir(parents=True,exist_ok=True)
torch.save(result,out)
print(digest(out),out)
