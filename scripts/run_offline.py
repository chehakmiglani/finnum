"""Execute a Colab notebook OFFLINE (no GPU, no torch, no internet) to check every non-GPU cell.

Model training/inference is replaced by a fake that answers from the gold labels with noise,
so data loading, benchmark building, saving, analysis, charts and summaries all really run.
Usage: python scripts/run_offline.py notebooks/0X_....ipynb [--keep]
"""
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

STUBS = '''
import sys, types, random, json, os
colab = types.ModuleType("google.colab")
colab.files = types.SimpleNamespace(upload=lambda: {}, download=lambda p: print("download", p))
colab.drive = types.SimpleNamespace(mount=lambda p: None)
google = types.ModuleType("google"); google.colab = colab
sys.modules["google"] = google; sys.modules["google.colab"] = colab
torch = types.ModuleType("torch"); torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch
'''

FAKES = '''
import random, json, os, glob
from pathlib import Path
from finnum.benchmark import read_jsonl

def _fake_answers(items_by_prompt, rng, err):
    def gen(prompts):
        out = []
        for p in prompts:
            it = items_by_prompt[p]
            out.append("I am unsure" if rng.random() < err else str(it["gold"]))
        return out
    return gen

# week 2: fake HF model
import finnum.eval.runner as R
class FakeGen:
    def __init__(self, model_id, batch_size=16):
        self.rng = random.Random(len(model_id))
        self.items = {}
        for f in glob.glob("/content/finnum_results/finnum_*.jsonl"):
            for it in read_jsonl(f):
                self.items.setdefault(it["prompt"], it)
    def count_tokens(self, p): return len(p) // 4
    def __call__(self, prompts): return _fake_answers(self.items, self.rng, 0.3)(prompts)
    def close(self): pass
R.HFGenerator = FakeGen

# week 3: fake training module (the real one needs torch)
import types
T = types.ModuleType("finnum.tokexp.train")
class _Cfg:
    def __init__(self, **kw): self.__dict__.update(kw)
T.TrainConfig = _Cfg
sys.modules["finnum.tokexp.train"] = T
def _fake_run_all(data, schemes, seeds, cfg, out_dir):
    sys.path.insert(0, "''' + str(ROOT / "tests") + '''")
    from test_report import fake_result
    os.makedirs(out_dir, exist_ok=True); rng = random.Random(0); res = []
    for seed in seeds:
        for s in schemes:
            r = fake_result(s, seed, rng)
            r["test_ood"]["errors"] = [{"task": "add", "prompt": "add 1 2=", "gold": "3", "pred": "4"}]
            Path(out_dir, f"{s}_seed{seed}.json").write_text(json.dumps(r)); res.append(r)
    return res
T.run_all = _fake_run_all

# weeks 4-6: fake retrofit module (the real one needs torch + peft)
L = types.ModuleType("finnum.retrofit.lm")
L.RetrofitConfig = _Cfg
sys.modules["finnum.retrofit.lm"] = L
from finnum.retrofit.preprocess import mark
def _fake_run_method(method, train_items, test_items, cfg, out_dir):
    from finnum.eval.runner import run_eval
    os.makedirs(out_dir, exist_ok=True)
    by_prompt = {}
    for it in test_items: by_prompt.setdefault(it["prompt"], it)
    err = {"text": 0.35, "xval": 0.6, "fone": 0.25, "bittokens": 0.3, "value_prefix": 0.2, "fone_units": 0.15}[method]
    run_eval(test_items, _fake_answers(by_prompt, random.Random(method), err), method, Path(out_dir, f"{method}.jsonl"),
             count_tokens=lambda p: len(mark(p, method)[0]) // 4, progress=False)
    log = {"method": method, "train_minutes": 0.0, "trainable_params": 1, "losses": [{"step": 1, "loss": 1.0}]}
    Path(out_dir, f"{method}_train.json").write_text(json.dumps(log)); return log
L.run_method = _fake_run_method
print("OFFLINE FAKES ACTIVE")
'''


def run(path: Path) -> int:
    nb = nbformat.read(path, as_version=4)
    s = nb.cells[1].source
    s = s.replace("SAVE_TO_DRIVE = True", "SAVE_TO_DRIVE = False").replace('DATA_SOURCE = "auto"', 'DATA_SOURCE = "sample"')
    s = s.replace("N_PER_TASK = 40", "N_PER_TASK = 6")
    nb.cells[1].source = s
    nb.cells.insert(1, nbformat.v4.new_code_cell(STUBS))
    setup_idx = next(i for i, c in enumerate(nb.cells) if "FINNUM_ZIP" in c.source)
    nb.cells.insert(setup_idx + 1, nbformat.v4.new_code_cell(FAKES))
    NotebookClient(nb, timeout=900, kernel_name="python3").execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    out = Path("/tmp/claude-0/executed_" + path.name)
    nbformat.write(nb, out)
    tail = [o.text for c in nb.cells[-2:] for o in c.get("outputs", []) if o.output_type == "stream"]
    print(f"{path.name}: errors={len(errors)}  executed copy -> {out}")
    print("".join(tail)[-2500:])
    return len(errors)


if __name__ == "__main__":
    sys.exit(sum(run(Path(p)) for p in sys.argv[1:]))
