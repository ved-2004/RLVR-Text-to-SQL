import sys, json, os, tempfile, torch
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_neuron_diff as E
from safetensors.torch import save_file, load_file
from transformers import Qwen2Config, Qwen2ForCausalLM
E.DEVICE = "cpu"
E.N_LAYERS, E.D_FF, E.N_HEADS, E.HEAD_DIM = 3, 96, 4, 16
E.TOP_K, E.LAYER_GROUPS = (4, 16, 32), [(0, 0), (1, 2)]
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=300, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                  tie_word_embeddings=True)
model = Qwen2ForCausalLM(cfg).eval()
base32 = {k: v.clone() for k, v in model.state_dict().items()}
base16 = {k: v.to(torch.bfloat16) for k, v in base32.items()}

# 1. load_lora reads a PEFT adapter: keys, shapes, scale = alpha / r
r, alpha = 4, 8
adapter = {}
for key, W in base32.items():
    if key.endswith("_proj.weight"):
        stem = "base_model.model." + key[:-len(".weight")]
        adapter[stem + ".lora_A.weight"] = torch.randn(r, W.shape[1]) * 0.1
        adapter[stem + ".lora_B.weight"] = torch.randn(W.shape[0], r) * 0.1
tmp = tempfile.mkdtemp()
save_file(adapter, f"{tmp}/adapter_model.safetensors")
json.dump({"r": r, "lora_alpha": alpha}, open(f"{tmp}/adapter_config.json", "w"))
lora = E.load_lora(tmp)
assert len(lora) == 3 * 7 and all(s == 2.0 for _, _, s in lora.values())
assert set(lora) <= set(base32), "LoRA keys must match base checkpoint keys"
print("load_lora ok:", len(lora), "modules")

# 2. closed-form row / column norms match the explicit matrix
B, A, s = lora["model.layers.1.mlp.down_proj.weight"]
D = s * B @ A
assert torch.allclose(E.row_norms(B, A, s), D.norm(dim=1), atol=1e-5)
assert torch.allclose(E.col_norms(B, A, s), D.norm(dim=0), atol=1e-5)
print("row/col norms ok")

# 3. base config is base; full config is base + scale * B @ A everywhere
every = list(range(3))
c = lambda **k: {"attn": [], "mlp": [], "neurons": [], "neuron_mode": "only", **k}
assert all(torch.equal(E.apply_config(base16, lora, c())[k], base16[k]) for k in base16)
full = E.apply_config(base16, lora, c(attn=every, mlp=every))
for key, (B, A, s) in lora.items():
    want = (base16[key].float() + s * B @ A).to(torch.bfloat16)
    assert torch.equal(full[key], want), key
print("base / full ok")

# 4. copying neurons touches only their gate/up rows and down columns
nset = [[1, 5], [1, 40], [2, 7]]
part = E.apply_config(base32, lora, c(neurons=nset))
for key in lora:
    L, mod = E.module_of(key)
    changed = (part[key] != base32[key])
    if mod in E.ATTN or L == 0:
        assert not changed.any(), key
    else:
        idx = [n for l, n in nset if l == L]
        hit = changed.any(dim=0) if mod == "down_proj" else changed.any(dim=1)
        assert hit.nonzero().squeeze(1).tolist() == idx, (key, hit.nonzero().tolist())
print("neuron copy ok")

# 5. "only" + "except" on the same neurons add up to full (fp32, exact algebra)
only = E.apply_config(base32, lora, c(neurons=nset))
exc = E.apply_config(base32, lora, c(neurons=nset, neuron_mode="except"))
full32 = E.apply_config(base32, lora, c(mlp=every))
for key in lora:
    if E.module_of(key)[1] in E.MLP:
        assert torch.allclose(only[key] + exc[key] - base32[key], full32[key], atol=1e-5), key
print("only + except = full ok")

# 6. configs: names, counts, random control matches per-layer counts
w = E.weight_scores(base32, lora)
assert w["rel"].shape == (3, 96) and len(w["heads"][0]) == 4
B, A, s = lora["model.layers.2.mlp.gate_proj.weight"]
W = base32["model.layers.2.mlp.gate_proj.weight"]
manual = (s * B @ A)[11].norm() / W[11].norm()
for m in ("up_proj", "down_proj"):
    B2, A2, s2 = lora[f"model.layers.2.mlp.{m}.weight"]
    W2 = base32[f"model.layers.2.mlp.{m}.weight"]
    d2 = s2 * B2 @ A2
    manual += (d2[11].norm() / W2[11].norm()) if m == "up_proj" else (d2[:, 11].norm() / W2[:, 11].norm())
assert abs(w["rel"][2, 11].item() - manual.item()) < 1e-4
cfgs = E.build_configs(w["rel"].numpy())
names = [x["name"] for x in cfgs]
print("configs", names)
by = {x["name"]: x for x in cfgs}
t, rd = by["top16"]["neurons"], by["rand16"]["neurons"]
count = lambda xs: [sum(1 for l, _ in xs if l == L) for L in range(3)]
assert len(t) == 16 and count(t) == count(rd) and len({tuple(x) for x in rd}) == 16
assert by["grpo_minus_top16"]["neuron_mode"] == "except" and by["grpo_minus_top16"]["attn"] == every
top_flat = w["rel"].flatten().topk(16).indices.tolist()
assert sorted(L * 96 + n for L, n in t) == sorted(top_flat)
print("weight scores / configs ok")

# 7. activation score is zero for identical models, positive otherwise
class Tok:
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [ord(ch) % 300 for ch in text]
        class O: pass
        o = O(); o.input_ids = torch.tensor([ids]) if return_tensors else ids
        return o
texts = [("prompt one ", "SELECT a FROM t"), ("prompt two ", "SELECT b FROM u")]
same = E.activation_scores(Tok(), model, model, texts, device="cpu")
assert same["abs"].abs().max() < 1e-6 and same["tokens"] == 30
other = Qwen2ForCausalLM(cfg).eval()
other.load_state_dict(full32 | {k: v for k, v in base32.items() if k not in full32})
diff = E.activation_scores(Tok(), model, other, texts, device="cpu")
assert diff["abs"].shape == (3, 96) and diff["abs"].min() >= 0 and diff["abs"].max() > 0
print("activation scores ok, max", round(diff["abs"].max().item(), 3))

# 8. write_model round trip: weights load back unchanged, side files copied
like = tempfile.mkdtemp()
json.dump({"x": 1}, open(f"{like}/config.json", "w"))
open(f"{like}/tokenizer.json", "w").write("{}")
save_file({"dummy": torch.zeros(1)}, f"{like}/model.safetensors")
out = tempfile.mkdtemp()
E.write_model(full, out, like_dir=like)
back = load_file(f"{out}/model.safetensors")
assert set(back) == set(full) and all(torch.equal(back[k], full[k]) for k in full)
assert sorted(os.listdir(out)) == ["config.json", "model.safetensors", "tokenizer.json"]
print("write_model ok")

# 9. ranking helpers
import numpy as np
x = np.array([[0.1, 5.0], [3.0, -1.0]])
assert E.ranks_desc(x).tolist() == [[3, 1], [2, 4]]
assert abs(E.spearman(x, x) - 1) < 1e-12
print("ALL OK")
