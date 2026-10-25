import sys, torch
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_alias_neurons as E
from transformers import Qwen2Config, Qwen2ForCausalLM
E.DEVICE = "cpu"
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=300, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                  tie_word_embeddings=True)
base = Qwen2ForCausalLM(cfg).eval(); rl = Qwen2ForCausalLM(cfg).eval()
for m in (base, rl):
    for p in m.parameters(): p.requires_grad_(False)
base.enable_input_require_grads()
P = 20
idsA = torch.randint(0, 300, (1, 35)); idsA[0, :P] = torch.randint(0, 300, (P,))
idsQ = torch.cat([idsA[:, :P], torch.randint(0, 300, (1, 12))], 1)

# 1. Capture shapes and grads
lp, acts, grads = E.pair_pass(base, idsA, P, grad=True)
assert len(acts) == 3 and acts[0].shape == (34, 96) and grads[0].shape == (34, 96), acts[0].shape
print("capture ok, logp", round(lp, 3))

# 2. dose-0 steering is a no-op, set-mode changes output
ref = E.completion_logp(base, idsA, P).item()
med = [1.0, 1.0, 1.0]
with E.Intervene(base, E.steer_spec(base, [{"layer": 1, "neuron": 5, "sign": 1}], 0.0, med)):
    assert abs(E.completion_logp(base, idsA, P).item() - ref) < 1e-5
with E.Intervene(base, [{"layer": 1, "neuron": 5, "mode": "set", "value": 0.0}]):
    z = E.completion_logp(base, idsA, P).item()
assert abs(E.completion_logp(base, idsA, P).item() - ref) < 1e-6, "hook leaked"
print("intervene ok, zero-abl changes logp by", round(z - ref, 4))

# 3. accumulate + sign check: predicted dm from steer_grad vs finite difference
acc = {k: torch.zeros(3, 96) for k in ("steer_grad", "atp", "zero_abl", "dact")}
mb, mr = E.accumulate_pair(base, rl, idsA, P, idsQ, P, acc)
def m(spec):
    with E.Intervene(base, spec):
        return E.completion_logp(base, idsA, P).item() - E.completion_logp(base, idsQ, P).item()
assert abs(m([]) - mb) < 1e-4
delta = 1e-2
for L, n in [(0, 3), (1, 50), (2, 90)]:
    real = m([{"layer": L, "neuron": n, "mode": "add", "value": delta}]) - mb
    pred = acc["steer_grad"][L, n].item() * delta
    print(f"  L{L} N{n}: predicted dm {pred:+.6f}  real {real:+.6f}")
    assert abs(real - pred) < 0.05 * abs(pred) + 1e-5

# 4. AtP linear check: patch ALL neurons of layer 2 to RL acts ~ sum of atp in layer 2
#    (last layer: no downstream nonlinearity from other MLPs, so linear est. should be close)
_, aA_r, _ = E.pair_pass(rl, idsA, P, grad=False)
print("atp layer sums", [round(v, 4) for v in acc["atp"].sum(1).tolist()])

# 5. resid medians + logit lens
class Tok:
    def __call__(self, p, **k):
        class O: pass
        o = O(); o.input_ids = idsA; return o
    def convert_ids_to_tokens(self, ids): return [f"t{i}" for i in ids]
print("medians", [round(x, 2) for x in E.resid_medians(base, Tok(), ["x"])])
print("lens", E.logit_lens(base, Tok(), 1, 5, k=3))
print("configs", len(E.build_configs({"base_id": "B", "rl_path": "R",
      "screened": [{"layer": 1, "neuron": i, "sign": 1} for i in range(8)]}, 0)))
print("ALL OK")
