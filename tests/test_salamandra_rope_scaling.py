from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SALAMANDRA_MODEL_SCRIPT = REPO_ROOT / "scripts" / "models" / "salamandra-7B.sh"
MODEL_PROVIDER = REPO_ROOT / "slime" / "backends" / "megatron_utils" / "model_provider.py"


def test_salamandra_7b_uses_llama3_rope_scaling_factor_20():
    text = SALAMANDRA_MODEL_SCRIPT.read_text()
    assert "--rope-scaling-factor 20.0" in text
    assert "--rotary-scaling-factor" not in text
    assert "--attention-dropout 0.0" in text
    assert "--hidden-dropout 0.0" in text


def test_model_provider_forwards_rope_scaling_factor_to_gpt_model():
    text = MODEL_PROVIDER.read_text()
    assert '"rope_scaling_factor": args.rope_scaling_factor' in text
    assert '"rope_scaling": args.use_rope_scaling' in text
