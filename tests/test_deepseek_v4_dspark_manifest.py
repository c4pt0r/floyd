import os
from pathlib import Path

from tools.deepseek_v4_dspark_manifest import (
    build_template_specs,
    inspect_checkpoint,
    inspect_support_gguf,
)


def test_official_checkpoint_has_complete_three_stage_dspark():
    manifest = inspect_checkpoint(Path(os.environ["DSPARK"]))

    assert manifest.stage_ids == (0, 1, 2)
    assert all(count > 1500 for count in manifest.stage_tensor_counts)
    assert manifest.required_shapes == {
        "mtp.0.main_proj.weight": (4096, 12288),
        "mtp.0.main_norm.weight": (4096,),
        "mtp.2.markov_head.markov_w1.weight": (129280, 256),
        "mtp.2.markov_head.markov_w2.weight": (129280, 256),
        "mtp.2.confidence_head.proj.weight": (1, 4352),
    }


def test_current_support_gguf_is_rejected_as_single_stage():
    manifest = inspect_support_gguf(Path(os.environ["DSPARK_MTP"]))

    assert manifest.stage_ids == (0,)
    assert manifest.missing_stage_ids == (1, 2)
    assert not manifest.complete


def test_three_stage_template_reuses_ds4_quant_layout():
    specs = build_template_specs(Path(os.environ["DSPARK_MTP"]))
    by_name = {spec.name: spec for spec in specs}

    assert len(specs) == 81
    assert tuple(sum(spec.name.startswith(f"mtp.{stage}.") for spec in specs)
                 for stage in range(3)) == (26, 24, 31)
    assert by_name["mtp.0.main_proj.weight"].shape == (4096, 12288)
    assert by_name["mtp.0.main_proj.weight"].quant_type == "Q8_0"
    assert by_name["mtp.2.markov_head.markov_w1.weight"].shape == (129280, 256)
    assert by_name["mtp.2.markov_head.markov_w2.weight"].quant_type == "Q8_0"
    assert by_name["mtp.2.confidence_head.proj.weight"].shape == (1, 4352)
    assert "mtp.0.e_proj.weight" not in by_name
    assert "mtp.1.hc_head_fn.weight" not in by_name
    assert "mtp.2.hc_head_fn.weight" in by_name


if __name__ == "__main__":
    test_official_checkpoint_has_complete_three_stage_dspark()
    test_current_support_gguf_is_rejected_as_single_stage()
    test_three_stage_template_reuses_ds4_quant_layout()
    print("DeepSeek V4 DSpark manifest: ok")
