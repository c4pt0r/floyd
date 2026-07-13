import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from safetensors import safe_open

from tools.make_moonlight_oracle import _load_model, capture_checkpoint


PROMPT_IDS = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]


class MoonlightOracleTest(unittest.TestCase):
    @patch("tools.make_moonlight_oracle.DeepseekV3ForCausalLM.from_pretrained")
    @patch("tools.make_moonlight_oracle.AutoModelForCausalLM.from_pretrained")
    def test_native_loader_handles_incompatible_remote_code(self, auto_load, native_load):
        auto_load.side_effect = ImportError("removed transformers helper")
        native_load.return_value.eval.return_value = "native-model"

        self.assertEqual(_load_model(Path("checkpoint")), "native-model")
        native_load.assert_called_once()

    def test_capture_contains_runtime_stage_contract(self):
        model_path = Path(os.environ["MOONLIGHT_TINY"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            capture_checkpoint(model_path, output, PROMPT_IDS, max_tokens=2)

            reference = json.loads((output / "ref.json").read_text())
            self.assertEqual(reference["prompt_ids"], PROMPT_IDS)
            self.assertEqual(len(reference["greedy_ids"]), 2)
            self.assertEqual(
                reference["full_ids"], PROMPT_IDS + reference["greedy_ids"]
            )
            self.assertEqual(len(reference["turns"]), 2)
            self.assertEqual(reference["turns"][0]["input_ids"], PROMPT_IDS)
            self.assertEqual(
                reference["turns"][1]["input_ids"][: len(reference["full_ids"])],
                reference["full_ids"],
            )
            self.assertEqual(len(reference["turns"][1]["output_ids"]), 2)
            cpu_reference = json.loads((output / "cpu_ref.json").read_text())
            self.assertEqual(cpu_reference["prompt_ids"], PROMPT_IDS)
            self.assertEqual(cpu_reference["full_ids"], PROMPT_IDS)
            self.assertEqual(len(cpu_reference["tf_pred"]), len(PROMPT_IDS))

            with safe_open(output / "oracle.safetensors", framework="pt") as oracle:
                keys = set(oracle.keys())
                required = {
                    "input_ids",
                    "embed",
                    "layer.0.input",
                    "layer.0.input_norm",
                    "layer.0.q",
                    "layer.0.kv_a",
                    "layer.0.kv_norm",
                    "layer.0.kv_b",
                    "layer.0.attn",
                    "layer.0.post_attn",
                    "layer.0.post_norm",
                    "layer.0.mlp",
                    "layer.0.output",
                    "layer.1.router_scores",
                    "layer.1.router_weights",
                    "layer.1.router_ids",
                    "layer.1.shared_mlp",
                    "layer.1.routed_mlp",
                    "final_input",
                    "final_norm",
                    "logits",
                }
                self.assertTrue(required <= keys, sorted(required - keys))
                self.assertEqual(
                    oracle.get_slice("input_ids").get_shape(), [len(PROMPT_IDS)]
                )
                self.assertEqual(
                    oracle.get_slice("embed").get_shape(), [len(PROMPT_IDS), 256]
                )
                self.assertEqual(
                    oracle.get_slice("layer.1.router_ids").get_shape(),
                    [len(PROMPT_IDS), 6],
                )
                self.assertEqual(
                    oracle.get_slice("logits").get_shape(), [len(PROMPT_IDS), 512]
                )


if __name__ == "__main__":
    unittest.main()
