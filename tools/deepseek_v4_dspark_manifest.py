#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from safetensors import safe_open


_STAGE_RE = re.compile(r"^mtp\.(\d+)\.")
_REQUIRED = (
    "mtp.0.main_proj.weight",
    "mtp.0.main_norm.weight",
    "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.confidence_head.proj.weight",
)


@dataclass(frozen=True)
class CheckpointManifest:
    stage_ids: tuple[int, ...]
    stage_tensor_counts: tuple[int, ...]
    required_shapes: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class SupportGGUFManifest:
    stage_ids: tuple[int, ...]
    missing_stage_ids: tuple[int, ...]
    complete: bool


def _stage_ids(names):
    return tuple(sorted({int(match.group(1)) for name in names
                         if (match := _STAGE_RE.match(name))}))


def inspect_checkpoint(model_dir: Path) -> CheckpointManifest:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as source:
        weight_map = json.load(source)["weight_map"]

    stage_ids = _stage_ids(weight_map)
    counts = tuple(sum(name.startswith(f"mtp.{stage}.") for name in weight_map)
                   for stage in stage_ids)
    missing = [name for name in _REQUIRED if name not in weight_map]
    if missing:
        raise ValueError(f"checkpoint is missing DSpark tensors: {', '.join(missing)}")

    required_shapes = {}
    by_shard = {}
    for name in _REQUIRED:
        by_shard.setdefault(weight_map[name], []).append(name)
    for shard, names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as tensors:
            for name in names:
                required_shapes[name] = tuple(tensors.get_slice(name).get_shape())

    return CheckpointManifest(stage_ids, counts, required_shapes)


def inspect_support_gguf(path: Path) -> SupportGGUFManifest:
    from gguf import GGUFReader

    reader = GGUFReader(Path(path), "r")
    stage_ids = _stage_ids(tensor.name for tensor in reader.tensors)
    missing = tuple(stage for stage in range(3) if stage not in stage_ids)
    required_names = {tensor.name for tensor in reader.tensors}
    complete = not missing and all(name in required_names for name in _REQUIRED)
    return SupportGGUFManifest(stage_ids, missing, complete)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DeepSeek V4 DSpark weights")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--support-gguf", type=Path)
    args = parser.parse_args()

    result = {"checkpoint": asdict(inspect_checkpoint(args.checkpoint))}
    if args.support_gguf:
        result["support_gguf"] = asdict(inspect_support_gguf(args.support_gguf))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
