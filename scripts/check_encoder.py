"""Check native layer extraction and the indexed position lookup on real media."""

import argparse
import json
from pathlib import Path

from mlx_vlm.models.gemma4.vision import VisionPatchEmbedder

from gemma_rlcd import State
from gemma_rlcd.core import ScoringRequest
from gemma_rlcd.head_backend import DecisionHeadBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--state-layers", type=int, default=35)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    backend = DecisionHeadBackend(
        args.model, state_layers=args.state_layers, compute_dtype=args.dtype
    )
    mx = backend.mx
    media = args.media.resolve()
    records = []
    for name, state in [
        ("text", State(text="A cat sits on a blue chair.")),
        ("image", State(images=(str(media / "red.png"),))),
        ("speech", State(audio=(str(media / "dog.wav"),))),
        ("video_speech", State(videos=(str(media / "red-blue-speech.mp4"),))),
    ]:
        prepared = backend.prepare(state, [ScoringRequest("Read the state.", ("A", "B"))])
        actual = backend._state_features(prepared.inputs)
        features = backend.model.get_input_embeddings(**prepared.inputs)
        captured = []
        final = backend.model.language_model.model(
            inputs=prepared.inputs["input_ids"],
            inputs_embeds=features.inputs_embeds,
            per_layer_inputs=features.per_layer_inputs,
            mm_token_type_ids=prepared.inputs.get("mm_token_type_ids"),
            hidden_sink=captured,
            capture_layer_ids=[backend.state_layers - 1],
            logits_to_keep=1,
        )
        mx.eval(actual, captured, final)
        record = {
            "state": name,
            "native_layer_max_abs_delta": float(mx.max(mx.abs(actual - captured[0])).item()),
        }
        indexed = backend.model.vision_tower.patch_embedder
        dense = VisionPatchEmbedder(backend.model.config.vision_config)
        dense.input_proj = indexed.input_proj
        dense.position_embedding_table = indexed.position_embedding_table
        backend.model.vision_tower.patch_embedder = dense
        reference = backend.model.get_input_embeddings(**prepared.inputs).inputs_embeds
        mx.eval(reference)
        backend.model.vision_tower.patch_embedder = indexed
        record["dense_vs_indexed_embedding_max_abs_delta"] = float(
            mx.max(mx.abs(reference - features.inputs_embeds)).item()
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    args.report.write_text(
        json.dumps(
            {"compute_dtype": args.dtype, "state_layers": args.state_layers, "results": records},
            indent=2,
        )
        + "\n"
    )
    if any(
        row["native_layer_max_abs_delta"] != 0
        or (args.dtype != "float32" and row["dense_vs_indexed_embedding_max_abs_delta"] != 0)
        for row in records
    ):
        raise SystemExit("Encoder equivalence check failed; inspect recorded differences")


if __name__ == "__main__":
    main()
