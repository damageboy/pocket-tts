"""Generate model-backed v3.3 parity data; missing assets are fatal, never skipped."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

from generate_upstream_reference import UPSTREAM_COMMIT, sha256, verify_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", default="english")
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--text", default="The price is 3.14 euros, not twelve.")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--sampler-decode-steps", type=int, default=1)
    args = parser.parse_args()
    root = args.reference.resolve()
    verify_reference(root)
    sys.path.insert(0, str(root))
    import torch
    from safetensors.torch import save_file
    from pocket_tts import TTSModel
    from pocket_tts.modules.stateful_module import init_states, increment_steps
    from pocket_tts.models.text_chunking import prepare_text_prompt
    from pocket_tts.utils.utils import download_if_necessary, get_predefined_voice

    torch.set_num_threads(1)
    model = TTSModel.load_model(language=args.language, sampler_decode_steps=args.sampler_decode_steps)
    model.eval()
    config = model.config
    weights_uri = config.weights_path if model.has_voice_cloning else config.weights_path_without_voice_cloning
    weights = download_if_necessary(weights_uri)
    tokenizer = download_if_necessary(config.flow_lm.lookup_table.tokenizer_path)
    voice_uri = get_predefined_voice(args.language, args.voice)
    voice = download_if_necessary(voice_uri)
    state = model.get_state_for_audio_prompt(args.voice)
    options = {name: getattr(config, name) for name in (
        "pad_with_spaces_for_short_inputs", "remove_semicolons", "append_terminal_punctuation",
        "capitalize_first_letter", "replace_characters",
    )}
    prepared, _ = prepare_text_prompt(args.text, **options)
    tokens = model.flow_lm.conditioner.prepare(prepared)
    tensors = {"tokens": tokens, "text_embeddings": model.flow_lm.conditioner(tokens)}
    model._expand_kv_cache(state, model._flow_lm_current_end(state) + tokens.shape[1] + args.steps)
    eos_logits = []
    hook = model.flow_lm.out_eos.register_forward_hook(
        lambda module, inputs, output: eos_logits.append(output.detach().clone())
    )
    with torch.no_grad():
        model._run_flow_lm_and_increment_step(model_state=state, text_tokens=tokens)
        prefill_length = model._flow_lm_current_end(state)
        for module_name, values in state.items():
            tensors[f"prefill.{module_name}.cache"] = values["cache"][:, :, :prefill_length].clone()
        sequence = torch.full((1, 1, model.flow_lm.ldim), float("nan"))
        mimi_state = init_states(model.mimi, batch_size=1, sequence_length=args.steps * 16)
        for step in range(args.steps):
            noise = (torch.sin(torch.arange(model.flow_lm.ldim) * 0.13 + step) * 0.3**0.5)[None]
            with patch("torch.nn.init.normal_", side_effect=lambda tensor, **kwargs: tensor.copy_(noise)):
                latent, eos = model._run_flow_lm_and_increment_step(
                    model_state=state, backbone_input_latents=sequence
                )
            pcm = model.mimi.decode_from_latent(
                latent * model.flow_lm.emb_std + model.flow_lm.emb_mean, mimi_state
            )
            increment_steps(model.mimi, mimi_state, increment=16)
            tensors[f"step_{step}.noise"] = noise
            tensors[f"step_{step}.latent"] = latent
            tensors[f"step_{step}.eos_logit"] = eos_logits[-1]
            tensors[f"step_{step}.pcm"] = pcm
            sequence = latent
    hook.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = args.output / "model.safetensors"
    save_file({key: value.detach().contiguous() for key, value in tensors.items()}, fixture)
    manifest = {
        "upstream_commit": UPSTREAM_COMMIT, "language": args.language,
        "voice": args.voice, "text": args.text, "prepared": prepared,
        "torch": torch.__version__, "steps": args.steps, "sampler_decode_steps": model.sampler_decode_steps,
        "has_voice_cloning": model.has_voice_cloning,
        "config_path": str(root / "pocket_tts" / "config" / f"{args.language}.yaml"),
        "weights": {"uri": weights_uri, "path": str(weights), "sha256": sha256(weights)},
        "tokenizer": {"uri": config.flow_lm.lookup_table.tokenizer_path,
                      "path": str(tokenizer), "sha256": sha256(tokenizer)},
        "voice_state": {"uri": voice_uri, "path": str(voice), "sha256": sha256(voice)},
        "fixture_sha256": sha256(fixture),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
