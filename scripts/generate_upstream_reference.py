"""Generate deterministic fixtures from an unmodified, pinned upstream checkout.

Use the interpreter in the isolated reference environment, not python-reference:
  python scripts/generate_upstream_reference.py --reference PATH --output PATH
No model weights are required. Tokenizers are downloaded at their config pins.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


UPSTREAM_COMMIT = "3dbee45d343d7dddd0d105468d17f8dcba14db3e"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_reference(root):
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Wrong reference revision: {revision}; expected {UPSTREAM_COMMIT}")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all",
         "--", "pocket_tts", "pyproject.toml"], text=True
    ).strip()
    if changed:
        raise ValueError(f"Modified reference source:\n{changed}")


def generate(root, output, languages):
    verify_reference(root)
    sys.path.insert(0, str(root))
    import pocket_tts
    import torch
    from safetensors.torch import save_file
    from pocket_tts.models.flow_lm import lsd_decode, ot_decode
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
    from pocket_tts.modules.mlp import LayerNorm, RMSNorm, SimpleMLPAdaLN
    from pocket_tts.modules.text_conditioner import build_tokenizer
    from pocket_tts.utils.config import load_config
    from pocket_tts.utils.utils import download_if_necessary

    if Path(pocket_tts.__file__).resolve().parent != root / "pocket_tts":
        raise ValueError("Imported pocket_tts is not from the pinned reference")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "upstream_commit": UPSTREAM_COMMIT,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "tokenizers", "sentencepiece", "safetensors")},
        "dtype": "float32", "device": "cpu", "torch_threads": 1,
        "files": {}, "languages": {},
    }
    text_cases = []
    inputs = [
        "hello world", "  hello   world\nagain  ", "3.14 is less than 12.75. Next sentence!",
        '"Hi?", she said', "bonjour : l’été d’aujourd’hui", "Grüße aus Köln; schön!",
        "Olá, amanhã é terça-feira", "Perché l’acqua è fredda?", "¡Hola! ¿Cómo estás?",
        "Goedenavond, dit is een Nederlandse zin.", "one two three four", "one two three four five",
        'ends with a quote"', "ends with a dash—", "", " \t\n ", '"()[]"',
        "This sentence is deliberately long, with several clauses; each one needs to remain "
        "in order: even when the token budget is small. Then we start another sentence!",
        "uninterrupted " * 60,
    ]
    for language in languages:
        config_path = root / "pocket_tts" / "config" / f"{language}.yaml"
        config = load_config(config_path)
        lookup = config.flow_lm.lookup_table
        tokenizer_path = download_if_necessary(lookup.tokenizer_path)
        tokenizer = build_tokenizer(lookup.n_bins, lookup.tokenizer_path, lookup.tokenizer)
        tokenizer_name = f"{language}.tokenizer.json"
        shutil.copyfile(tokenizer_path, output / tokenizer_name)
        manifest["languages"][language] = {
            "config_sha256": sha256(config_path),
            "tokenizer_uri": lookup.tokenizer_path,
            "tokenizer_file": tokenizer_name,
            "n_bins": lookup.n_bins,
            "weights_uri": config.weights_path,
            "open_weights_uri": config.weights_path_without_voice_cloning,
        }
        options = {name: getattr(config, name) for name in (
            "pad_with_spaces_for_short_inputs", "remove_semicolons",
            "append_terminal_punctuation", "capitalize_first_letter", "replace_characters",
        )}
        cases = [(text, options) for text in inputs]
        cases += [("salAm", dict(options, capitalize_first_letter=False)),
                  ("keep trailing,", dict(options, append_terminal_punctuation=False)),
                  ("hi", dict(options, pad_with_spaces_for_short_inputs=True))]
        for text, settings in cases:
            case = {"language": language, "input": text, "options": settings}
            try:
                prepared, tail_guess = prepare_text_prompt(text, **settings)
                case.update(prepared=prepared, frames_after_eos_guess=tail_guess,
                            tokens=tokenizer.encode(prepared))
                case["chunks"] = {str(limit): split_into_best_sentences(
                    tokenizer, text, limit, **settings) for limit in (12, 50)}
            except ValueError as error:
                case["error"] = str(error)
            text_cases.append(case)
    (output / "text.json").write_text(
        json.dumps(text_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    tensors = {}
    x = torch.tensor([[0.25, -1.75, 2.5, 0.125]], dtype=torch.float32)
    condition = torch.tensor([[0.3, -0.7, 1.1, -1.3, 0.2, 2.0]], dtype=torch.float32)
    tensors["input"] = x
    tensors["condition"] = condition
    tensors["gelu"] = torch.nn.functional.gelu(x, approximate="tanh")
    rms = RMSNorm(4)
    norm = LayerNorm(4, elementwise_affine=False)
    tensors["rms_norm"] = rms(x)
    tensors["layer_norm"] = norm(x)
    for kind, time_conditions, decode in (("lsd", 2, lsd_decode), ("flow_matching", 1, ot_decode)):
        head = SimpleMLPAdaLN(4, 8, 4, 6, 2, num_time_conds=time_conditions)
        with torch.no_grad():
            for index, parameter in enumerate(head.parameters()):
                values = torch.arange(parameter.numel(), dtype=torch.float32)
                parameter.copy_((0.08 * torch.sin(values * 0.17 + index)).reshape(parameter.shape))
            for steps in (1, 3):
                tensors[f"{kind}.steps_{steps}"] = decode(
                    lambda *args: head(condition, *args), x, steps
                )
        for name, tensor in head.state_dict().items():
            tensors[f"{kind}.weights.{name}"] = tensor
    save_file({key: tensor.detach().contiguous() for key, tensor in tensors.items()},
              output / "operators.safetensors")
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--languages", nargs="+", default=[
        "english", "french", "german", "italian", "spanish", "portuguese", "dutch",
    ])
    args = parser.parse_args()
    generate(args.reference.resolve(), args.output.resolve(), args.languages)


if __name__ == "__main__":
    main()
