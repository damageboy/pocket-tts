"""Pinned upstream oracle for the November sync; no checked-in Python oracle is used.

Run with an isolated Python 3.12 environment, installing requirements.lock.txt.
Large model assets stay in --asset-cache. The manifest retains their immutable
Hugging Face URI, revision and SHA-256; local paths may be relocated explicitly.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

KNOWN_COMMITS = {
    '3dbee45d343d7dddd0d105468d17f8dcba14db3e',
    '9a1b48922deb44cd4a805fde279eb33b71c1a473',
    'c91f50f38c6b914d49dbdff51b8d322edf843db2',
}
TARGET_COMMIT = 'c91f50f38c6b914d49dbdff51b8d322edf843db2'
OPTIONS = ('pad_with_spaces_for_short_inputs', 'remove_semicolons',
           'append_terminal_punctuation', 'capitalize_first_letter', 'replace_characters')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_reference(root, expected):
    if expected not in KNOWN_COMMITS:
        raise ValueError(f'Unknown reference revision: {expected}; wrong reference revision')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != expected:
        raise ValueError(f'Wrong reference revision: {head}; expected {expected}')
    changed = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                      '--untracked-files=all', '--', 'pocket_tts',
                                      'pyproject.toml', 'uv.lock'], text=True).strip()
    if changed:
        raise ValueError(f'Modified reference source: {changed}')


def validate_asset(asset, expected_uri=None):
    if expected_uri is not None and asset.get('uri') != expected_uri:
        raise ValueError(f'config/asset mismatch: {asset.get("uri")} != {expected_uri}')
    if sha256(asset['path']) != asset['sha256']:
        raise ValueError(f'asset hash mismatch: {asset["path"]}')


def validate_fixture(manifest_path):
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest['upstream_commit'] not in KNOWN_COMMITS:
        raise ValueError('Unknown reference revision')
    config = manifest['config']
    for key in ('weights', 'tokenizer', 'voice_state'):
        item = dict(manifest[key])
        item['path'] = str(directory / item['path'])
        validate_asset(item)
    if manifest['weights']['uri'] not in (config['weights_path'], config['weights_path_without_voice_cloning']):
        raise ValueError('config/asset mismatch: weights')
    validate_asset(dict(manifest['tokenizer'], path=str(directory / manifest['tokenizer']['path'])),
                   config['flow_lm']['lookup_table']['tokenizer_path'])
    for name, digest in manifest['files'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'fixture hash mismatch: {name}')
    return manifest


def asset(uri, cache):
    """Download only immutable public assets, with fail-closed hash validation."""
    from huggingface_hub import hf_hub_download
    if not uri.startswith('hf://') or '@' not in uri:
        raise ValueError(f'Asset needs immutable hf:// revision: {uri}')
    location, revision = uri[5:].rsplit('@', 1)
    owner, repo, filename = location.split('/', 2)
    if len(revision) != 40:
        raise ValueError(f'Asset revision must be a full SHA: {uri}')
    path = hf_hub_download(f'{owner}/{repo}', filename, revision=revision, cache_dir=cache)
    result = dict(uri=uri, revision=revision, path=str(Path(path).absolute()), sha256=sha256(path))
    validate_asset(result, uri)
    return result


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def text_fixtures(tokenizer, options, extended=False):
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
    inputs = ['hello world', '  hello   world\nagain  ', 'Dr. Miller met Prof. Smith. Next!',
              'The fee is $822.95, please pay tomorrow. Then leave.',
              'I counted to 3. 14 people were there.', 'Visit www.kyutai.org. Email jo@icloud.com.',
              '“Stop.” She left. (Quietly.) Then returned.', 'Mr. Mrs. Ms. Rev. vs. e.g. i.e. Smith.',
              'The time is 4 p.m. Next one.', 'one two three four', 'one two three four five',
              'ends with a quote"', 'ends with a dash—', '"Hi?", she said',
              'one clause, another clause; a third clause: final clause. Next sentence!', '', ' \t\n ']
    if extended:
        inputs += [
            'Pi is 3.14. Next comes a sentence.',
            'I counted to 3. 14 people were there.',
            'The fee is $822.95, which must be paid, before leaving. Then leave.',
            'La somme est de 822,95 €, ensuite partez. Puis revenez.',
            'The fee is 1,994.42, which must be paid. Then leave.',
            'Dr. Miller met Prof. Smith. Next!',
            'Visit https://www.kyutai.org. Email jo@icloud.com. Then leave.',
            'Meet at 4 p.m. Next comes a sentence.',
            'We live in the U.S.',
            'He said “Stop.” Then left.',
            'He left. “Hi.”',
            'He said "Stop.")] Then left.',
            'St. John left. Next came Jane.',
            'The fee is $822.95.',
        ]
    results = []
    for text in dict.fromkeys(inputs):
        case = dict(input=text, options=options)
        try:
            prepared, tail = prepare_text_prompt(text, **options)
            case.update(prepared=prepared, frames_after_eos_guess=tail,
                        token_ids=tokenizer(prepared)[0].tolist(),
                        chunks={str(limit): split_into_best_sentences(tokenizer, text, limit, **options)
                                for limit in ((1, 4, 8, 12, 25, 50) if extended else (12, 25, 50))})
        except ValueError as error:
            case['error'] = str(error)
        results.append(case)
    return results


def prompt_fixtures():
    """Deterministic edge cases evaluated by the pinned upstream implementation."""
    import torch
    from pocket_tts.data.audio_utils import end_on_pause
    signal = torch.sin(torch.arange(960, dtype=torch.float32) * .1)
    # Opposite signs and unequal channels catch averaging samples before squaring.
    batch = torch.zeros(2, 2, 1920)
    batch[0, 0, :480] = 1
    batch[0, 1, :480] = 1
    batch[1, 0, 960:1440] = .1
    batch[1, 1, 960:1440] = -.1
    floor = torch.cat([torch.ones(480), torch.full((480,), 10 ** (-35 / 20)), torch.full((480,), .001)])
    inputs = {
        'short': torch.arange(100, dtype=torch.float32)[None, None] / 100,
        'all_zero': torch.zeros(1, 1, 1440),
        'speech_and_pause': torch.cat([signal, torch.zeros(1440)])[None, None],
        'speech_ending': signal[None, None],
        'partial_frame': torch.cat([signal, torch.zeros(480), torch.ones(239)])[None, None],
        'long_prompt': torch.cat([signal.repeat(325), torch.zeros(24000)])[None, None],
        'batch_channels': batch,
        'strict_floor': floor[None, None],
    }
    result = {}
    for name, values in inputs.items():
        before = values.clone()
        result[f'{name}.input'] = values
        result[f'{name}.output'] = end_on_pause(values, 24000)
        assert torch.equal(before, values), name
    return {name: value.contiguous().clone() for name, value in result.items()}


def capture_state(tensors, prefix, state, end, start=0):
    for name, values in state.items():
        for key, value in values.items():
            if key == 'cache':
                value = value[:, :, start:end]
            tensors[f'{prefix}.{name}.{key}'] = value.detach().clone().contiguous()


def generate(args):
    root = args.reference.resolve()
    verify_reference(root, args.expected_commit)  # Before importing any oracle code.
    sys.path.insert(0, str(root))
    import pocket_tts
    import torch
    import yaml
    from safetensors.torch import save_file
    from pocket_tts import TTSModel
    from pocket_tts.models.text_chunking import prepare_text_prompt
    from pocket_tts.modules.stateful_module import init_states, increment_steps
    from pocket_tts.utils.config import load_config
    from pocket_tts.utils.utils import get_predefined_voice
    if Path(pocket_tts.__file__).resolve().parent != root / 'pocket_tts':
        raise ValueError('Imported pocket_tts is not from the pinned reference')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original_config = root / 'pocket_tts/config' / f'{args.language}.yaml'
    config = load_config(original_config)
    options = {name: getattr(config, name) for name in OPTIONS}
    weights = asset(config.weights_path_without_voice_cloning, args.asset_cache)
    tokenizer = asset(config.flow_lm.lookup_table.tokenizer_path, args.asset_cache)
    voice = asset(get_predefined_voice(args.language, args.voice), args.asset_cache)
    source_config = yaml.safe_load(original_config.read_text())
    # Preserve normative URI config, and build a separate local loading config.
    (output / 'config.yaml').write_bytes(original_config.read_bytes())
    local_config = copy.deepcopy(source_config)
    local_config.pop('timestamp_heads', None)
    local_config['weights_path'] = weights['path']
    local_config['weights_path_without_voice_cloning'] = weights['path']
    local_config['flow_lm']['lookup_table']['tokenizer_path'] = tokenizer['path']
    (output / 'local.yaml').write_text(yaml.safe_dump(local_config))
    model = TTSModel.load_model(config=str(output / 'local.yaml'), sampler_decode_steps=args.sampler_decode_steps,
                                temp=0.0)
    model.eval()
    voice_state = model.get_state_for_audio_prompt(voice['path'])
    prepared, _ = prepare_text_prompt(args.text, **options)
    tokens = model.flow_lm.conditioner.prepare(prepared)
    tensors = {'tokens': tokens, 'text_embeddings': model.flow_lm.conditioner(tokens)}
    state = copy.deepcopy(voice_state)
    voice_length = model._flow_lm_current_end(state)
    capture_state(tensors, 'voice', state, voice_length)
    model._expand_kv_cache(state, model._flow_lm_current_end(state) + tokens.shape[1] + args.steps + 1)
    eos_logits = []
    hook = model.flow_lm.out_eos.register_forward_hook(
        lambda module, inputs, out: eos_logits.append(out.detach().clone()))
    with torch.no_grad():
        model._run_flow_lm_and_increment_step(model_state=state, text_tokens=tokens)
        capture_state(tensors, 'prefill', state, model._flow_lm_current_end(state), voice_length)
        sequence = torch.full((1, 1, model.flow_lm.ldim), float('nan'))
        mimi_state = init_states(model.mimi, batch_size=1, sequence_length=(args.steps + 12) * 16)
        latents = []
        for step in range(args.steps):
            noise = (torch.sin(torch.arange(model.flow_lm.ldim) * 0.13 + step) * 0.3**0.5)[None]
            with patch('torch.nn.init.normal_', side_effect=lambda tensor, **kwargs: tensor.copy_(noise)):
                latent, _ = model._run_flow_lm_and_increment_step(
                    model_state=state, backbone_input_latents=sequence)
            pcm = model.mimi.decode_from_latent(
                latent * model.flow_lm.emb_std + model.flow_lm.emb_mean, mimi_state)
            increment_steps(model.mimi, mimi_state, increment=16)
            for name, value in dict(noise=noise, latent=latent, eos_logit=eos_logits[-1], pcm=pcm).items():
                tensors[f'step_{step}.{name}'] = value.detach().clone()
            latents.append(latent)
            sequence = latent
        # Fade is orchestration only: raw decoder tensors remain untouched.
        faded = tensors['step_0.pcm'].clone()
        n = min(model.sample_rate // 200, faded.shape[-1])
        faded[..., :n] *= torch.linspace(0, 1, model.sample_rate // 200)[:n]
        tensors['output_fade.first_frame'] = faded
        continuation = None
        if args.teacher_force_previous_chunk:
            forced = torch.cat(latents, dim=1)
            combined = f'{args.text} {args.continuation_text}'
            continued_text, _ = prepare_text_prompt(combined, **options)
            continued_tokens = model.flow_lm.conditioner.prepare(continued_text)
            continued_state = copy.deepcopy(voice_state)
            model._expand_kv_cache(continued_state, model._flow_lm_current_end(continued_state)
                                  + continued_tokens.shape[1] + forced.shape[1] + args.steps + 1)
            bos = torch.full_like(forced[:, :1], float('nan'))
            prefix = torch.cat([bos, forced[:, :-1]], dim=1)
            # One real upstream call performs combined text + audio prefill.
            model._run_flow_lm_and_increment_step(model_state=continued_state,
                                                  text_tokens=continued_tokens,
                                                  backbone_input_latents=prefix)
            tensors['continuation.tokens'] = continued_tokens
            tensors['continuation.forced_latents'] = forced
            tensors['continuation.forced_prefix'] = prefix
            tensors['continuation.first_input'] = forced[:, -1:]
            capture_state(tensors, 'continuation.prefill', continued_state,
                          model._flow_lm_current_end(continued_state), voice_length)
            warmup = forced[:, -model._DECODER_WARMUP_FRAMES:]
            tensors['continuation.warmup_latents'] = warmup
            decoder = init_states(model.mimi, 1, (args.steps + warmup.shape[1]) * 16)
            tensors['continuation.discarded_warmup_pcm'] = model.mimi.decode_from_latent(
                warmup * model.flow_lm.emb_std + model.flow_lm.emb_mean, decoder)
            increment_steps(model.mimi, decoder, increment=warmup.shape[1] * 16)
            capture_state(tensors, 'continuation.decoder_warmup', decoder, warmup.shape[1] * 16)
            sequence = forced[:, -1:]
            for step in range(args.steps):
                noise = tensors[f'step_{step}.noise']
                with patch('torch.nn.init.normal_', side_effect=lambda tensor, **kwargs: tensor.copy_(noise)):
                    latent, _ = model._run_flow_lm_and_increment_step(
                        model_state=continued_state, backbone_input_latents=sequence)
                pcm = model.mimi.decode_from_latent(
                    latent * model.flow_lm.emb_std + model.flow_lm.emb_mean, decoder)
                increment_steps(model.mimi, decoder, increment=16)
                for name, value in dict(noise=noise, latent=latent, eos_logit=eos_logits[-1], pcm=pcm).items():
                    tensors[f'continuation.step_{step}.{name}'] = value.detach().clone()
                sequence = latent
            continuation = dict(combined_text=combined, prepared=continued_text,
                                new_text=args.continuation_text, forced_frames=forced.shape[1],
                                forced_prefix_frames=prefix.shape[1], warmup_frames=warmup.shape[1],
                                emitted_frames=args.steps, mode='bounded fixed-noise trace; not EOS-limited audio')
    hook.remove()
    # Copy every tensor: safetensors refuses overlapping cache views.
    logical_dtypes = {name: 'bool' for name, value in tensors.items() if value.dtype == torch.bool}
    save_file({name: (value.to(torch.uint8) if value.dtype == torch.bool else value).detach().contiguous().clone()
               for name, value in tensors.items()},
              output / 'model.safetensors')
    save_json(output / 'text.json', text_fixtures(model.flow_lm.conditioner.tokenizer, options,
                                                   extended=args.expected_commit == TARGET_COMMIT))
    if args.expected_commit == TARGET_COMMIT:
        save_file(prompt_fixtures(), output / 'prompt.safetensors')
    manifest = dict(schema_version=2, upstream_commit=args.expected_commit, language=args.language,
                    voice=args.voice, text=args.text, prepared=prepared, token_ids=tokens[0].tolist(),
                    python=platform.python_version(), packages={dist.metadata['Name']: dist.version
                    for dist in importlib.metadata.distributions() if dist.metadata['Name'] != 'pocket-tts'},
                    device='cpu', dtype='float32', torch_threads=1, temperature=0.0, noise_override=True,
                    config=source_config, config_sha256=sha256(output / 'config.yaml'),
                    config_path='config.yaml', weights=weights, tokenizer=tokenizer, voice_state=voice,
                    tokenizer_family=config.flow_lm.lookup_table.tokenizer, options=options,
                    steps=args.steps, prefill_cache_start=voice_length, sampler_decode_steps=model.sampler_decode_steps,
                    eos_threshold=model.eos_threshold, has_voice_cloning=False,
                    fixed_noise='sin(arange(ldim)*0.13 + step)*sqrt(0.3)',
                    continuation=continuation, tensor_names=sorted(tensors), logical_dtypes=logical_dtypes,
                    fixture_sha256=sha256(output / 'model.safetensors'))
    manifest['files'] = {file.name: sha256(file) for file in sorted(output.iterdir())
                         if file.is_file() and file.name not in ('manifest.json', 'local.yaml')}
    save_json(output / 'manifest.json', manifest)
    (output / 'local.yaml').unlink()
    print(json.dumps(dict(output=str(output), tensors=len(tensors), fixture_sha256=manifest['fixture_sha256'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--language', default='english')
    parser.add_argument('--voice', default='alba')
    parser.add_argument('--text', default='The price is 3.14 euros, not twelve.')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--sampler-decode-steps', type=int, default=3)
    parser.add_argument('--teacher-force-previous-chunk', action='store_true')
    parser.add_argument('--continuation-text', default='Tomorrow, Anna will read another book.')
    parser.add_argument('--asset-cache', type=Path, default=Path(tempfile.gettempdir()) / 'pocket-tts-reference-assets')
    generate(parser.parse_args())


if __name__ == '__main__':
    main()
