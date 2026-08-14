# ai4s-myGPT

A compact, readable implementation of a decoder-only GPT language model for the
AI4S final project. The repository is deliberately small enough to study: it
contains a byte-level BPE tokenizer, causal self-attention, a training loop,
checkpointing, evaluation, and text generation.

## Project layout

```text
configs/              Model and training presets
src/mygpt/            Reusable Python package
scripts/              Command-line entry points
tests/                Unit and smoke tests
notebooks/            Optional experiments
outputs/              Generated data/checkpoints (gitignored)
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

On Apple Silicon, PyTorch will use MPS automatically when it is available.

## Quick start

The default presets train on a local text export of
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Prepare it
once before training. The preparation command streams up to 2,000,000 stories
by default and writes an explicit `<eos>` boundary after every story. On the
first fresh run, training learns and saves a 10,000-token byte-level BPE
tokenizer at `outputs/tokenizers/tinystories-10k.json`.

```bash
# Download and export the default TinyStories subset
python scripts/prepare_data.py

# Equivalent explicit TinyStories selection
python scripts/prepare_data.py --tinystories

# A short smoke-training run
python scripts/train.py --config configs/gpt_small.yaml --max-steps 20

# Inspect validation loss from the final checkpoint
python scripts/evaluate.py --checkpoint outputs/gpt-small-bpe/last.pt

# Generate text
python scripts/sample.py --checkpoint outputs/gpt-small-bpe/last.pt \
  --prompt "The " --max-new-tokens 120
```

Corpora prepared before the BPE migration have blank-line separators but no
`<eos>` markers. Regenerate `outputs/data/tinystories.txt` before starting a
fresh BPE run. Character-tokenized checkpoints are not compatible with this
BPE-only pipeline.

To use your own UTF-8 corpus instead:

```bash
python scripts/prepare_data.py --input path/to/corpus.txt \
  --output outputs/data/custom.txt
python scripts/train.py --config configs/gpt_small.yaml \
  --data outputs/data/custom.txt
```

To fetch the original structured Stanford Alpaca data for later instruction
tuning, run:

```bash
python scripts/prepare_data.py --stanford-alpaca
```

This validates and saves the `instruction`, `input`, and `output` records to
`outputs/data/stanford_alpaca.json`. The dataset is licensed CC BY-NC 4.0 for
non-commercial research use.

## Evaluation datasets

Fetch the recommended language-modeling and reasoning benchmarks with:

```bash
# WikiText-2 raw test split -> outputs/data/wikitext-2-raw-v1-test.txt
python scripts/prepare_data.py --wikitext

# All 57 MMLU subjects in the test split -> outputs/data/mmlu-all-test.json
python scripts/prepare_data.py --mmlu
```

WikiText is saved as UTF-8 text with its raw line and paragraph structure
preserved. MMLU remains structured JSON with its `question`, `subject`, four
`choices`, and integer `answer` fields. Use `--split` to fetch another split
(for example, MMLU's `dev` split for few-shot examples), and `--output` to
choose another destination.

## Instruction tuning

The instruction-tuning stage starts from a completed myGPT checkpoint and
trains only on response targets. BPE checkpoints already contain byte coverage,
EOS, and padding tokens, so SFT does not resize their vocabulary. Its sequence
length follows the pretrained checkpoint.

```bash
# Full baseline configured in configs/gpt_sft.yaml
python scripts/instruction_tune.py --config configs/gpt_sft.yaml

# Resume an interrupted SFT run with a larger final step
python scripts/instruction_tune.py \
  --resume outputs/gpt-sft-bpe/last.pt --max-steps 7000

# Evaluate held-out response loss
python scripts/evaluate.py --checkpoint outputs/gpt-sft-bpe/best.pt

# Generate from the same Stanford Alpaca prompt template used in training
python scripts/instruct.py \
  --checkpoint outputs/gpt-sft-bpe/best.pt \
  --instruction "Explain why the sky is blue."
```

For a short pipeline check without starting the full schedule:

```bash
python scripts/instruction_tune.py --config configs/gpt_sft.yaml \
  --device cpu --max-records 64 --max-steps 2 \
  --output-dir outputs/gpt-sft-smoke
```

This smoke run verifies data preparation, checkpoint loading, training,
evaluation, and generation; it is not evidence of instruction-following
quality. Records with empty responses are skipped. Overlong examples preserve
the complete instruction, trim optional input first, and then truncate the
response to the checkpoint context length.

Use `--max-stories` to change the downloaded TinyStories subset size. The
upstream dataset has separate train and validation splits; this compact pipeline
streams the requested training stories and applies its configured contiguous
train/validation split locally.

Run the tests with:

```bash
pytest -q
```

## Configuring an experiment

Every experiment is controlled by a YAML file. Important settings are:

- `model.block_size`: maximum context length in BPE tokens.
- `model.n_layer`, `n_head`, `n_embd`: Transformer size.
- `data.path`: UTF-8 text file; the bundled presets default to the prepared
  `outputs/data/tinystories.txt` corpus.
- `tokenizer.path`: saved tokenizer JSON, shared by the pretraining presets.
- `tokenizer.vocab_size`: target BPE vocabulary size (10,000 by default).
- `training.max_steps`: number of optimizer updates.
- `training.device`: `auto`, `cpu`, `mps`, or `cuda`.

`gpt_small.yaml` is intended for learning and smoke tests. The medium preset is
still educational rather than production-scale, but requires substantially more
memory and training time.

## Notes

- Checkpoints include the model, optimizer, configuration, serialized tokenizer,
  step, and best validation loss.
- The validation split is the final contiguous portion of the text. For serious
  experiments, provide a sufficiently large corpus and treat this split as a
  development set rather than repeatedly tuning on a final test set.
- Generated prose from a short demo run is only a pipeline check; coherent text
  requires more data and training.
