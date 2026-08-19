# myGPT Evaluation Results

## Technical summary

Evaluation now covers both the selected pretraining checkpoint and the selected
instruction-tuned (SFT) checkpoint. SFT improves broad-text likelihood and
instruction-shaped response behavior, but it does not establish factual or
multiple-choice reasoning ability.

- **Held-out SFT response loss:** the complete deterministic Alpaca validation
  split gives response-only loss **2.5409** and response-token perplexity
  **12.69** over 2,600 usable records (325 batches).
- **WikiText-2:** SFT lowers full-test BPE-token perplexity from **929.84** to
  **183.63** on the identical 442,868 next-token targets. Average NLL falls from
  **6.8350** to **5.2129**. This is a substantial out-of-domain likelihood
  improvement, although the absolute perplexity remains high.
- **MMLU:** the SFT checkpoint scores **23.80%** (**3,342/14,042**) with its
  native Stanford Alpaca instruction template and **22.87%**
  (**3,211/14,042**) with the original completion prompt. Both are below the
  25% four-choice chance baseline and below the pretraining result of **25.27%**.
- **MMLU answer bias remains severe:** instruction prompting produces almost
  only A or C, while completion prompting produces A for **94.20%** of items.
- **Qualitative SFT generation:** the model returns direct task-shaped answers,
  but the story drifts away from gardening, the sky explanation is incorrect,
  and the presentation advice is repetitive and awkward. These fixed samples
  show partial instruction-format learning, not reliable instruction following.

## Evaluation scope

| Item | Pretraining | SFT |
|---|---:|---:|
| Checkpoint | `outputs/gpt-pretrain-bpe/best.pt` | `outputs/gpt-sft-bpe/best.pt` |
| Checkpoint step | 75,000 | 59,500 |
| Source pretraining step | — | 75,000 |
| Saved best stage validation loss | 1.1276 | 2.5576 response-only |
| Parameters | 66,053,760 | 66,053,760 |
| Transformer shape | 12 layers, 10 heads, width 640 | Same |
| Context length | 1,024 BPE tokens | 1,024 BPE tokens |
| Vocabulary | 10,000 byte-level BPE tokens | Same tokenizer and vocabulary |
| Evaluation device | CPU | Apple MPS |

The evaluations use the complete locally prepared WikiText-2 raw test split and
complete MMLU test split. MMLU is zero-shot because no development split was
provided. For SFT, `auto` prompt selection uses the same Stanford Alpaca
instruction template used in training; a second completion-mode run holds the
pretraining prompt fixed for direct comparison. Each qualitative assessment
uses three fixed prompts and one deterministic sample per prompt.

| Evaluation | Scope | Denominator |
|---|---|---:|
| WikiText-2 raw test | Complete test text | 442,869 tokens; 442,868 predictions |
| Alpaca response validation | Complete deterministic 5% split | 2,600 usable records; 325 batches |
| MMLU test | All 57 subjects, zero-shot | 14,042 questions per prompt mode |
| Qualitative generation | Story, explanation, conversation | 3 prompts per checkpoint |

## SFT checkpoint evaluation

### Complete held-out response loss

The SFT evaluator reconstructed the dataset contract saved in the checkpoint:
52,002 Stanford Alpaca records, a seeded 95/5 split, response-only targets, and
the `stanford_alpaca_v1` prompt template. The validation split contains 2,601
records; one empty response is skipped, leaving 2,600 usable records. No
validation prompt, input, or response was truncated.

| Metric | Result |
|---|---:|
| Full response-only validation loss | 2.5409 |
| Response-token perplexity | 12.69 |
| Usable validation records | 2,600 / 2,601 |
| Validation batches | 325 |
| Batch size | 8 |
| Saved training-time best validation loss | 2.5576 |

The full result is slightly lower than the saved training-time value because
training selected checkpoints using at most 50 validation batches, whereas this
evaluation uses all 325 batches. Response-token perplexity is conditional on the
instruction prompt and is not directly comparable with raw WikiText perplexity.

### WikiText language-model retention and transfer

| Metric | Pretraining | SFT | Change after SFT |
|---|---:|---:|---:|
| Average NLL | 6.8350 | 5.2129 | -1.6221 |
| Perplexity | 929.84 | 183.63 | -80.25% (5.06x lower) |
| Bits per token | 9.8608 | 7.5206 | -2.3402 |
| Predictions | 442,868 | 442,868 | Identical targets |
| Context windows | 865 | 865 | Identical |

This is an apples-to-apples comparison: both checkpoints share the tokenizer,
model shape, context length, raw text, stride, and target-counting convention.
SFT substantially improves likelihood on WikiText, plausibly because Alpaca
adds broader prose and task language than TinyStories. The SFT perplexity of
183.63 is still high, so this is evidence of improvement rather than strong
general-domain language modeling.

### MMLU remains below chance

| Metric | Pretraining completion | SFT instruction | SFT completion |
|---|---:|---:|---:|
| Correct / total | 3,548 / 14,042 | 3,342 / 14,042 | 3,211 / 14,042 |
| Micro accuracy | 25.27% | 23.80% | 22.87% |
| 95% Wilson interval | 24.56%–25.99% | 23.10%–24.51% | 22.18%–23.57% |
| Macro subject accuracy | 24.98% | 23.71% | 23.15% |
| Difference from 25% chance | +0.27 pp | -1.20 pp | -2.13 pp |
| Prompts over context | 11 | 12 | 11 |

Instruction mode wraps the conventional question/choices/`Answer:` task in the
checkpoint's Stanford Alpaca template and scores response labels `A`, `B`, `C`,
and `D` without a leading space, matching SFT response tokenization. Completion
mode uses exactly the pretraining evaluator's prompt and continuations. Neither
SFT result supports a reasoning claim, and the prompt-sensitive label collapse
shows that accuracy changes mainly reflect output calibration.

| Predicted label | SFT instruction | Share | SFT completion | Share | Gold share |
|---|---:|---:|---:|---:|---:|
| A | 7,388 | 52.61% | 13,228 | 94.20% | 22.95% |
| B | 274 | 1.95% | 79 | 0.56% | 24.65% |
| C | 6,368 | 45.35% | 222 | 1.58% | 25.51% |
| D | 12 | 0.09% | 513 | 3.65% | 26.89% |

For the native instruction prompt, the highest observed subject accuracy is
international law (33.9%, 41/121), followed by world religions (33.3%, 57/171).
The lowest is high-school government and politics (17.1%, 33/193), followed by
professional medicine (17.3%, 47/272). These are descriptive slices of an
overall below-chance, label-biased result and should not be treated as stable
subject capabilities.

### Complete SFT instruction-mode per-subject results

| Subject | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Abstract algebra | 24 | 100 | 24.0% |
| Anatomy | 42 | 135 | 31.1% |
| Astronomy | 36 | 152 | 23.7% |
| Business ethics | 24 | 100 | 24.0% |
| Clinical knowledge | 69 | 265 | 26.0% |
| College biology | 34 | 144 | 23.6% |
| College chemistry | 18 | 100 | 18.0% |
| College computer science | 24 | 100 | 24.0% |
| College mathematics | 22 | 100 | 22.0% |
| College medicine | 39 | 173 | 22.5% |
| College physics | 18 | 102 | 17.6% |
| Computer security | 28 | 100 | 28.0% |
| Conceptual physics | 65 | 235 | 27.7% |
| Econometrics | 28 | 114 | 24.6% |
| Electrical engineering | 37 | 145 | 25.5% |
| Elementary mathematics | 87 | 378 | 23.0% |
| Formal logic | 24 | 126 | 19.0% |
| Global facts | 28 | 100 | 28.0% |
| High school biology | 58 | 310 | 18.7% |
| High school chemistry | 52 | 203 | 25.6% |
| High school computer science | 26 | 100 | 26.0% |
| High school European history | 38 | 165 | 23.0% |
| High school geography | 44 | 198 | 22.2% |
| High school government and politics | 33 | 193 | 17.1% |
| High school macroeconomics | 85 | 390 | 21.8% |
| High school mathematics | 65 | 270 | 24.1% |
| High school microeconomics | 51 | 238 | 21.4% |
| High school physics | 34 | 151 | 22.5% |
| High school psychology | 121 | 545 | 22.2% |
| High school statistics | 40 | 216 | 18.5% |
| High school US history | 51 | 204 | 25.0% |
| High school world history | 63 | 237 | 26.6% |
| Human aging | 50 | 223 | 22.4% |
| Human sexuality | 35 | 131 | 26.7% |
| International law | 41 | 121 | 33.9% |
| Jurisprudence | 25 | 108 | 23.1% |
| Logical fallacies | 41 | 163 | 25.2% |
| Machine learning | 25 | 112 | 22.3% |
| Management | 20 | 103 | 19.4% |
| Marketing | 69 | 234 | 29.5% |
| Medical genetics | 19 | 100 | 19.0% |
| Miscellaneous | 200 | 783 | 25.5% |
| Moral disputes | 95 | 346 | 27.5% |
| Moral scenarios | 214 | 895 | 23.9% |
| Nutrition | 84 | 306 | 27.5% |
| Philosophy | 63 | 311 | 20.3% |
| Prehistory | 74 | 324 | 22.8% |
| Professional accounting | 68 | 282 | 24.1% |
| Professional law | 373 | 1,534 | 24.3% |
| Professional medicine | 47 | 272 | 17.3% |
| Professional psychology | 150 | 612 | 24.5% |
| Public relations | 20 | 110 | 18.2% |
| Security studies | 49 | 245 | 20.0% |
| Sociology | 54 | 201 | 26.9% |
| US foreign policy | 25 | 100 | 25.0% |
| Virology | 36 | 166 | 21.7% |
| World religions | 57 | 171 | 33.3% |

### SFT qualitative samples show partial task-shape learning

Generation used the saved Alpaca prompt template, temperature 0.8, top-k 40,
160 maximum new tokens, and seeds 42–44. All outputs below are unedited.

**Story instruction (seed 42)**

> Write a short children's story about a robot that learns to garden.

**Model output**

> Once upon a time, there was a robot named Fyri who had never been out on the
> same road. He was a robot who felt like he was never in the same area. He felt
> like he was going on a new adventure, no matter how hard it took to get there.
>
> One day, Fyri decided it was time to take a step into the forest. It was an
> adventure and he was sure he would never be able to get there. He went walking
> and looking around, but he couldn't make any sound.
>
> He continued his journey, but he still felt like he was the only one that he
> had ever been on the other side. After a long and difficult journey, he finally
> arrived at the end of the forest.
>
> He was so

The output has the requested children's-story form and includes a robot, but it
omits gardening, repeats itself, and ends mid-sentence at the token limit.

**Explanation instruction (seed 43)**

> Explain why the sky appears blue to a curious ten-year-old.

**Model output**

> The sky appears blue because of the vast blue hue to its vast blue hue and its
> beauty. The blue hue is also captivating of a person's interests, with a sense
> of purpose and optimism. The vast blue hue is often associated with the blue
> and is a desire to adapt.

The model attempts a direct explanation, but it is circular and never mentions
sunlight, atmospheric scattering, or wavelength. It is factually inadequate.

**Conversation instruction and input (seed 44)**

> Respond helpfully and politely to the user.
>
> User: I am nervous about my first presentation tomorrow. What should I do?

**Model output**

> To help you stay organized, it is important to set a timeline and focus on one
> thing to do your research. Check out your presentation and make sure to have a
> priority time for it. Make sure to include all of the tasks you need to achieve
> as well as stay organized. Allow your presentation on time and get some time to
> plan.

The response is on topic and offers planning advice, but it is repetitive,
grammatically awkward, and lacks concrete presentation-specific guidance.

## Pretraining WikiText shows a large out-of-domain language-modeling gap

| Metric | Result |
|---|---:|
| Average negative log-likelihood | 6.8350 |
| Perplexity | 929.84 |
| Bits per token | 9.8608 |
| Evaluated BPE tokens | 442,869 |
| Next-token predictions | 442,868 |
| Context windows | 865 |

The checkpoint's saved TinyStories validation loss is 1.1276, corresponding to
a within-domain perplexity of approximately 3.09. That figure and the WikiText
perplexity are computed with the same checkpoint tokenizer, but on very different
corpora, so the contrast is a domain-shift diagnostic rather than a like-for-like
benchmark comparison. The result indicates that a model specialized on
TinyStories assigns low probability to the broader vocabulary, formatting, and
subject matter in WikiText.

WikiText was encoded as one preserved raw-text sequence. Evaluation used a
1,024-token context, stride 512, and overlapping windows. Every target token
after the first was counted exactly once. Perplexity is BPE-token perplexity and
should not be directly compared with results that use a different tokenizer or
word-level normalization.

## Pretraining MMLU is at chance and dominated by answer-label bias

| Metric | Result |
|---|---:|
| Correct answers | 3,548 / 14,042 |
| Micro accuracy | 25.27% |
| 95% Wilson interval | 24.56%–25.99% |
| Uniform four-choice chance baseline | 25.00% |
| Difference from chance | +0.27 percentage points |
| Macro accuracy across 57 subjects | 24.98% |
| Questions whose prompts exceeded context | 11 (0.08%) |

The aggregate accuracy is not meaningfully above chance. The label distribution
shows why accuracy alone would be misleading:

| Predicted label | Predictions | Share of predictions | Gold-answer share |
|---|---:|---:|---:|
| A | 404 | 2.88% | 22.95% |
| B | 549 | 3.91% | 24.65% |
| C | 12,974 | 92.39% | 25.51% |
| D | 115 | 0.82% | 26.89% |

The mean answer-label log-likelihoods were -4.988 for A, -4.121 for B,
-3.264 for C, and -4.815 for D. The model therefore has a strong prompt-level
preference for the token representing ` C`; it is not reliably selecting answers
from question content. Subject differences below are descriptive only: with 57
simultaneous subject cuts and mostly 100–300 questions per subject, the apparent
high and low subjects should not be treated as stable capabilities.

The highest observed accuracy was international law (38.0%, 46/121), followed by
anatomy (33.3%, 45/135). The lowest was formal logic (15.1%, 19/126), followed by
professional medicine (16.5%, 45/272). These variations do not override the
near-chance overall result and pervasive C-label bias.

### Complete per-subject results

| Subject | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Abstract algebra | 30 | 100 | 30.0% |
| Anatomy | 45 | 135 | 33.3% |
| Astronomy | 46 | 152 | 30.3% |
| Business ethics | 21 | 100 | 21.0% |
| Clinical knowledge | 58 | 265 | 21.9% |
| College biology | 34 | 144 | 23.6% |
| College chemistry | 19 | 100 | 19.0% |
| College computer science | 28 | 100 | 28.0% |
| College mathematics | 23 | 100 | 23.0% |
| College medicine | 43 | 173 | 24.9% |
| College physics | 22 | 102 | 21.6% |
| Computer security | 26 | 100 | 26.0% |
| Conceptual physics | 49 | 235 | 20.9% |
| Econometrics | 28 | 114 | 24.6% |
| Electrical engineering | 36 | 145 | 24.8% |
| Elementary mathematics | 102 | 378 | 27.0% |
| Formal logic | 19 | 126 | 15.1% |
| Global facts | 32 | 100 | 32.0% |
| High school biology | 78 | 310 | 25.2% |
| High school chemistry | 63 | 203 | 31.0% |
| High school computer science | 32 | 100 | 32.0% |
| High school European history | 43 | 165 | 26.1% |
| High school geography | 48 | 198 | 24.2% |
| High school government and politics | 41 | 193 | 21.2% |
| High school macroeconomics | 93 | 390 | 23.8% |
| High school mathematics | 72 | 270 | 26.7% |
| High school microeconomics | 51 | 238 | 21.4% |
| High school physics | 40 | 151 | 26.5% |
| High school psychology | 117 | 545 | 21.5% |
| High school statistics | 45 | 216 | 20.8% |
| High school US history | 55 | 204 | 27.0% |
| High school world history | 57 | 237 | 24.1% |
| Human aging | 43 | 223 | 19.3% |
| Human sexuality | 25 | 131 | 19.1% |
| International law | 46 | 121 | 38.0% |
| Jurisprudence | 25 | 108 | 23.1% |
| Logical fallacies | 47 | 163 | 28.8% |
| Machine learning | 30 | 112 | 26.8% |
| Management | 18 | 103 | 17.5% |
| Marketing | 58 | 234 | 24.8% |
| Medical genetics | 21 | 100 | 21.0% |
| Miscellaneous | 217 | 783 | 27.7% |
| Moral disputes | 95 | 346 | 27.5% |
| Moral scenarios | 221 | 895 | 24.7% |
| Nutrition | 75 | 306 | 24.5% |
| Philosophy | 91 | 311 | 29.3% |
| Prehistory | 96 | 324 | 29.6% |
| Professional accounting | 74 | 282 | 26.2% |
| Professional law | 418 | 1,534 | 27.2% |
| Professional medicine | 45 | 272 | 16.5% |
| Professional psychology | 163 | 612 | 26.6% |
| Public relations | 23 | 110 | 20.9% |
| Security studies | 60 | 245 | 24.5% |
| Sociology | 49 | 201 | 24.4% |
| US foreign policy | 27 | 100 | 27.0% |
| Virology | 36 | 166 | 21.7% |
| World religions | 49 | 171 | 28.7% |

## Pretraining qualitative samples show fluency but weak task adherence

Generation used temperature 0.8, top-k 40, a maximum of 160 new tokens, and
seeds 42–44. The checkpoint is pretrained rather than instruction-tuned, so the
test measures prompt continuation rather than instruction following. All outputs
below are unedited.

### Story completion — coherent but repetitive

**Prompt (seed 42)**

> Once upon a time, a small robot found a seed

**Model output**

> . The robot wanted to grow big and strong. The robot thought, "Maybe if I
> plant my seed, I will have a big tree and a nice tree."
>
> The robot dug a hole in the ground. He dug and dug until his seed began to
> grow. The robot was very happy because his seed had grown. He had a big tree
> and a nice tree.
>
> The robot took care of his tree every day. He planted more and more. Soon,
> the tree had many big, pretty tree. The robot loved his new tree and loved it
> a lot.

The sample maintains the seed/tree topic and a simple event sequence. Repetition
and the grammatical error “many big, pretty tree” show that fluency is limited.

### Factual explanation — fluent form, incorrect content

**Prompt (seed 43)**

> The reason the sky looks blue during the day is

**Model output**

> beginning to rise. The sun comes out and Tom and Anna get tired and hungry.
> They want to go home, but they are also very tired.
>
> Then they see a big cloud. It is white and fluffy and shiny. It makes the
> cloud bigger and bigger. It looks like a giant balloon. Tom and Anna are
> surprised and happy. They wonder who made it.
>
> They run to the cloud and pick it up. It is soft and warm and warm. They hug
> it and each other. They are glad they found it. They tell their mom and dad
> about the cloud and the cloud. Their mom and dad smile and hug them. They are
> proud of them. They say the cloud will always rise and make the sky blue and
> pretty again.

The continuation is readable, but it never explains Rayleigh scattering and is
factually unsuitable as an answer. It converts an explanatory prompt into a
TinyStories-style narrative.

### Dialogue continuation — initially relevant, then loses context

**Prompt (seed 44)**

> Maya said, "I am nervous about my presentation tomorrow."
> Her friend replied, "

**Model output**

> Don't worry, I will help you find it!"
>
> The next day, Lily and her friend saw the present again. This time, Lily
> remembered what her mom said about the present and was happy that her friend
> was there.
>
> The moral of the story is that it's important to help others when they need
> it. Even if you are nervous, you can always count on someone who can help you.

The first phrase has a supportive conversational tone, but “presentation” is
misread as a missing “present,” Maya becomes Lily, and the model falls back to a
moralized story. This does not demonstrate sustained conversational ability.

## Methodology and validation

### Language modeling

The WikiText evaluator tokenizes the entire raw test export with the tokenizer
saved in the checkpoint. It uses overlapping 1,024-token windows with stride
512. Targets already scored in an earlier window are masked, so every next token
is counted once. Losses are summed over tokens before calculating mean negative
log-likelihood and perplexity.

The held-out SFT metric instead reconstructs the saved Alpaca split and prompt
template, masks every prompt target, and averages cross-entropy only over
response and EOS targets. It evaluates every validation batch; no weights or
optimizer state are changed.

### Multiple-choice reasoning

Each MMLU item is formatted with its question, four labeled choices, and the
suffix `Answer:`. Completion mode scores the continuations (` A`, ` B`, ` C`,
` D`). SFT instruction mode wraps that task in the saved Alpaca template and
scores response continuations (`A`, `B`, `C`, `D`) without a leading space,
matching how responses were separately tokenized during tuning. All labels are
one token under the shared tokenizer. Accuracy is exact match against the
provided answer index.

Eleven completion prompts and twelve instruction prompts exceeded 1,024 tokens
and were left-truncated by the scoring function, retaining the final context
that contains the choices and answer suffix. This affects at most 0.09% of MMLU
and cannot explain the aggregate results.

### Batching and result integrity

Right-padding was used only to combine inference examples. Because the model is
causal and scores each prompt's last real token, padding placed after that token
cannot influence its logits. Batched and single-example checks agreed:

- WikiText 128-token check: average-NLL difference
  `4.81e-7` between batch size 1 and batch size 4.
- First five MMLU questions: identical predicted labels; maximum answer-score
  difference `1.24e-5` between single-example and batched inference.
- The MMLU JSONL contains exactly 14,042 sequentially indexed predictions, and
  recomputing correctness from those rows reproduces 3,548 correct answers.
- The SFT 128-token WikiText CPU/MPS average-NLL difference is `1.44e-6`.
- The first five instruction-mode SFT MMLU questions have identical CPU/MPS
  predictions; maximum answer-score difference is `7.15e-6`.
- Both SFT MMLU JSONL files contain 14,042 sequential indices; recomputation
  reproduces 3,342 instruction-mode and 3,211 completion-mode correct answers.
- The complete project suite passes: 38 tests and 4 subtests.

## Limitations and recommended next steps

1. **The pretraining checkpoint is specialized to TinyStories.** Alpaca SFT
   improves WikiText likelihood substantially, but perplexity 183.63 is still
   not evidence of strong general-domain language modeling.
2. **MMLU is zero-shot and strongly label-biased.** Repeat with the official
   subject-matched development examples (for example, five-shot prompting) and
   examine whether demonstrations calibrate the answer-label distributions in
   both completion and instruction modes.
3. **Held-out response loss is teacher-forced.** It measures probability on
   reference Alpaca responses, not whether free-form answers are correct,
   helpful, harmless, or preferred by users.
4. **Qualitative sampling is deliberately small.** Preserve fixed prompts and
   seeds, but expand to multiple samples per prompt and score factuality,
   coherence, repetition, and instruction adherence with a written rubric.
5. **Perplexity is tokenizer-dependent.** Compare checkpoints only when the
   dataset, tokenizer, normalization, context, and stride are held constant.

## Further evaluation questions

- Does subject-matched five-shot prompting reduce the pretraining C bias and the
  SFT A/C biases, or is explicit answer-label calibration required?
- Does evaluation on held-out instruction benchmarks (rather than Alpaca
  teacher-forced loss) confirm any general instruction-following improvement?
- Which additions to pretraining data preserve the SFT WikiText gain while also
  improving factuality and MMLU beyond chance?

## Reproduction commands

Run from the repository root in the `ai4s-gpt` environment with
`PYTHONPATH=src`. The completed SFT evaluations used:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/gpt-sft-bpe/best.pt --device mps

python scripts/evaluate_wikitext.py \
  --checkpoint outputs/gpt-sft-bpe/best.pt --device mps \
  --batch-size 4 --output outputs/evaluation/sft-wikitext.json

python scripts/evaluate_mmlu.py \
  --checkpoint outputs/gpt-sft-bpe/best.pt --device mps --batch-size 8 \
  --output outputs/evaluation/sft-mmlu-instruction.json \
  --predictions-output outputs/evaluation/sft-mmlu-instruction-predictions.jsonl

python scripts/evaluate_mmlu.py \
  --checkpoint outputs/gpt-sft-bpe/best.pt --device mps --batch-size 8 \
  --prompt-mode completion \
  --output outputs/evaluation/sft-mmlu-completion.json \
  --predictions-output outputs/evaluation/sft-mmlu-completion-predictions.jsonl

python scripts/generate_samples.py \
  --checkpoint outputs/gpt-sft-bpe/best.pt --device mps \
  --output outputs/evaluation/sft-generation-samples.md \
  --json-output outputs/evaluation/sft-generation-samples.json
```

## Reproducibility artifacts

- WikiText aggregate report:
  [`outputs/evaluation/wikitext.json`](outputs/evaluation/wikitext.json)
- MMLU aggregate and per-subject report:
  [`outputs/evaluation/mmlu.json`](outputs/evaluation/mmlu.json)
- MMLU per-question predictions and answer log-likelihoods:
  [`outputs/evaluation/mmlu-predictions.jsonl`](outputs/evaluation/mmlu-predictions.jsonl)
- Qualitative Markdown report:
  [`outputs/evaluation/generation_samples.md`](outputs/evaluation/generation_samples.md)
- Qualitative machine-readable report:
  [`outputs/evaluation/generation_samples.json`](outputs/evaluation/generation_samples.json)
- SFT WikiText aggregate report:
  [`outputs/evaluation/sft-wikitext.json`](outputs/evaluation/sft-wikitext.json)
- SFT native instruction-mode MMLU aggregate and predictions:
  [`outputs/evaluation/sft-mmlu-instruction.json`](outputs/evaluation/sft-mmlu-instruction.json),
  [`outputs/evaluation/sft-mmlu-instruction-predictions.jsonl`](outputs/evaluation/sft-mmlu-instruction-predictions.jsonl)
- SFT completion-mode MMLU aggregate and predictions:
  [`outputs/evaluation/sft-mmlu-completion.json`](outputs/evaluation/sft-mmlu-completion.json),
  [`outputs/evaluation/sft-mmlu-completion-predictions.jsonl`](outputs/evaluation/sft-mmlu-completion-predictions.jsonl)
- SFT qualitative Markdown and machine-readable reports:
  [`outputs/evaluation/sft-generation-samples.md`](outputs/evaluation/sft-generation-samples.md),
  [`outputs/evaluation/sft-generation-samples.json`](outputs/evaluation/sft-generation-samples.json)
- Evaluation implementation:
  [`src/mygpt/evaluation.py`](src/mygpt/evaluation.py),
  [`scripts/evaluate_wikitext.py`](scripts/evaluate_wikitext.py),
  [`scripts/evaluate_mmlu.py`](scripts/evaluate_mmlu.py), and
  [`scripts/generate_samples.py`](scripts/generate_samples.py)
