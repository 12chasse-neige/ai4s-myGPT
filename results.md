# myGPT Evaluation Results

## Technical summary

The evaluated checkpoint demonstrates locally coherent English story generation,
but it does not yet show reliable out-of-domain language modeling, factual
explanation, conversation, or multiple-choice reasoning.

- **WikiText-2:** full-test BPE-token perplexity is **929.84**
  (average negative log-likelihood **6.8350**, **9.8608 bits/token**) over
  442,868 next-token predictions. This is far worse than the checkpoint's
  TinyStories validation loss, showing a substantial domain-generalization gap.
- **MMLU:** zero-shot micro accuracy is **25.27%** (**3,548/14,042**) and
  macro subject accuracy is **24.98%** across all 57 subjects. The 95% Wilson
  interval for micro accuracy is **24.56%–25.99%**, which contains the uniform
  four-choice chance baseline of 25%.
- **MMLU answer bias:** the model selected answer **C for 92.39%** of questions,
  even though gold answers are approximately balanced. The MMLU result therefore
  does not constitute evidence of reasoning ability.
- **Qualitative generation:** the story sample is grammatical and maintains a
  simple narrative, but it is repetitive. The factual explanation is incorrect,
  and the dialogue loses the original speaker and topic. These samples support a
  limited claim of language-generation ability, not conversational competence.

## Evaluation scope

| Item | Evaluated value |
|---|---:|
| Checkpoint | `outputs/gpt-pretrain-bpe/best.pt` |
| Training stage | Pretraining |
| Checkpoint step | 75,000 |
| Saved best TinyStories validation loss | 1.1276 |
| Parameters | 66,053,760 |
| Transformer shape | 12 layers, 10 heads, embedding width 640 |
| Context length | 1,024 BPE tokens |
| Vocabulary | 10,000 byte-level BPE tokens |
| Evaluation device | CPU |

The evaluation used the complete locally prepared WikiText-2 raw test split and
the complete MMLU test split. MMLU was evaluated zero-shot because no development
split was supplied for few-shot demonstrations. The qualitative assessment used
three fixed prompts with one deterministic sample per prompt.

| Evaluation | Scope | Denominator |
|---|---|---:|
| WikiText-2 raw test | Complete test text | 442,869 tokens; 442,868 predictions |
| MMLU test | All 57 subjects, zero-shot | 14,042 questions |
| Qualitative generation | Story, explanation, dialogue | 3 prompts |

## WikiText shows a large out-of-domain language-modeling gap

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

## MMLU performance is at chance and dominated by answer-label bias

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

## Qualitative samples show fluency but weak task adherence

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

### Multiple-choice reasoning

Each MMLU item is formatted with its question, four labeled choices, and the
suffix `Answer:`. For each label continuation (` A`, ` B`, ` C`, ` D`), the
evaluator calculates conditional log-likelihood and selects the maximum. All
four labels are one token under this checkpoint's tokenizer. This is a zero-shot
evaluation; accuracy is exact match against the provided answer index.

Eleven unusually long prompts exceeded 1,024 tokens and were left-truncated by
the scoring function, retaining the final context that contains the choices and
answer suffix. This affects 0.08% of MMLU and cannot explain the overall result.

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
- The complete project suite passes: 37 tests and 4 subtests.

## Limitations and recommended next steps

1. **The checkpoint is specialized to TinyStories.** Broader pretraining data is
   needed before WikiText can be used as evidence of general language modeling.
2. **MMLU is zero-shot and strongly label-biased.** Repeat with the official
   subject-matched development examples (for example, five-shot prompting) and
   examine whether demonstrations calibrate the answer-label distribution.
3. **The checkpoint is not instruction-tuned.** Evaluate factual explanation and
   conversation again after SFT, using the instruction prompt template rather
   than treating completion as dialogue competence.
4. **Qualitative sampling is deliberately small.** Preserve fixed prompts and
   seeds, but expand to multiple samples per prompt and score factuality,
   coherence, repetition, and instruction adherence with a written rubric.
5. **Perplexity is tokenizer-dependent.** Compare checkpoints only when the
   dataset, tokenizer, normalization, context, and stride are held constant.

## Further evaluation questions

- Does subject-matched five-shot prompting reduce the 92.4% preference for C,
  or is answer-label calibration still required?
- Does instruction tuning improve factual and conversational prompt adherence
  without degrading the pretrained model's story-generation fluency?
- How much broader pretraining data is required before WikiText perplexity and
  MMLU accuracy improve together rather than only within the TinyStories domain?

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
- Evaluation implementation:
  [`src/mygpt/evaluation.py`](src/mygpt/evaluation.py),
  [`scripts/evaluate_wikitext.py`](scripts/evaluate_wikitext.py),
  [`scripts/evaluate_mmlu.py`](scripts/evaluate_mmlu.py), and
  [`scripts/generate_samples.py`](scripts/generate_samples.py)
