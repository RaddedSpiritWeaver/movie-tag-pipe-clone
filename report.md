# Movie-tag prediction: model comparison report

Ground truth: 983 unique annotated clips, 91 valid tags (`data/simple_labels.txt`). Ablation sample: 98 clips (`data/ablation_sample_ids.txt`, fixed seed, identical across all models).

**Methodology note:** Precision/recall/F1 are micro-averaged (pooled TP/FP/FN across all clips and tags, not averaged per-tag) - this is the standard choice for multi-label problems with many rare tags, where per-tag macro-averaging is dominated by noise on tags that appear only a handful of times. Accuracy is per-label accuracy (a tag counts as "correct" for a clip if its predicted presence/absence matches ground truth), computed across all clip x tag pairs - this is dominated by true negatives (most tags don't apply to most clips), so high accuracy numbers here are expected and not by themselves evidence of a good model; read them alongside F1.

Three thresholding schemes convert continuous 0-1 tag-likelihood scores into a predicted tag set: **max** takes only the single highest-scoring tag per clip (top-1); **0.7** and **0.3** take every tag scoring at or above that cutoff, regardless of count.

## Model comparison (neutral prompt)

Head-to-head comparison of all five models on the full dataset, neutral prompt (`instructions.md`), one table per threshold. Best F1 per threshold is **bolded**.

**Threshold: max (top-1)**

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| DeepSeek V3.2 | 0.540 | 0.097 | **0.165** | 0.940 |
| Grok 4.1 Fast Reasoning | 0.537 | 0.096 | 0.163 | 0.940 |
| GPT-OSS 120B | 0.402 | 0.072 | 0.122 | 0.937 |
| Gemini 2.5 Flash | 0.395 | 0.071 | 0.120 | 0.937 |
| Qwen3-Next-80B Instruct | 0.378 | 0.068 | 0.115 | 0.936 |

**Threshold: 0.7**

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Grok 4.1 Fast Reasoning | 0.336 | 0.499 | **0.401** | 0.909 |
| DeepSeek V3.2 | 0.364 | 0.408 | 0.385 | 0.920 |
| Gemini 2.5 Flash | 0.278 | 0.496 | 0.357 | 0.891 |
| GPT-OSS 120B | 0.320 | 0.322 | 0.321 | 0.917 |
| Qwen3-Next-80B Instruct | 0.194 | 0.621 | 0.296 | 0.819 |

**Threshold: 0.3**

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| DeepSeek V3.2 | 0.240 | 0.629 | **0.348** | 0.856 |
| Grok 4.1 Fast Reasoning | 0.210 | 0.723 | 0.325 | 0.817 |
| GPT-OSS 120B | 0.203 | 0.581 | 0.301 | 0.835 |
| Gemini 2.5 Flash | 0.184 | 0.710 | 0.292 | 0.789 |
| Qwen3-Next-80B Instruct | 0.128 | 0.848 | 0.222 | 0.636 |

## Models

### Gemini 2.5 Flash

Google's mid-tier Gemini model on Vertex AI, balancing quality, latency, and cost. Called via the native Gemini API with `response_json_schema` structured output, so every call is schema-constrained at the API level rather than relying on prompted JSON formatting.

Coverage: 983/983 clips scored.

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| max (top-1) | 0.395 | 0.071 | 0.120 | 0.937 |
| 0.7 | 0.278 | 0.496 | 0.357 | 0.891 |
| 0.3 | 0.184 | 0.710 | 0.292 | 0.789 |

**Movie-expert ablation** (same 98-clip sample, neutral prompt vs. movie-expert prompt):

- Mean absolute score change: **0.071** (on a 0-1 scale)
- Mean signed score change: **-0.001** - scores run lower on average under the movie-expert prompt
- Correlation between a tag's neutral-prompt score and its change under the movie-expert prompt: **-0.276** (a negative correlation - already-high scores get pulled down and already-low scores get pushed up under the movie-expert prompt (scores compress toward the middle))

### Grok 4.1 Fast Reasoning

xAI's fast, reasoning-tuned model, accessed through Vertex AI's OpenAI-compatible endpoint. One clip in the main run triggered Grok's own content-safety filter and returned no output at all (a deterministic refusal, not a transient error) - it's excluded from this model's metrics rather than counted as a miss, so Grok's coverage is 982/983 clips instead of 983/983.

Coverage: 982/983 clips scored.

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| max (top-1) | 0.537 | 0.096 | 0.163 | 0.940 |
| 0.7 | 0.336 | 0.499 | 0.401 | 0.909 |
| 0.3 | 0.210 | 0.723 | 0.325 | 0.817 |

**Movie-expert ablation** (same 98-clip sample, neutral prompt vs. movie-expert prompt):

- Mean absolute score change: **0.053** (on a 0-1 scale)
- Mean signed score change: **+0.012** - scores run higher on average under the movie-expert prompt
- Correlation between a tag's neutral-prompt score and its change under the movie-expert prompt: **-0.128** (close to no correlation - the shift is roughly independent of the tag's neutral-prompt score)

### GPT-OSS 120B

OpenAI's open-weight 120B model, hosted directly on Vertex AI as a managed open model (MaaS) rather than through OpenAI's own API - billed through GCP, no OpenAI account needed. Called via the same OpenAI-compatible endpoint as Grok.

Coverage: 983/983 clips scored.

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| max (top-1) | 0.402 | 0.072 | 0.122 | 0.937 |
| 0.7 | 0.320 | 0.322 | 0.321 | 0.917 |
| 0.3 | 0.203 | 0.581 | 0.301 | 0.835 |

**Movie-expert ablation** (same 98-clip sample, neutral prompt vs. movie-expert prompt):

- Mean absolute score change: **0.049** (on a 0-1 scale)
- Mean signed score change: **+0.009** - scores run higher on average under the movie-expert prompt
- Correlation between a tag's neutral-prompt score and its change under the movie-expert prompt: **-0.194** (a negative correlation - already-high scores get pulled down and already-low scores get pushed up under the movie-expert prompt (scores compress toward the middle))

### DeepSeek V3.2

DeepSeek's large open-weight model, hosted on Vertex AI as a managed open model, reachable only via the endpoint's `global` location rather than a specific region.

Coverage: 983/983 clips scored.

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| max (top-1) | 0.540 | 0.097 | 0.165 | 0.940 |
| 0.7 | 0.364 | 0.408 | 0.385 | 0.920 |
| 0.3 | 0.240 | 0.629 | 0.348 | 0.856 |

**Movie-expert ablation** (same 98-clip sample, neutral prompt vs. movie-expert prompt):

- Mean absolute score change: **0.041** (on a 0-1 scale)
- Mean signed score change: **+0.009** - scores run higher on average under the movie-expert prompt
- Correlation between a tag's neutral-prompt score and its change under the movie-expert prompt: **-0.127** (close to no correlation - the shift is roughly independent of the tag's neutral-prompt score)

### Qwen3-Next-80B Instruct

Alibaba's Qwen3-Next 80B model, standard instruction-tuned (non-reasoning) variant. The sibling "Thinking" (reasoning) variant was tried first but got stuck in a long repetitive reasoning loop and never produced output even on a trivial prompt, so it was dropped in favor of this Instruct variant.

Coverage: 983/983 clips scored.

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| max (top-1) | 0.378 | 0.068 | 0.115 | 0.936 |
| 0.7 | 0.194 | 0.621 | 0.296 | 0.819 |
| 0.3 | 0.128 | 0.848 | 0.222 | 0.636 |

**Movie-expert ablation** (same 98-clip sample, neutral prompt vs. movie-expert prompt):

- Mean absolute score change: **0.086** (on a 0-1 scale)
- Mean signed score change: **-0.009** - scores run lower on average under the movie-expert prompt
- Correlation between a tag's neutral-prompt score and its change under the movie-expert prompt: **-0.144** (close to no correlation - the shift is roughly independent of the tag's neutral-prompt score)

## Key takeaways

**Movie-title identification rate** (out of the same 98-clip ablation sample):

| Model | Clips with a guessed title |
|---|---|
| Gemini 2.5 Flash | 77/98 (79%) |
| Grok 4.1 Fast Reasoning | 44/98 (45%) |
| GPT-OSS 120B | 6/98 (6%) |
| DeepSeek V3.2 | 30/98 (31%) |
| Qwen3-Next-80B Instruct | 63/98 (64%) |

**Cross-model agreement** is used as a correctness proxy in place of external lookups: if independent models converge on the same title for the same clip, that's stronger evidence of a real, recognizable movie than any single confident-sounding guess (a model guessing fluently doesn't mean it guessed *correctly* - we have no ground-truth titles to check against directly, and we did not perform external lookups to verify these guesses).

![Cross-model title agreement histogram](results/report_assets/title_agreement_histogram.png)

Clips where 2 or more models agreed on the same title:

| Clip id | Agreed title | # models agreeing | Models |
|---|---|---|---|
| 4gSkh86zcaU | American Made | 5 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, GPT-OSS 120B, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| MBSxl8y36zg | Concussion | 5 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, GPT-OSS 120B, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| XyEHIOZf5yE | Mutiny on the Bounty | 5 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, GPT-OSS 120B, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| tG1crFI87ro | Hancock | 5 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, GPT-OSS 120B, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| Do7U7AkA5jA | Hoosiers | 4 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| MkKO-Z_Sjm4 | 300: Rise of an Empire | 4 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| oR3h33DSGPM | Roman J. Israel, Esq. | 4 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| pBd7XYjjvRw | Darkest Hour | 4 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| ryyEEyKD9EU | Concussion | 4 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| -JNyHnAi8zk | The Karate Kid | 3 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, Qwen3-Next-80B Instruct |
| AALREbJZEZk | The Best Little Whorehouse in Texas | 3 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2 |
| C0vM89y4088 | Mighty Morphin Power Rangers | 3 | Grok 4.1 Fast Reasoning, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| E-gwIn1O1pE | Neruda | 3 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, Qwen3-Next-80B Instruct |
| OJNBsuHFdM4 | Spider-Man 3 | 3 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2 |
| P-KZ7N30lFY | Stuart Little | 3 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, Qwen3-Next-80B Instruct |
| YEdNAX9h_vI | Stuart Little | 3 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2 |
| e48T01eVXtU | Shrek | 3 | Gemini 2.5 Flash, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| f2FzrfnfQPY | Zack Snyder's Justice League | 3 | Grok 4.1 Fast Reasoning, DeepSeek V3.2, Qwen3-Next-80B Instruct |
| oNoOFf527GU | Mutiny on the Bounty | 3 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2 |
| yodVQ5QAc88 | Man of Steel | 3 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning, DeepSeek V3.2 |
| -fL94BTrFhs | 500 Days of Summer | 2 | Gemini 2.5 Flash, DeepSeek V3.2 |
| 8aK7LsC0G-4 | The Big Short | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| DHWbxxpKb04 | Cloudy with a Chance of Meatballs 2 | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| EMKD4L6Peto | Supergirl | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| KjWPSq6-Ets | The Boss Baby: Family Business | 2 | Gemini 2.5 Flash, Qwen3-Next-80B Instruct |
| NR-BaVehxrA | Barbershop: The Next Cut | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| SCiMcDcoi3E | Get Shorty | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| UeeK-Fgup9w | The Dark Knight | 2 | Gemini 2.5 Flash, DeepSeek V3.2 |
| ViF6HrzoOj4 | Interstellar | 2 | Gemini 2.5 Flash, DeepSeek V3.2 |
| WBreuW9LLSw | American Made | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| WYY9Epog0rs | Annie Hall | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| cyUf4tLI53o | Yours, Mine and Ours | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| f1bk5a_jaEA | Willy Wonka & the Chocolate Factory | 2 | Grok 4.1 Fast Reasoning, Qwen3-Next-80B Instruct |
| hzmZJcPlJlE | Supergirl | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| ilCjx9gigWI | Stuart Little 2 | 2 | Gemini 2.5 Flash, Grok 4.1 Fast Reasoning |
| q292IDwEWZ0 | Spider-Man | 2 | Gemini 2.5 Flash, Qwen3-Next-80B Instruct |
| sTbJQwezQZY | RoboCop | 2 | Gemini 2.5 Flash, Qwen3-Next-80B Instruct |
| yijQXDtxgic | Despicable Me | 2 | Grok 4.1 Fast Reasoning, Qwen3-Next-80B Instruct |
