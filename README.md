# CrossHOI-Bench: rescoring the HOI-specialist baselines

This repo re-runs the HOI-specialist scoring of **CrossHOI-Bench** on CPU, using the authors' code and the predictions they released. It reproduces the paper's Setting 1 and Setting 3 tables, then tests two details of the specialist scoring that change the Setting 1 comparison.

- **Paper:** Lei, Wang, Tan, *CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods*, CVPR 2026. [arXiv 2508.18753](https://arxiv.org/abs/2508.18753)
- **Code:** [ChelsieLei/CrossHOI-Bench](https://github.com/ChelsieLei/CrossHOI-Bench), checked at commit `abc793f` (19 May 2026)
- **Predictions:** [chelsielei/CrossHOI_Bench on Hugging Face](https://huggingface.co/chelsielei/CrossHOI_Bench/tree/main/HOI_pred_hicodet)

This is an independent reproduction note. The benchmark itself, its construction and its VLM results are not in question here. Only the step that turns a specialist model's ranked predictions into a multiple-choice answer is examined.

## 1. Reproduction

`rescore.py` ports `script_eval_32B_newbench_HOI.py` and `newbench_question_func.py` to NumPy. It runs with the paper's settings (`--pred_thres 5`, `--pred_select question_rank`, from `scripts/script_instruct_newbench_eval_HOI.sh`).

Setting 1:

| Model | Instance-F1, paper / here | Micro-F1, paper / here | EM, paper / here | Macro-F1, paper / here |
|---|---|---|---|---|
| ADA-CM | 47.76 / 47.74 | 61.69 / 61.68 | 19.15 / 19.07 | 43.02 / 41.53 |
| CMMP | 46.62 / 46.62 | 60.85 / 60.83 | 18.84 / 18.84 | 43.06 / 41.57 |
| LAIN | 45.64 / 45.66 | 59.09 / 59.13 | 19.31 / 19.39 | 41.28 / 40.04 |
| HOLa | 47.12 / 47.17 | 61.29 / 61.32 | 19.78 / 19.78 | 43.61 / 42.28 |
| CMD-SE | 44.66 / 44.63 | 58.71 / 58.70 | 20.33 / 20.25 | 47.49 / 39.94 |

Notes:
- **Setting 3** reproduces just as closely: Instance-F1, Micro-F1 and EM are exact for four models and within 0.06 for CMD-SE.
- **Macro-F1** comes out about 1.3–1.5 points below Table 1. However, ADA-CM's 41.53 equals the paper's own Table 11 for the same configuration.
- **CMD-SE's Macro-F1** is the one figure that does not reproduce (see section 5).

## 2. Issue 1: "no interaction" predictions can never match an option

The two sides spell the class differently:
- Specialist prediction files: `"no_interaction a/an bicycle"`
- Benchmark options: `"no_interaction with a/an giraffe"`

The scoring keeps a prediction only if its label is exactly one of the four option strings (`script_eval_32B_newbench_HOI.py`, line 241). So a specialist can never give a "no interaction" answer. VLMs are unaffected because they answer with option letters.

| | Setting 1 | Setting 3 |
|---|---|---|
| Questions where "no interaction" is a correct answer | 136 / 1,274 | 206 / 1,274 |
| Questions where it is one of the four options | 441 | 354 |
| ADA-CM Instance-F1 on those 136 questions: as published → labels normalised | 15.4 → 42.2 | n/a |

## 3. Issue 2: the top-5 cut-off is computed per image, not per question

Section 4.1 of the paper says the top-5 predictions are taken for each question. In the code, the cut-off is the 5th-highest score over **all** human–object pairs in the image (line 236, `ho_scores.topk(...)`). Only after that are the pairs restricted to the target person (lines 217–226).

In multi-person images, other people's pairs can fill the five slots. In Setting 1, between 126 and 182 questions per model have a target person with a valid on-option prediction who still ends up with no answer.

It's possible this is the intended protocol and the text is just loosely worded. If so, the text is what needs changing.

## 4. Effect on the Setting 1 comparison

Both changes are applied with k kept at 5 and nothing tuned. The VLM row is from the paper. The 95% intervals are question-level bootstraps (1,000 resamples), and the VLM's interval comes from the paper's Table 7.

| Model | Macro-F1 | Instance-F1 | Micro-F1 | EM |
|---|---|---|---|---|
| Qwen2.5-VL-32B (paper) | 50.71 | 52.94 [50.8, 55.1] | 61.41 | 26.06 |
| ADA-CM, as published | 41.53 | 47.74 | 61.68 | 19.07 |
| ADA-CM, both changes | 50.32 | 54.35 [52.2, 56.4] | 65.29 | 15.70 |
| HOLa, both changes | 50.28 | 53.86 [51.7, 56.1] | 64.53 | 16.17 |
| CMMP, both changes | 50.40 | 53.72 [51.6, 55.9] | 64.42 | 15.78 |

What the table shows:
- **Macro-F1** is level.
- **Instance-F1 and Micro-F1** are slightly higher for the specialists, with overlapping intervals.
- **Exact match** still clearly favours the VLM, because five guesses per person over-answer questions that average 1.83 correct options.

In **Setting 3**, only issue 1 applies. It moves ADA-CM from 45.80 / 56.23 to 51.20 / 59.29 (Macro / Instance-F1), against the VLM's 63.16 / 67.19, so the VLM's lead there remains.

More variants are in `results_experiments.txt`:
- ranking only within the four options;
- sweeping k;
- k set to the true answer count.

The last one leaks the answer count, and the option-only variants were explored after seeing results. They show how sensitive the comparison is to the selection rule; they are not proposed as fair comparisons.

## 5. Smaller observations
- **CMD-SE Macro-F1** comes out at 39.94 here, against 47.49 in Tables 1 and 11. Its other three metrics reproduce, so the selection matches and the difference is in the per-class averaging. Also, the released CMD-SE file is missing 92 of the 1,274 benchmark images.
- **ADA-CM (top-5, Setting 1)** has Macro-F1 43.02 in Table 1 and 41.53 in Table 11.
- **Duplicate labels:** the de-duplication check at line 240 is commented out. Duplicate *wrong* labels are each counted as a false positive, while duplicate *correct* labels are ignored.

## Limitations
- Only the specialist side was rescored. The VLM outputs were not re-examined.
- Issue 2 rests on reading "for each question" literally.

## Run it

```bash
git clone https://github.com/ChelsieLei/CrossHOI-Bench repo
pip install -r requirements.txt
python fetch_predictions.py      # 5 HICO-DET prediction files, ~270 MB, into ./predictions
python rescore.py                # reproduces Tables 1 and 3 (about 30 s on CPU)
python experiments.py            # issue counts, variants, k sweeps, bootstrap CIs
```

`rescore.py` imports the label tables from the cloned repo (`repo/hico_text_label.py`) and reads its benchmark files (`repo/hicodet/our_test/`). Nothing from that repo or the predictions is redistributed here. The two switches that implement the changes are `fix_noint=True` and `rank_within="target"` in `select()`.

Code: MIT. Questions and corrections are welcome as issues.
