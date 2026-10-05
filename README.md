# CrossHOI-Bench: rescoring the HOI-specialist baselines

This repo re-runs the HOI-specialist scoring of **CrossHOI-Bench** on CPU, from the authors' released evaluation code and predictions. It then checks how sensitive the results are to three details of the step that turns a specialist's ranked detections into a multiple-choice answer.

- **Paper:** Lei, Wang, Tan, *CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods*, CVPR 2026. [arXiv 2508.18753](https://arxiv.org/abs/2508.18753) (v4 checked)
- **Code:** [ChelsieLei/CrossHOI-Bench](https://github.com/ChelsieLei/CrossHOI-Bench) at commit `abc793f` (19 May 2026)
- **Predictions:** [chelsielei/CrossHOI_Bench on Hugging Face](https://huggingface.co/chelsielei/CrossHOI_Bench/tree/main/HOI_pred_hicodet)

**Scope.** This is an independent reproduction note about the specialist-side scoring of the released predictions. It does not re-run any model, re-examine the VLM outputs, or address the benchmark's construction or its other settings and datasets.

**Settings, as the paper defines them (Sec. 3.2):**
- **Setting 1:** the model detects everyone itself. The ground-truth target-person box is used **only at evaluation**, to decide which detected person's answers are scored (IoU > 0.5).
- **Setting 3:** all interactions in the image are scored.

## 1. Reproduction (partial)

`rescore.py` ports `script_eval_32B_newbench_HOI.py` and `newbench_question_func.py` to NumPy. It runs with the paper's settings (`--pred_thres 5`, `--pred_select question_rank`, from `scripts/script_instruct_newbench_eval_HOI.sh`). Instance-F1, Micro-F1 and exact match (EM) agree with the paper to within 0.1 points for all five specialists in both settings. **Macro-F1 does not fully reproduce.**

Setting 1, paper value / this repo:

| Model | Instance-F1 | Micro-F1 | EM | Macro-F1 |
|---|---|---|---|---|
| ADA-CM | 47.76 / 47.74 | 61.69 / 61.68 | 19.15 / 19.07 | 43.02 / 41.53 |
| CMMP | 46.62 / 46.62 | 60.85 / 60.83 | 18.84 / 18.84 | 43.06 / 41.57 |
| LAIN | 45.64 / 45.66 | 59.09 / 59.13 | 19.31 / 19.39 | 41.28 / 40.04 |
| HOLa | 47.12 / 47.17 | 61.29 / 61.32 | 19.78 / 19.78 | 43.61 / 42.28 |
| CMD-SE | 44.66 / 44.63 | 58.71 / 58.70 | 20.33 / 20.25 | 47.49 / 39.94 |

On Macro-F1:
- **Setting 1:** every value differs from Table 1. ADA-CM's 41.53 does equal the paper's Table 11 for the same configuration.
- **Setting 3:** four models are within about 0.1 points; CMD-SE gives 43.97 against 46.51.

The cause of these differences is unknown. Close agreement on the other three metrics supports the port, but it does not prove the released inputs are the ones behind the paper's tables.

## 2. "No interaction" predictions can never match an option

The two sides spell the class differently:
- Specialist prediction files: `"no_interaction a/an bicycle"`
- Benchmark options: `"no_interaction with a/an giraffe"`

The scorer keeps a prediction only if its label is exactly one of the four option strings (`script_eval_32B_newbench_HOI.py`, line 241). So specialists never give a "no interaction" answer. VLMs are unaffected, because they answer with option letters. Normalising this one spelling changes nothing else: same boxes, same scores.

| | Setting 1 | Setting 3 |
|---|---|---|
| Questions where "no interaction" is a correct answer | 136 / 1,274 | 206 / 1,274 |
| Questions where it is one of the four options | 441 | 354 |
| ADA-CM Macro / Instance-F1: released → spelling normalised | 41.53 / 47.74 → 46.66 / 50.15 | 45.80 / 56.23 → 51.20 / 59.29 |

On the 136 Setting 1 questions, ADA-CM's Instance-F1 goes from 15.4 to 42.2. The fix need not help every model: CMD-SE's Setting 1 Instance-F1 dips slightly, because restoring the label also admits some wrong "no interaction" answers.

## 3. Which predictions the top-5 cut-off ranks (a protocol question)

Section 4.1 of the paper says the top-5 predictions are taken for each question. The released scorer works in four steps:
1. It forms the target-matched candidates (lines 217–226).
2. It sets the cut-off at the 5th-highest score over **all** human–object pairs in the image (line 236).
3. It applies that cut-off to the target-matched candidates.
4. It keeps only labels that are among the options.

In Setting 1, between 126 and 182 questions per model have a target-matched prediction that is among the options, yet the question ends up with no answer:

| Model | Empty, despite a target-matched on-option prediction | Of these, a target-matched *correct* prediction exists |
|---|---|---|
| ADA-CM | 126 | 103 |
| CMMP | 143 | 123 |
| LAIN | 157 | 140 |
| HOLa | 133 | 116 |
| CMD-SE | 182 | 151 |

These counts show the result is sensitive to the selection rule. They do not mean every such question would be answered correctly. The image-wide cut-off may be intended, for example to make detections compete across the whole image. Below, **ranking within the target-matched candidates** is treated as an alternative protocol, not as a correction.

## 4. Setting 1 under the alternative protocol

This applies the spelling normalisation plus target-matched top-5, with k kept at 5 and nothing tuned. The Qwen row is copied from the paper's Table 1; it was not rescored here. Intervals are 95% question-level bootstraps (1,000 resamples, `experiments.py` section 4).

| Model | Macro-F1 | Instance-F1 | Micro-F1 | EM |
|---|---|---|---|---|
| Qwen2.5-VL-32B (paper) | 50.71 | 52.94 | 61.41 | 26.06 |
| ADA-CM, released selection | 41.53 | 47.74 | 61.68 | 19.07 |
| ADA-CM, spelling only | 46.66 | 50.15 | 63.10 | 19.39 |
| ADA-CM, target ranking only | 45.25 | 52.07 | 64.33 | 15.70 |
| ADA-CM, both | 50.32 | 54.35 [52.16, 56.41] | 65.29 | 15.70 |
| HOLa, both | 50.28 | 53.86 [51.74, 56.08] | 64.53 | 16.17 |
| CMMP, both | 50.40 | 53.72 [51.63, 55.87] | 64.42 | 15.78 |

For ADA-CM, the paired change from the released selection to "both" is:
- Instance-F1: +6.62 points [+5.10, +8.03]
- exact match: −3.38 points [−5.26, −1.41]

So the protocol choice moves the specialists' Setting 1 scores materially, with a trade-off between F1 and exact match. The specialists' point estimates end up close to Qwen's on Macro-F1, and above it on Instance-F1 and Micro-F1. Qwen stays clearly ahead on exact match.

The VLM's per-question outputs aren't available here, so no paired comparison with the VLM is possible. Nothing here claims the models are statistically equal, or that either is better.

In **Setting 3**, only the spelling issue applies. Qwen keeps higher published Macro-F1, Instance-F1 and EM; ADA-CM's normalised Micro-F1 is slightly higher.

## 5. Other observations

- **Duplicate labels.** The paper describes set-based metrics, but answer lists are not de-duplicated (the check at line 240 is commented out). The effects:
  - a repeated correct label adds nothing;
  - each repeated wrong label adds a false positive;
  - any repeat makes exact match fail, because exact match compares list lengths.

  De-duplicating ADA-CM's answers in Setting 1 changes exact match from 19.39 to 25.12 with the spelling fix only, and from 15.70 to 24.41 with both changes. F1 barely moves. See `experiments.py` section 5.
- **CMD-SE coverage.** The released CMD-SE file lacks 92 of the 1,274 benchmark images, and 91 of those 92 have "no interaction" as a correct answer. The scorer computes F1 over the available images but divides exact matches by all 1,274. Comparisons involving CMD-SE should be read with that in mind.
- **ADA-CM Macro-F1 for top-5, Setting 1** is 43.02 in Table 1 and 41.53 in Table 11.

## 6. Exploratory variants (`results_experiments.txt`)

These variants were explored after seeing results:
- ranking only among the four options;
- sweeping k;
- setting k to the true number of answers.

The last uses privileged information (the answer count). They are sensitivity checks only, not fair comparisons, not upper bounds, and not a breakdown of where the gap comes from. The paper already discusses the F1 versus exact-match trade-off across k (Table 11).

## Run it

```bash
git clone https://github.com/ChelsieLei/CrossHOI-Bench repo
pip install -r requirements.txt
python fetch_predictions.py      # 5 HICO-DET prediction files, ~270 MB, into ./predictions
python rescore.py                # released-protocol scores for Settings 1 and 3 (about 30 s on CPU)
python experiments.py            # issue counts, variants, k sweeps, bootstrap CIs, duplicates, coverage
```

`rescore.py` imports label tables from the cloned repo (`repo/hico_text_label.py`) and reads its benchmark files (`repo/hicodet/our_test/`). Nothing from that repo or the predictions is redistributed here. In `select()`, the switches are:
- `fix_noint=True`: spelling normalisation;
- `rank_within="target"`: target-matched ranking;
- `dedup=True`: de-duplication.

An independent second check of the selections against the authors' original metric functions agreed with this port on every audited configuration.

Code: MIT. Corrections are welcome as issues.
