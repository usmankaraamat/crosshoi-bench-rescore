"""CPU port of CrossHOI-Bench's specialist-model scoring (repo/script_eval_32B_newbench_HOI.py
+ repo/newbench_question_func.py), with switches for the variants we want to test.

Faithful mode (all switches off, k=5) must reproduce the paper's Table 1 / Table 3 numbers.
"""
import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from hico_text_label import HICO_INTERACTIONS  # numpy-only module

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = {"S1": "crosshoi_bench_hum_box_final.json", "S3": "crosshoi_bench_allscene_final.json"}
MODELS = {"ADA-CM": "ADA_CM-test_default_L.json", "CMMP": "CMMP-test_default_L.json",
          "LAIN": "LAIN-test_default_B.json", "HOLa": "HOLa-test_default_L.json",
          "CMD-SE": "CMD_SE-test_default_B.json"}


def box_iou(a, b):
    # torchvision.ops.box_iou semantics: a (1,4), b (N,4), no +1
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def norm_noint(label):
    # benchmark choices say "no_interaction with a/an X"; HOI models emit "no_interaction a/an X"
    return label.replace("no_interaction a/an ", "no_interaction with a/an ")


def select(pred, q, setting, k, fix_noint=False, rank_within="image", dedup=False, k_rule="fixed",
           thresh=None):
    """Return the list of answer strings the specialist 'gives' for one question.

    rank_within: 'image'   = paper: cut-off is the k-th highest score over ALL pairs in the image
                 'target'  = cut-off over the pairs matched to the target person only
                 'choices' = rank only predictions that are among the 4 options (what a VLM sees)
    k_rule: 'fixed' = k; 'oracle' = number of GT answers for this question
    thresh: if set, use a score threshold instead of top-k
    """
    labels = pred["ao_names"]
    if fix_noint:
        labels = [norm_noint(l) for l in labels]
    scores = np.asarray(pred["ho_scores"], dtype=np.float32)
    if len(pred["h_boxes"]) == 0 or len(pred["o_boxes"]) == 0:
        return []
    if setting == "S1":
        hb = np.asarray(pred["h_boxes"], dtype=np.float32)
        gt = np.asarray([q["boxes"]["human"]], dtype=np.float32)
        idx = np.nonzero(box_iou(gt, hb)[0] > 0.5)[0]
    else:
        idx = np.arange(len(labels))
    choices = q["gt_choices"] + q["wrong_choices"]
    kk = len(q["gt_choices"]) if k_rule == "oracle" else k

    if thresh is not None:
        cut = np.float32(thresh)
        pool = idx
    elif rank_within == "image":
        n = min(int(kk), len(scores))
        cut = np.sort(scores)[::-1][n - 1] if n > 0 else np.inf
        pool = idx
    elif rank_within == "target":
        if len(idx) == 0:
            return []
        n = min(int(kk), len(idx))
        cut = np.sort(scores[idx])[::-1][n - 1]
        pool = idx
    elif rank_within == "choices":
        pool = np.array([i for i in idx if labels[i] in choices], dtype=int)
        if len(pool) == 0:
            return []
        if dedup:  # best score per distinct label, then top-k distinct labels
            best = {}
            for i in pool:
                if labels[i] not in best or scores[i] > scores[best[labels[i]]]:
                    best[labels[i]] = i
            pool = np.array(list(best.values()), dtype=int)
        n = min(int(kk), len(pool))
        cut = np.sort(scores[pool])[::-1][n - 1]
    out = []
    for i in pool:
        if scores[i] < cut:
            continue
        if labels[i] in choices:
            if dedup and labels[i] in out:
                continue
            out.append(labels[i])
    return out


def score(bench, preds, sel):
    """mllm_instancef1_eval + mllm_macrof1_eval, faithful. sel: file -> answer list."""
    f1s, macro, tp, fp, full_gt, em = [], {}, 0, 0, 0, 0
    per_q = {}
    for f in preds:
        if f not in bench:
            continue
        q = bench[f]
        gti, wrong = q["gt_choices"], q["wrong_choices"]
        rpi = sel.get(f, [])
        if len(rpi) == 0:
            full_gt += len(gti)
            f1 = 0 if len(gti) > 0 else 1
            f1s.append(f1); per_q[f] = (f1, 0)
            for g in gti:
                macro.setdefault(g, {"tp": 0, "fp": 0, "gt": 0})["gt"] += 1
            continue
        pool = gti.copy(); tq = fq = 0
        for r in rpi:
            macro.setdefault(r, {"tp": 0, "fp": 0, "gt": 0})
            if r in pool:
                tp += 1; tq += 1; macro[r]["tp"] += 1; pool.remove(r)
            elif r in wrong:
                fp += 1; fq += 1; macro[r]["fp"] += 1
        full_gt += len(gti)
        true_pred = [i for i in rpi if i in gti or i in wrong]
        exact = int(len(true_pred) > 0 and len(true_pred) == len(gti) and set(true_pred) == set(gti))
        em += exact
        for g in gti:
            macro.setdefault(g, {"tp": 0, "fp": 0, "gt": 0})["gt"] += 1
        p = tq / (tq + fq) if tq + fq else 0
        r_ = tq / len(gti) if gti else 0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0
        f1s.append(f1); per_q[f] = (f1, exact)
    # macro over HICO 600 classes, keyed (act, obj)
    cls = {(h["action"].replace(" ", "_"), h["object"].replace(" ", "_")): [0, 0, 0] for h in HICO_INTERACTIONS}
    for lbi, v in macro.items():
        act, obj = lbi.split(" a/an ")
        act, obj = act.replace(" ", "_"), obj.replace(" ", "_")
        if "no_interaction" in act:
            act = "no_interaction"
        if (act, obj) in cls:
            c = cls[(act, obj)]; c[0] += v["tp"]; c[1] += v["fp"]; c[2] += v["gt"]
    mf = []
    for t, fpc, g in cls.values():
        if t == 0 and fpc == 0 and g == 0:
            continue
        if t == 0:
            mf.append(0.0)
        else:
            pr, rc = t / (t + fpc), t / g
            mf.append(2 * pr * rc / (pr + rc))
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / full_gt
    return {"macro": 100 * np.mean(mf), "inst": 100 * np.mean(f1s),
            "micro": 100 * (2 * prec * rec / (prec + rec) if prec + rec else 0),
            "em": 100 * em / len(bench), "per_q": per_q}


def load(setting, model):
    bench = json.load(open(os.path.join(HERE, "repo/hicodet/our_test", BENCH[setting])))
    preds = json.load(open(os.path.join(HERE, "predictions", MODELS[model])))
    return bench, preds


def run(setting, model, cache={}, **kw):
    key = (setting, model)
    if key not in cache:
        cache[key] = load(setting, model)
    bench, preds = cache[key]
    sel = {f: select(preds[f], bench[f], setting, **kw) for f in preds if f in bench}
    return score(bench, preds, sel)


if __name__ == "__main__":
    for s in ("S1", "S3"):
        for m in MODELS:
            r = run(s, m, k=5)
            print(f"{s} {m:7s} macro {r['macro']:.2f} inst {r['inst']:.2f} micro {r['micro']:.2f} em {r['em']:.2f}")
