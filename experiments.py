import json, os, numpy as np
from rescore import run, load, MODELS, select, box_iou, norm_noint

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ---- 1. diagnostics on the benchmark and on the selection rule ----
for s in ("S1", "S3"):
    bench, _ = load(s, "ADA-CM")
    n = len(bench)
    noint_gt = sum(any("no_interaction" in g for g in q["gt_choices"]) for q in bench.values())
    noint_opt = sum(any("no_interaction" in g for g in q["gt_choices"] + q["wrong_choices"]) for q in bench.values())
    multi = sum(len(q["gt_choices"]) > 1 for q in bench.values())
    print(f"{s}: {n} questions | GT contains no_interaction: {noint_gt} | any option no_interaction: {noint_opt} | multi-answer: {multi}")

print()
for m in MODELS:
    bench, preds = load("S1", m)
    no_target = crowded = 0
    for f, q in bench.items():
        p = preds.get(f)
        if p is None:
            no_target += 1; continue
        if not p["h_boxes"]:
            no_target += 1; continue
        hb = np.asarray(p["h_boxes"], np.float32)
        idx = np.nonzero(box_iou(np.asarray([q["boxes"]["human"]], np.float32), hb)[0] > 0.5)[0]
        if len(idx) == 0:
            no_target += 1; continue
        choices = q["gt_choices"] + q["wrong_choices"]
        lab = [norm_noint(l) for l in p["ao_names"]]
        on_choice = [i for i in idx if lab[i] in choices]
        if on_choice and not select(p, q, "S1", 5, fix_noint=True):
            crowded += 1
    print(f"S1 {m:7s} target person not detected: {no_target}/{len(bench)} | "
          f"target has an on-option prediction but top-5 (image-wide) leaves it with nothing: {crowded}")

# ---- 2. variants ----
VARIANTS = {
    "paper (image top-5)":            dict(k=5),
    "+ no_interaction label fix":     dict(k=5, fix_noint=True),
    "+ rank within target person":    dict(k=5, fix_noint=True, rank_within="target"),
    "+ rank within the 4 options":    dict(k=5, fix_noint=True, rank_within="choices", dedup=True),
    "options, top-1":                 dict(k=1, fix_noint=True, rank_within="choices", dedup=True),
    "options, top-2":                 dict(k=2, fix_noint=True, rank_within="choices", dedup=True),
    "options, k = #GT (oracle)":      dict(k=5, k_rule="oracle", fix_noint=True, rank_within="choices", dedup=True),
}
VLM = {"S1": ("Qwen2.5-VL-32B", 50.71, 52.94, 61.41, 26.06), "S3": ("Qwen2.5-VL-32B", 63.16, 67.19, 68.82, 23.16)}
res = {}
for s in ("S1", "S3"):
    print(f"\n=== {s} ===   VLM {VLM[s][0]}: macro {VLM[s][1]} inst {VLM[s][2]} micro {VLM[s][3]} em {VLM[s][4]}")
    for vname, kw in VARIANTS.items():
        if s == "S3" and vname == "+ rank within target person":
            continue
        row = []
        for m in MODELS:
            r = run(s, m, **kw)
            res[(s, vname, m)] = r
            row.append(f"{m} {r['macro']:5.2f}/{r['inst']:5.2f}/{r['micro']:5.2f}/{r['em']:5.2f}")
        print(f"{vname:32s} | " + " | ".join(row))

# ---- 3. k sweep and threshold sweep (fixed labels, ranked within options) ----
print("\nk sweep, ranked within options (macro/inst/em), and paper rule (image-wide):")
for s in ("S1", "S3"):
    for m in ("ADA-CM", "HOLa", "CMD-SE"):
        a = [run(s, m, k=k, fix_noint=True, rank_within="choices", dedup=True) for k in (1, 2, 3, 4)]
        b = [run(s, m, k=k) for k in (1, 3, 5, 7, 10)]
        print(f"{s} {m:7s} options k1-4: " + "  ".join(f"{x['macro']:.1f}/{x['inst']:.1f}/{x['em']:.1f}" for x in a)
              + " || image k1,3,5,7,10: " + "  ".join(f"{x['macro']:.1f}/{x['inst']:.1f}/{x['em']:.1f}" for x in b))

json.dump({f"{k[0]}|{k[1]}|{k[2]}": {kk: v for kk, v in r.items() if kk != "per_q"} for k, r in res.items()},
          open(os.path.join(HERE, "results_variants.json"), "w"), indent=1)

# ---- 4. bootstrap CIs (S1, question-level, 1,000 resamples) ----
# Individual intervals for the "both changes" variant, plus paired intervals for the change
# against the released selection. Individual intervals overlapping another model's interval
# does NOT show equivalence; no paired test against the VLM is possible without its per-question outputs.
BOTH = "+ rank within target person"   # = label fix + target-matched top-5
rng = np.random.default_rng(0)
for m in ("ADA-CM", "HOLa", "CMMP"):
    a = res[("S1", "paper (image top-5)", m)]["per_q"]; b = res[("S1", BOTH, m)]["per_q"]
    keys = list(a)
    A = np.array([a[k] for k in keys], float); B = np.array([b[k] for k in keys], float)   # cols: inst-F1, exact
    idx = [rng.integers(0, len(keys), len(keys)) for _ in range(1000)]
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    lo, hi = ci([100 * B[i, 0].mean() for i in idx])
    dlo, dhi = ci([100 * (B[i, 0] - A[i, 0]).mean() for i in idx])
    elo, ehi = ci([100 * (B[i, 1] - A[i, 1]).mean() for i in idx])
    print(f"S1 {m:7s} both changes: inst-F1 {100*B[:,0].mean():.2f} CI [{lo:.2f}, {hi:.2f}] | "
          f"change vs released: inst-F1 {100*(B[:,0]-A[:,0]).mean():+.2f} [{dlo:+.2f}, {dhi:+.2f}], "
          f"EM-per-question {100*(B[:,1]-A[:,1]).mean():+.2f} [{elo:+.2f}, {ehi:+.2f}]")

# ---- 5. duplicate labels (the released scorer does not de-duplicate answer lists) ----
print()
for s, kw in (("S1", dict(k=5, fix_noint=True)), ("S1", dict(k=5, fix_noint=True, rank_within="target"))):
    for m in ("ADA-CM", "HOLa"):
        x = run(s, m, **kw); y = run(s, m, dedup=True, **kw)
        print(f"{s} {m:7s} {kw}: macro/inst/micro/EM {x['macro']:.2f}/{x['inst']:.2f}/{x['micro']:.2f}/{x['em']:.2f}"
              f"  -> de-duplicated {y['macro']:.2f}/{y['inst']:.2f}/{y['micro']:.2f}/{y['em']:.2f}")

# ---- 6. CMD-SE coverage ----
bench, preds = load("S1", "CMD-SE")
miss = [f for f in bench if f not in preds]
print(f"\nCMD-SE released predictions missing {len(miss)} of {len(bench)} benchmark images; "
      f"{sum(any('no_interaction' in g for g in bench[f]['gt_choices']) for f in miss)} of them have no_interaction as a correct answer (S1)")
