"""Compare diag_determinism.py records across runs and find where they first diverge (diagnostic branch only).

    uv run python scripts/diag_compare.py DIR [--md summary.md] [--json summary.json]

DIR holds the JSON files written by diag_determinism.py (any depth). Runs are grouped by the hash of all their
pipeline records (stages 10-90); for every group other than the reference, the first record whose hash differs
is reported in pipeline order, plus the first differing record of each stage. Kernel probes (stage 00) are
compared separately, because they show whether the CPU/library itself gives different bits.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

IGNORED = {"51_worker"}


def fp(records) -> str:
    return hashlib.sha256("".join(f"{r['stage']}|{r['key']}|{r['hash']}\n" for r in records).encode()).hexdigest()[:12]


def load(directory: Path) -> list[dict]:
    runs = []
    for f in sorted(directory.rglob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "records" not in d:
            continue
        seen = Counter()
        recs = {}
        for r in d["records"]:
            if r["stage"] in IGNORED:
                continue
            k = (r["stage"], r["key"])
            seen[k] += 1
            recs[k if seen[k] == 1 else (r["stage"], f"{r['key']}#{seen[k]}")] = r
        env = d["env"]
        arch = sorted({i.get("architecture") or "" for i in env.get("threadpool_main", []) if i.get("internal_api") == "openblas"})
        pipe = [r for k, r in recs.items() if k[0] >= "10"]
        runs.append({
            "file": str(f.relative_to(directory)), "label": d.get("label") or f.stem, "seconds": d.get("seconds"),
            "run_id": (env.get("github") or {}).get("GITHUB_RUN_ID"), "cpu": env.get("cpu_model"),
            "avx512": bool(env.get("avx512")), "openblas_arch": "/".join(arch),
            "npy_v4": "X86_V4" in env.get("numpy_cpu_features_on", []) and "X86_V4" in env.get("numpy_cpu_dispatch", []),
            "thread_env": {k: v for k, v in (env.get("thread_env") or {}).items() if v},
            "recs": recs,
            "probe_fp": fp([r for k, r in recs.items() if k[0] == "00_probe"]),
            "pipe_fp": fp(pipe),
            "table_fp": fp([r for k, r in recs.items() if k[0] == "90_table"]),
        })
    return runs


def first_diffs(ref: dict, other: dict) -> tuple[dict | None, dict]:
    first, per_stage = None, {}
    for k, r in ref["recs"].items():
        o = other["recs"].get(k)
        if o is not None and o["hash"] == r["hash"]:
            continue
        item = {"stage": k[0], "key": k[1], "ref": r, "other": o}
        if k[0] != "00_probe" and first is None:
            first = item
        per_stage.setdefault(k[0], item)
    return first, per_stage


def count_diffs(ref: dict, other: dict, stage: str) -> Counter:
    c = Counter()
    for k, r in ref["recs"].items():
        if k[0] == stage and other["recs"].get(k, {}).get("hash") != r["hash"]:
            c[k[1].split("|")[1].split("[")[0] if stage == "50_ml_fold" else k[1].split("|")[0]] += 1
    return c


def mlp_stats(run: dict) -> dict:
    d = [r["detail"] for k, r in run["recs"].items() if k[0] == "50_ml_fold" and "|mlp[" in k[1]]
    if not d:
        return {}
    return {"folds": len(d), "not_converged": sum(not x["converged"] for x in d),
            "hit_max_iter": sum(x["hit_max_iter"] for x in d),
            "messages": Counter(x["message"] for x in d).most_common(4),
            "n_iter_median": sorted(x["n_iter"] for x in d)[len(d) // 2]}


def short(v, n=6) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.10g}" if isinstance(x, float) else str(x) for x in v[:n]) + (", ...]" if len(v) > n else "]")
    return f"{v:.12g}" if isinstance(v, float) else str(v)


def describe(item: dict) -> list[str]:
    r, o = item["ref"], item["other"]
    out = [f"- **{item['stage']}** `{item['key']}`"]
    if o is None:
        return out + ["  - missing in the other run (different key set: the computation upstream changed)"]
    rd, od = r.get("detail", {}), o.get("detail", {})
    for f in ("forecast", "sse", "params", "n_iter", "loss", "status", "message", "chosen", "inner_wape", "fun", "nit"):
        if f in rd and rd.get(f) != od.get(f):
            out.append(f"  - {f}: ref {short(rd[f])} / other {short(od.get(f))}")
    if rd.get("opt"):
        for i, (a, b) in enumerate(zip(rd["opt"], od.get("opt", []))):
            out.append(f"  - optimiser call {i}: x0 ref {short(a['x0'])} / other {short(b['x0'])}; "
                       f"x0 identical={a['x0'] == b['x0']}; fun {a['fun']:.15g} / {b['fun']:.15g}; "
                       f"nit {a['nit']}/{b['nit']}; msg {a['message']!r} / {b['message']!r}")
    for f in ("train", "test", "unordered"):
        if f in rd:
            out.append(f"  - {f} hash equal: {rd[f] == od.get(f)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    runs = load(args.directory)
    if not runs:
        raise SystemExit("no diag JSON files found")
    L = ["# Determinism diagnostic summary", "",
         "| run | GitHub run | CPU | AVX-512 | OpenBLAS arch | numpy X86_V4 | thread env | probe fp | pipeline fp | table fp | s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        env = ", ".join(f"{k}={v}" for k, v in r["thread_env"].items()) or "-"
        L.append(f"| {r['label']} | {r['run_id']} | {r['cpu']} | {'yes' if r['avx512'] else 'no'} | {r['openblas_arch']} | "
                 f"{'on' if r['npy_v4'] else 'off'} | {env} | {r['probe_fp']} | {r['pipe_fp']} | {r['table_fp']} | {r['seconds']} |")

    by_variant = defaultdict(list)
    for r in runs:
        by_variant[r["label"].rsplit("-r", 1)[0]].append(r)
    L += ["", "## Reproducibility per variant", "",
          "| variant | runs | distinct pipeline fp | distinct table fp | distinct probe fp | same-job A==B | CPUs seen |",
          "|---|---|---|---|---|---|---|"]
    summary = {}
    for v, rs in by_variant.items():
        pairs = defaultdict(dict)
        for r in rs:
            stem, _, ab = r["label"].rpartition("-")
            pairs[stem][ab] = r
        ab = [p["A"]["pipe_fp"] == p["B"]["pipe_fp"] for p in pairs.values() if "A" in p and "B" in p]
        cpus = Counter(r["cpu"] for r in rs)
        summary[v] = {"runs": len(rs), "pipe_fp": len({r["pipe_fp"] for r in rs}), "table_fp": len({r["table_fp"] for r in rs}),
                      "probe_fp": len({r["probe_fp"] for r in rs}), "ab_equal": f"{sum(ab)}/{len(ab)}", "cpus": dict(cpus)}
        s = summary[v]
        L.append(f"| {v} | {s['runs']} | {s['pipe_fp']} | {s['table_fp']} | {s['probe_fp']} | {s['ab_equal']} | "
                 + "; ".join(f"{c} x{n}" for c, n in cpus.items()) + " |")

    L += ["", "## Pipeline fingerprint by CPU and variant", "", "| variant | CPU | pipeline fps (count) |", "|---|---|---|"]
    for v, rs in by_variant.items():
        g = defaultdict(Counter)
        for r in rs:
            g[r["cpu"]][r["pipe_fp"]] += 1
        for cpu, c in g.items():
            L.append(f"| {v} | {cpu} | " + ", ".join(f"{k} ({n})" for k, n in c.items()) + " |")

    L += ["", "## First divergence", ""]
    for v, rs in by_variant.items():
        ref = rs[0]
        done = {ref["pipe_fp"]}
        for o in rs[1:]:
            if o["pipe_fp"] in done:
                continue
            done.add(o["pipe_fp"])
            first, per = first_diffs(ref, o)
            L += [f"### {v}: {ref['label']} ({ref['cpu']}) vs {o['label']} ({o['cpu']})", ""]
            if first:
                L += ["First differing pipeline record:", *describe(first), ""]
            L += ["First differing record of each stage:"]
            for st in sorted(per):
                L += describe(per[st])
            for st in ("40_stat_fit", "50_ml_fold", "70_reconcile", "90_table"):
                c = count_diffs(ref, o, st)
                if c:
                    L.append(f"- differing {st} records by model: " + ", ".join(f"{k} {n}" for k, n in c.most_common()))
            L.append("")
        if len(done) == 1:
            L += [f"### {v}: all {len(rs)} runs identical in every pipeline record", ""]

    L += ["## Probe differences across all runs", ""]
    probe_keys = [k for k in runs[0]["recs"] if k[0] == "00_probe"]
    for k in probe_keys:
        vals = defaultdict(set)
        for r in runs:
            if k in r["recs"]:
                vals[r["recs"][k]["hash"]].add(f"{r['cpu']} / arch {r['openblas_arch']} / X86_V4 {'on' if r['npy_v4'] else 'off'}")
        if len(vals) > 1:
            L.append(f"- `{k[1]}`: " + " | ".join(f"{h}: {sorted(s)}" for h, s in vals.items()))

    L += ["", "## MLP L-BFGS convergence (per run)", "", "| run | folds | not converged | hit max_iter | median n_iter | messages |",
          "|---|---|---|---|---|---|"]
    for r in runs:
        m = mlp_stats(r)
        if m:
            L.append(f"| {r['label']} | {m['folds']} | {m['not_converged']} | {m['hit_max_iter']} | {m['n_iter_median']} | "
                     + "; ".join(f"{msg} x{n}" for msg, n in m["messages"]) + " |")
    text = "\n".join(L) + "\n"
    print(text)
    if args.md:
        args.md.write_text(text, encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps({"variants": summary,
                                         "runs": [{k: v for k, v in r.items() if k != "recs"} for r in runs]},
                                        indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
