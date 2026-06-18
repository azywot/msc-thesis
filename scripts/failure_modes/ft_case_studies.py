"""Pull representative base-vs-LoRA trajectories for the FT report case studies.

Categories: wrong->right (LoRA helps), right->wrong (LoRA hurts),
both-wrong-different, both-identical. Prints compact trajectory excerpts.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_failure_modes import classify_failure  # noqa

ROOT = Path(__file__).resolve().parent.parent.parent

def latest(folder):
    folder = ROOT / folder
    cands = [d for d in folder.iterdir() if d.is_dir() and (d/"raw_results.json").exists()]
    return sorted(cands, key=lambda d: d.name)[-1]

def load(folder):
    rd = latest(folder)
    recs = json.load(open(rd/"raw_results.json"))
    return {r["question_id"]: r for r in recs}

def traj(r, maxres=220):
    out = []
    for s in (r.get("action_history") or []):
        sg = (s.get("sub_goal") or "").strip().replace("\n"," ")[:140]
        cmd = (s.get("command") or "").strip().replace("\n"," ")[:120]
        res = (s.get("result") or "").strip().replace("\n"," ")[:maxres]
        out.append(f"    [{s.get('tool_name')}] goal={sg!r}\n       cmd={cmd!r}\n       -> {res!r}")
    return "\n".join(out) if out else "    (no tool calls)"

def show(tag, ds, base_f, v_f, qids):
    B = load(base_f); V = load(v_f)
    print(f"\n################ {tag} ({ds}) ################")
    for qid in qids:
        if qid not in B or qid not in V:
            continue
        b, v = B[qid], V[qid]
        q = (b.get("question") or "").strip().replace("\n"," ")
        print(f"\n--- qid={qid}  GT={str(b.get('ground_truth'))[:60]!r}")
        print(f"Q: {q[:300]}")
        print(f"BASE  correct={b.get('correct')}  turns={b.get('turns')}  pred={str(b.get('prediction'))[:80]!r}  fmode={None if b.get('correct') else classify_failure(b)}")
        print(traj(b))
        print(f"LoRA  correct={v.get('correct')}  turns={v.get('turns')}  pred={str(v.get('prediction'))[:80]!r}  fmode={None if v.get('correct') else classify_failure(v)}")
        print(traj(v))

def pick(base_f, v_f, want, k=2):
    """Find qids matching a transition category. want in {WR,RW,WWdiff,identical}."""
    B = load(base_f); V = load(v_f)
    res = []
    for qid in sorted(set(B)&set(V)):
        b, v = B[qid], V[qid]
        bc, vc = bool(b.get("correct")), bool(v.get("correct"))
        bseq = tuple(s.get("tool_name") for s in (b.get("action_history") or []))
        vseq = tuple(s.get("tool_name") for s in (v.get("action_history") or []))
        bp, vp = (b.get("prediction") or "").strip(), (v.get("prediction") or "").strip()
        if want=="WR" and (not bc and vc): res.append(qid)
        elif want=="RW" and (bc and not vc): res.append(qid)
        elif want=="WWdiff" and (not bc and not vc and bp!=vp): res.append(qid)
        elif want=="identical" and (bp and bp==vp and bseq==vseq): res.append(qid)
    return res[:k]

if __name__ == "__main__":
    # AIME thinking-OFF base vs v2 (trained regime, clearest math regression)
    bf = "experiments/results/orchestrator_capacity/aime/orch8b_sub1_7b_none"
    vf = "experiments/results/lora_inference/aime/qwen8B_sub1_7b_none_v2"
    for cat in ("WR","RW","WWdiff","identical"):
        show(f"AIME thinking-OFF base->v2  [{cat}]", "aime", bf, vf, pick(bf,vf,cat,1))
    # GPQA thinking-OFF base vs v2
    bf = "experiments/results/orchestrator_capacity/gpqa/orch8b_sub1_7b_none"
    vf = "experiments/results/lora_inference/gpqa/qwen8B_sub1_7b_none_v2"
    for cat in ("WR","RW","identical"):
        show(f"GPQA thinking-OFF base->v2  [{cat}]", "gpqa", bf, vf, pick(bf,vf,cat,1))
