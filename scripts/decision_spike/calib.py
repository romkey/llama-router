"""Can calibration or a contrastive prompt rescue noul and score? Leave-one-out evaluation."""

import math
import statistics
import sys

import spike

base, model = sys.argv[1], sys.argv[2]


def q_noul_contrast(t):
    p = spike.state_block(t) + (
        "Question: Is this message urgent?\nOptions:\n"
        "A) Urgent: something is broken or blocked right now, or there is an explicit deadline within hours\n"
        "B) Not urgent: a question, request or complaint that can wait for a normal reply\n"
        "Answer with the option letter only."
    )
    return p, {"a": 1, "b": 0}


def q_score_anchored(t):
    p = spike.state_block(t) + (
        "Question: How frustrated does the customer appear?\nLevels:\n"
        "0: Neutral or friendly; asks a question or states facts with no complaint\n"
        "1: Mildly annoyed; names a problem as frustrating or disappointing but stays polite\n"
        "2: Angry; capitals, threats, demands like 'fix it today', or sarcasm\n"
        "Answer with the level number only."
    )
    return p, {"0": 0, "1": 1, "2": 2}


def collect(qfn):
    out = []
    for t, _d, _u, _f in spike.DATA:
        p, labels = qfn(t)
        dist, _ = spike.distribution(spike.chat(base, model, p)[0], labels)
        out.append(dist)
    return out


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels, strict=False) if y]
    neg = [s for s, y in zip(scores, labels, strict=False) if not y]
    return statistics.mean((p > n) + 0.5 * (p == n) for p in pos for n in neg)


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def platt_fit(xs, ys, iters=3000, lr=0.05):
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys, strict=False):
            p = 1 / (1 + math.exp(-(a * x + b)))
            ga += (p - y) * x
            gb += p - y
        a -= lr * ga / len(xs)
        b -= lr * gb / len(xs)
    return a, b


def eval_noul(name, dists):
    ys = [u for _, _, u, _ in spike.DATA]
    ps = [d[1] for d in dists]
    raw_acc = statistics.mean((p > 0.5) == y for p, y in zip(ps, ys, strict=False))
    raw_brier = statistics.mean((p - y) ** 2 for p, y in zip(ps, ys, strict=False))
    xs = [logit(p) for p in ps]
    cal = []
    for i in range(len(xs)):  # leave-one-out Platt scaling
        a, b = platt_fit(xs[:i] + xs[i + 1 :], ys[:i] + ys[i + 1 :])
        cal.append(1 / (1 + math.exp(-(a * xs[i] + b))))
    c_acc = statistics.mean((p > 0.5) == y for p, y in zip(cal, ys, strict=False))
    c_brier = statistics.mean((p - y) ** 2 for p, y in zip(cal, ys, strict=False))
    print(
        f"noul {name:9s} AUC={auc(ps, ys):.2f} raw acc={raw_acc:.2f} brier={raw_brier:.3f} | "
        f"LOO-Platt acc={c_acc:.2f} brier={c_brier:.3f}"
    )


def bias_fit(dists, ys, k=3, iters=2000, lr=0.1):
    """Per-level additive log-offsets (prior correction), fit by gradient descent on log loss."""
    w = [0.0] * k
    for _ in range(iters):
        g = [0.0] * k
        for d, y in zip(dists, ys, strict=False):
            z = [math.log(max(d[j], 1e-9)) + w[j] for j in range(k)]
            m = max(z)
            e = [math.exp(v - m) for v in z]
            s = sum(e)
            for j in range(k):
                g[j] += e[j] / s - (1 if j == y else 0)
        w = [wj - lr * gj / len(dists) for wj, gj in zip(w, g, strict=False)]
    return w


def apply_bias(d, w):
    z = [math.log(max(d[j], 1e-9)) + w[j] for j in range(len(w))]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return {j: e[j] / s for j in range(len(w))}


def eval_score(name, dists):
    ys = [f for _, _, _, f in spike.DATA]

    def stats(ds):
        acc = statistics.mean(
            int(max(d, key=d.get)) == y for d, y in zip(ds, ys, strict=False)
        )
        mae = statistics.mean(
            abs(sum(int(i) * p for i, p in d.items()) - y)
            for d, y in zip(ds, ys, strict=False)
        )
        return acc, mae

    ra, rm = stats(dists)
    cal = []
    for i in range(len(dists)):
        w = bias_fit(dists[:i] + dists[i + 1 :], ys[:i] + ys[i + 1 :])
        cal.append(apply_bias(dists[i], w))
    ca, cm = stats(cal)
    print(
        f"score {name:8s} raw acc={ra:.2f} MAE={rm:.2f} | LOO-bias acc={ca:.2f} MAE={cm:.2f}"
    )


print(f"== {model}")
eval_noul("yes/no", collect(spike.q_noul))
eval_noul("contrast", collect(q_noul_contrast))
eval_score("rubric", collect(spike.q_score))
eval_score("anchored", collect(q_score_anchored))
