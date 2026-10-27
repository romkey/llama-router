"""Phase 0 spike: emulate Jev-style decision questions on llama.cpp via first-token logprobs.

Usage: python spike.py <base_url> <model> [--scheme letters|words] [--shuffle]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# --- Labelled triage set: (ticket, department, urgent, frustration 0..2) ---
DATA = [
    (
        "Our API started returning 500 errors on every request 20 minutes ago and our checkout is down. Please help now!",
        "technical",
        1,
        2,
    ),
    (
        "I was charged twice for my subscription this month. Can you refund one of the charges?",
        "billing",
        0,
        1,
    ),
    ("What does the enterprise plan cost for 200 seats?", "sales", 0, 0),
    (
        "The SDK throws a TypeError when I pass a dict to create_session. Stack trace attached.",
        "technical",
        0,
        0,
    ),
    (
        "This is the THIRD time I've asked. Cancel my account and stop charging my card or I'm calling my bank.",
        "billing",
        1,
        2,
    ),
    ("Do you offer discounts for nonprofits?", "sales", 0, 0),
    (
        "Webhooks stopped arriving yesterday. Nothing changed on our side.",
        "technical",
        0,
        1,
    ),
    (
        "My invoice shows the wrong company name. Could you update it to Acme Corp?",
        "billing",
        0,
        0,
    ),
    (
        "We'd like a demo of the analytics add-on next week for our leadership team.",
        "sales",
        0,
        0,
    ),
    (
        "Login page is completely broken for all our users right now. We're losing customers by the minute. Unacceptable.",
        "technical",
        1,
        2,
    ),
    (
        "Why did my bill go up by 40% without any notice? This is pretty frustrating.",
        "billing",
        0,
        1,
    ),
    (
        "Can I upgrade from the starter plan to pro mid-cycle, and how is that priced?",
        "sales",
        0,
        0,
    ),
    (
        "Rate limits seem lower than documented: we get 429s at 50 requests per second.",
        "technical",
        0,
        1,
    ),
    (
        "Payment failed with my new card. I need this fixed before our launch in two hours.",
        "billing",
        1,
        1,
    ),
    ("Is there a volume discount if we commit to an annual contract?", "sales", 0, 0),
    (
        "The dashboard takes 30 seconds to load every page. It's been like this all week and it's driving us nuts.",
        "technical",
        0,
        2,
    ),
    (
        "Please send me a copy of last year's receipts for tax purposes.",
        "billing",
        0,
        0,
    ),
    (
        "We're evaluating vendors and need a security questionnaire filled out along with pricing.",
        "sales",
        0,
        0,
    ),
    (
        "Production database migration failed halfway and the app is down. Need an engineer immediately.",
        "technical",
        1,
        1,
    ),
    ("You charged my card after I cancelled. Fix it. Today.", "billing", 1, 2),
    ("Hi, just curious whether you have a reseller program?", "sales", 0, 0),
    (
        "The CSV export has the columns in the wrong order since the last update.",
        "technical",
        0,
        0,
    ),
    ("How do I change the credit card on file?", "billing", 0, 0),
    (
        "Our contract renews next month; can we talk about adding more seats at a better rate?",
        "sales",
        0,
        0,
    ),
    (
        "Every single deploy fails with a certificate error. Your status page says all green. Seriously?",
        "technical",
        0,
        2,
    ),
    (
        "I was promised a refund three weeks ago and still haven't seen it. I'm very disappointed.",
        "billing",
        0,
        2,
    ),
    ("Could someone walk me through which plan fits a team of 15?", "sales", 0, 0),
    (
        "Mobile app crashes on launch for iOS 19 users. About half our customers are affected as of this morning.",
        "technical",
        1,
        1,
    ),
    (
        "Tax was added to my invoice but we're tax exempt. Certificate attached.",
        "billing",
        0,
        0,
    ),
    (
        "Our board meets tomorrow morning and I need a quote for the enterprise tier tonight.",
        "sales",
        1,
        0,
    ),
    (
        "Search results are returning documents from other tenants. This is a data leak, escalate now.",
        "technical",
        1,
        2,
    ),
    ("Minor thing: the receipt email has a typo in the footer.", "billing", 0, 0),
    ("Do you have a partner program for agencies?", "sales", 0, 0),
    (
        "SSO with Okta stopped working after your maintenance window. Nobody on our team can log in.",
        "technical",
        1,
        1,
    ),
    (
        "I keep getting billed in USD but my account is set to EUR. Annoying, please fix.",
        "billing",
        0,
        1,
    ),
    (
        "We want to expand to three more regions; who can discuss pricing for that?",
        "sales",
        0,
        0,
    ),
]

DEPTS = {
    "billing": "Payment, invoices, refunds or subscription charges",
    "technical": "Bugs, outages, errors or integration problems",
    "sales": "Pricing, plans, quotes, demos or new purchases",
}
FRUSTRATION = [
    "Calm, just stating facts",
    "Frustrated but civil",
    "Very angry, strong language",
]

SYSTEM = (
    "You are a precise classifier. You read a state and answer one question about it. "
    "Reply with exactly one option label and nothing else."
)


def chat(base: str, model: str, prompt: str, top: int = 20) -> tuple[list[dict], float]:
    body = {
        "model": model,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top,
        "cache_prompt": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    ms = (time.monotonic() - t) * 1000
    return data["choices"][0]["logprobs"]["content"][0]["top_logprobs"], ms


def distribution(
    top: list[dict], labels: dict[str, str]
) -> tuple[dict[str, float], float]:
    """labels: token text (normalised) -> answer key. Returns (renormalised dist, raw mass on labels)."""
    lp: dict[str, list[float]] = {}
    for t in top:
        key = labels.get(t["token"].strip().lower().rstrip(")."))
        if key is not None:
            lp.setdefault(key, []).append(t["logprob"])
    raw = {k: sum(math.exp(x) for x in v) for k, v in lp.items()}
    mass = sum(raw.values())
    keys = set(labels.values())
    if mass == 0:
        return {k: 1 / len(keys) for k in keys}, 0.0
    return {k: raw.get(k, 0.0) / mass for k in keys}, mass


def state_block(ticket: str) -> str:
    return f"State:\n{ticket}\n\n"


def q_choice(ticket: str, scheme: str, order: list[str]):
    if scheme == "letters":
        labs = "ABCDEFG"[: len(order)]
        opts = "\n".join(f"{l}) {DEPTS[k]}" for l, k in zip(labs, order))
        labels = {l.lower(): k for l, k in zip(labs, order)}
        tail = "Answer with the option letter only."
    else:
        opts = "\n".join(f"{k}: {DEPTS[k]}" for k in order)
        labels = {k: k for k in order}
        tail = "Answer with the option name only."
    p = (
        state_block(ticket)
        + f"Question: Which team should handle this?\nOptions:\n{opts}\n{tail}"
    )
    return p, labels


def q_noul(ticket: str):
    p = state_block(ticket) + (
        "Statement: The message conveys urgency or time-sensitivity.\n"
        "Is the statement true of the state? Answer yes or no."
    )
    return p, {"yes": 1, "no": 0}


def q_score(ticket: str):
    opts = "\n".join(f"{i}: {d}" for i, d in enumerate(FRUSTRATION))
    p = (
        state_block(ticket)
        + f"Question: How frustrated does the customer appear?\nLevels:\n{opts}\nAnswer with the level number only."
    )
    return p, {str(i): i for i in range(len(FRUSTRATION))}


def brier(dist: dict, truth) -> float:
    return sum((p - (1.0 if k == truth else 0.0)) ** 2 for k, p in dist.items())


def ece(pairs: list[tuple[float, bool]], bins: int = 5) -> float:
    total, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pairs if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            err += (
                len(sel)
                / total
                * abs(
                    statistics.mean(c for c, _ in sel)
                    - statistics.mean(ok for _, ok in sel)
                )
            )
    return err


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--scheme", default="letters", choices=["letters", "words"])
    ap.add_argument(
        "--shuffle", action="store_true", help="randomise option order per item"
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    res = {"choice": [], "noul": [], "score": []}
    single_ms, batch_ms, masses = [], [], []
    pool = ThreadPoolExecutor(8)
    for ticket, dept, urgent, frust in DATA:
        order = list(DEPTS)
        if a.shuffle:
            rng.shuffle(order)
        qs = {
            "choice": (q_choice(ticket, a.scheme, order), dept),
            "noul": (q_noul(ticket), urgent),
            "score": (q_score(ticket), frust),
        }
        # one question alone (cold-ish prefix) then all three batched in parallel
        _, ms = chat(a.base, a.model, qs["choice"][0][0])
        single_ms.append(ms)
        t = time.monotonic()
        futs = {k: pool.submit(chat, a.base, a.model, v[0][0]) for k, v in qs.items()}
        outs = {k: f.result() for k, f in futs.items()}
        batch_ms.append((time.monotonic() - t) * 1000)
        for k, ((_, labels), truth) in qs.items():
            dist, mass = distribution(outs[k][0], labels)
            masses.append(mass)
            res[k].append((dist, truth))

    print(f"model={a.model} scheme={a.scheme} shuffle={a.shuffle} n={len(DATA)}")
    print(
        f"latency single q   p50={pct(single_ms, 0.5):.0f}ms p90={pct(single_ms, 0.9):.0f}ms"
    )
    print(
        f"latency 3q batched p50={pct(batch_ms, 0.5):.0f}ms p90={pct(batch_ms, 0.9):.0f}ms"
    )
    print(
        f"label mass (pre-renorm) min={min(masses):.3f} median={statistics.median(masses):.3f}"
    )
    for k, rows in res.items():
        acc = statistics.mean(max(d, key=d.get) == t for d, t in rows)
        b = statistics.mean(brier(d, t) for d, t in rows)
        conf = [(max(d.values()), max(d, key=d.get) == t) for d, t in rows]
        line = f"{k:6s} acc={acc:.2f} brier={b:.3f} ece={ece(conf):.3f} meanconf={statistics.mean(c for c, _ in conf):.2f}"
        if k == "score":
            mae = statistics.mean(
                abs(sum(int(i) * p for i, p in d.items()) - t) for d, t in rows
            )
            line += f" expected-level-MAE={mae:.2f}"
        print(line)


if __name__ == "__main__":
    main()
