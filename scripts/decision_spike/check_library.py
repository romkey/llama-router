"""Run the Phase 0 ticket set through llama_router.decisions against a live
llama.cpp server. Usage: uv run python scripts/decision_spike/check_library.py URL MODEL"""

import json
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import spike

from llama_router import decisions as d

base, model = sys.argv[1], sys.argv[2]
URGENT_NEG = "The message can wait for a normal reply: nothing is broken right now and there is no deadline within hours"
QS = {
    "dept": d.ChoiceQuestion(
        type="choice",
        instructions="Which team should handle this?",
        criteria=spike.DEPTS,
    ),
    "urgent_generic": d.NoulQuestion(
        type="noul", instructions="The message conveys urgency or time-sensitivity"
    ),
    "urgent_negative": d.NoulQuestion(
        type="noul",
        instructions="The message conveys urgency or time-sensitivity",
        negative=URGENT_NEG,
    ),
    "frustration": d.ScoreQuestion(
        type="score",
        instructions="How frustrated does the customer appear?",
        criteria=spike.FRUSTRATION,
    ),
}


def call(body):
    req = urllib.request.Request(  # noqa: S310  # --base is operator-supplied
        base + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        return json.load(r)


hits = {k: [] for k in QS}
masses = []
for ticket, dept, urgent, frust in spike.DATA:
    truth = {
        "dept": dept,
        "urgent_generic": urgent,
        "urgent_negative": urgent,
        "frustration": frust,
    }
    for k, q in QS.items():
        p = d.build_prompt(ticket, q)
        dist = d.distribution(
            d.top_logprobs_from_response(call(d.upstream_body(model, p))), p.labels
        )
        masses.append(dist.label_mass)
        a = d.answer(q, dist)
        if a["type"] == "choice":
            hits[k].append(a["choice"] == truth[k])
        elif a["type"] == "noul":
            hits[k].append((a["noul"] > 0.5) == bool(truth[k]))
        else:
            hits[k].append(
                max(a["probabilities"], key=a["probabilities"].get) == str(truth[k])
            )
print(f"{model}: min label mass {min(masses):.3f}")
for k, v in hits.items():
    print(f"  {k:16s} acc={statistics.mean(v):.2f}")
