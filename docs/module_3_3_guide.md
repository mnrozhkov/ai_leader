
# 📦 Notebook A — Core MVP Evaluation (v3 specification)

---

# 1. Purpose

Notebook A helps you answer a practical product question:

> **Is the routing prototype strong enough to justify the next MVP step?**

You will evaluate:

* **Routing quality** (primary)
* **Safety under a review rule**
* **Cost at scale**
* **Speed feasibility**

The goal is **not to prove success**, but to support a **clear, justified decision**:

* Stop
* Improve and re-test
* Proceed with guardrails
* Proceed to MVP workflow

---

# 2. Evaluation logic

Notebook A follows a **two-stage process**:

```text
Stage 1 — Model selection
    ↓
Choose the most promising routing model

Stage 2 — MVP evaluation
    ↓
Test routing, safety, cost, speed
    ↓
Experiment → improve → decide
```

---

# 3. Hypotheses and metrics

## H1 — Routing quality (primary)

> The system must misroute few enough requests to be operationally useful.

**Primary metric**

* Misroute Rate

**Supporting**

* Department Accuracy
* Category Accuracy (diagnostic)

---

## H2 — Safety under review rule

> If only high-confidence cases are automated, risk must remain acceptable.

**Primary metric**

* Unsafe Auto-Route Rate

**Supporting**

* Auto-route coverage
* Auto-route precision
* Manual review rate

---

## H3 — Unit economics

> Cost per message must be acceptable at Candlekeep scale.

**Primary metric**

* Cost per message

**Supporting**

* Monthly cost
* Annual cost

---

## H4 — Speed feasibility

> Latency must be low enough to support a better workflow.

**Primary metric**

* Median latency

**Supporting**

* p95 latency

---

# 4. Separation of layers

## 4.1 User-facing notebook

Students interact with:

* configuration (models, prompt, params)
* comparison table
* evaluation outputs
* decision summary

They do **not** see:

* raw token usage
* internal orchestration
* intermediate artifacts

---

## 4.2 Implementation layer (.py)

The runner:

* executes model calls
* computes metrics
* saves outputs

Artifacts:

* predictions.csv
* run_meta.json
* summary.json
* decision_summary.json
* comparison_table.csv

---

# 5. User-facing notebook structure

## 5.1 Setup

Students configure:

* model list (optional)
* prompt (default or override)
* temperature / params (max 3)
* monthly volume

---

## 5.2 Model selection

### Goal

Find the strongest routing candidate.

### Metrics shown

| Model | Department Accuracy | Misroute Rate | Coverage |

### Selection rule

1. Lowest **misroute rate**
2. If tied → higher department accuracy
3. If tied → lower cost
4. If tied → lower latency

---

### Output

* comparison table
* optional plots
* suggested model

Students may **override selection**.

---

## 5.3 MVP evaluation (selected configuration)

Students run evaluation with chosen:

* model
* prompt
* parameters

---

### Routing quality (H1)

* Misroute rate
* Department accuracy

---

### Safety (H2)

* Auto-route coverage
* Auto-route precision
* Unsafe auto-route rate

---

### Cost (H3)

* Cost per message
* Monthly cost
* Source (measured / estimated)

---

### Speed (H4)

* Median latency
* Source (measured / unknown)

---

## 5.4 Results summary

Students see:

| Dimension | Status | Value |
| --------- | ------ | ----- |
| Routing   | ...    | ...   |
| Safety    | ...    | ...   |
| Cost      | ...    | ...   |
| Speed     | ...    | ...   |

---

## 5.5 Experimentation

Students are expected to iterate.

They can modify:

* model
* prompt
* parameters

Then:

* re-run evaluation
* compare results

---

## 5.6 Final decision

Students write a short decision:

```text
Decision: [Stop / Improve / Proceed]

Reason:
- Routing:
- Safety:
- Cost:
- Speed:
- Next step:
```

---

# 6. User-facing output

## decision_summary.json

```json
{
  "model": "...",
  "row_count": 100,
  "routing_verdict": {...},
  "safety_verdict": {...},
  "cost_verdict": {...},
  "speed_verdict": {...},
  "final_recommendation": "...",
  "short_rationale": "..."
}
```

---

# 7. Implementation flow (unchanged)

```text
1. Load dataset
2. Run models
3. Compute metrics
4. Save predictions + metadata
5. Build comparison table
6. Select best model
7. Evaluate decision
8. Save summaries
```

---

# 8. Coverage checks

Each run must include:

* dataset_row_count
* prediction_row_count
* scored_row_count
* coverage_rate

If coverage < 1.0:

* log warning
* save missing_row_ids.csv

---

# 9. Cost and latency sources

## Cost

* measured → token usage
* estimated → fallback

## Latency

* measured → from API timing
* unknown → no data

Latency is **never estimated**.

---

# 10. Decision logic

Each dimension:

* pass
* borderline
* fail
* unknown

Final recommendation:

* any fail → Improve and re-test
* all pass → Proceed
* unknown present → append “incomplete evaluation”

---

# 11. Notebook tutorial (student guide)

Use these section headers in the notebook.

---

## 🔹 Step 1 — Configure your experiment

* Choose models to compare
* Adjust prompt or parameters (optional)

👉 Goal: define what you want to test

---

## 🔹 Step 2 — Compare models

* Review routing metrics
* Identify best candidate

👉 Question:
Which model misroutes the least?

---

## 🔹 Step 3 — Select your model

* Accept suggested model or override

👉 Question:
Do you agree with the selection?

---

## 🔹 Step 4 — Evaluate routing quality

* Check misroute rate

👉 Question:
Is routing good enough for workflow use?

---

## 🔹 Step 5 — Evaluate safety

* Apply confidence rule

👉 Question:
What can be safely automated?

---

## 🔹 Step 6 — Evaluate cost and speed

* Review cost and latency

👉 Question:
Is this viable at scale?

---

## 🔹 Step 7 — Review results

* Inspect summary table

👉 Question:
Where does the system fail?

---

## 🔹 Step 8 — Improve and re-test

* Change model / prompt / parameters
* Re-run evaluation

👉 Question:
Did results improve?

---

## 🔹 Step 9 — Make a decision

Write your recommendation.

👉 Question:
Should we proceed, improve, or stop?

---

# Final takeaway

Notebook A is not about finding the best model.

It is about learning to:

> **evaluate, improve, and make decisions under uncertainty**

That is the core skill behind real AI product development.
