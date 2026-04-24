Absolutely — what you need is not a generic product spec, but a **Colab practice notebook spec** for Module 3.3: a student-facing structure with short section text and only the minimum visible code they need to run. The notebook should help them evaluate whether Candlekeep’s routing prototype is strong enough for the next MVP step, in the context of 10,000+ monthly messages, 60 hours/week of manual work, and a 4.2-hour first response baseline.

Below is a **Notebook A spec written in that format**.

---

# Notebook A — Core MVP Evaluation (Colab practice spec)

## Notebook goal

This notebook helps you evaluate whether Candlekeep’s routing prototype is strong enough to justify the next MVP step.

You will:

* compare a few candidate models on routing
* choose the most promising one
* test routing quality, safety, cost, and speed
* make a recommendation
* optionally try improvements and rerun

This is not a production evaluation system. It is a practical MVP decision tool.

---

# 1. Title + intro section

## Section title

**Module 3.3 — Notebook A: Core MVP Evaluation**

## Section text

In this notebook, you will test whether Candlekeep’s routing prototype is strong enough to move forward.

Candlekeep handles **10,000+ customer messages per month**, with **60 hours of manual processing per week** and an average **4.2-hour first response time**. The opportunity is clear: if routing can be automated safely and affordably, the team can reduce manual effort and improve response handling.

Your job here is not to prove that the whole system works perfectly. Your job is to decide whether this routing setup is good enough to:

* continue,
* improve and re-test,
* or stop.

---

# 2. What you will evaluate

## Section title

**What this notebook evaluates**

## Section text

This notebook focuses on four questions:

1. **Routing quality** — does the model send requests to the right department often enough to be useful?
2. **Safety** — if we only automate the safest-looking cases, how risky is that?
3. **Cost** — is the estimated cost acceptable at Candlekeep scale?
4. **Speed** — is the system fast enough to support a better workflow?

You will first compare candidate models, then evaluate the selected setup in more detail.

---

# 3. Setup

## Section title

**Step 1 — Configure the experiment**

## Section text

Start by setting your API key and the main experiment settings.

You can:

* choose which models to compare
* adjust temperature and a few other parameters
* use the default routing prompt or try your own version
* set the monthly message volume for cost projection

## User-facing code

```python
import os

TOKENFACTORY_API_KEY = "YOUR_TF_API_KEY"

CANDIDATE_MODELS = [
    "deepseek-ai/DeepSeek-R1-0528",
    "zai-org/GLM-5",
    "openai/gpt-oss-120b",
]

TEMPERATURE = 0.0
MONTHLY_MESSAGES = 20_000

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
```

---

# 4. Load dataset

## Section title

**Step 2 — Load the evaluation dataset**

## Section text

Now load the fixed evaluation dataset.

This dataset represents labeled customer requests with gold routing targets. We use it to measure how the routing prototype performs beyond a few hand-picked examples.

## User-facing code

```python
df = load_and_validate_dataset(DATASET_URL)
print(f"Rows loaded: {len(df)}")
df.head(3)
```

Optional small inspection cell:

```python
df["[Agent] Routing to Department"].value_counts()
```

---

# 5. Model selection

## Section title

**Step 3 — Compare candidate models**

## Section text

Before testing safety, cost, and speed in detail, first identify the most promising routing candidate.

For simplicity, we select the model with the **lowest misroute rate** on a sample of the dataset.

If two models are close, use:

* department accuracy,
* then cost,
* then latency

as secondary tie-breakers.

## User-facing code

```python
client = create_client(api_key=TOKENFACTORY_API_KEY)

model_runs = await run_model_comparison_async(
    df=df.sample(frac=0.3, random_state=42),
    models=CANDIDATE_MODELS,
    client=client,
    system_prompt=SYSTEM_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

## User-facing code

Show the comparison table:

```python
comparison_df = build_comparison_table(model_runs)
comparison_df
```

Optional plot:

```python
plot_quality_vs_cost(comparison_df)
```

## Section text

Review the comparison table. Which model has the lowest misroute rate?

The notebook can suggest a model automatically, but you may override that choice if you have a clear reason.

## User-facing code

```python
selected_model = select_best_model(model_runs)
print("Suggested model:", selected_model)
```

Optional override:

```python
MODEL_TO_EVALUATE = selected_model  # change manually if you want
```

---

# 6. Run full evaluation

## Section title

**Step 4 — Run full evaluation on the selected model**

## Section text

Now run the full evaluation on the model you selected.

This will generate predictions for the full dataset and compute the metrics used in the MVP decision.

## User-facing code

```python
selected_run = evaluate_model_on_dataframe(
    df=df,
    model=MODEL_TO_EVALUATE,
    client=client,
    system_prompt=SYSTEM_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

---

# 7. Routing quality

## Section title

**Step 5 — Check routing quality**

## Section text

Routing quality is the primary decision area in this notebook.

If the system sends too many requests to the wrong department, it increases friction instead of reducing it.

Focus on:

* **Misroute Rate** — primary metric
* **Department Accuracy** — supporting metric
* **Category Accuracy** — diagnostic only

## User-facing code

```python
qm = selected_run.quality_metrics

print("Misroute Rate:", round(qm["misroute_rate"], 4))
print("Department Accuracy:", round(qm["department_accuracy"], 4))
print("Category Accuracy:", round(qm["category_accuracy"], 4))
```

## Section text

Use this as your first decision check:

* Is misroute rate low enough to support workflow use?
* If not, is it close enough to justify further improvement?

For Candlekeep, a useful working target is roughly **≤10% misroutes** for routing to be considered strong enough for the next step. That aligns with the company’s routing and escalation goals.

---

# 8. Safety under review rule

## Section title

**Step 6 — Check safety under a simple review rule**

## Section text

A model can be useful without being safe to trust on every case.

Here we apply a simple rule:

* **High confidence** → candidate for auto-route
* **Everything else** → human review

This gives a practical MVP view of safety.

## User-facing code

```python
sm = compute_safety_metrics(df, selected_run.predictions)

print("Auto-route Coverage:", round(sm["auto_route_coverage"], 4))
print("Auto-route Precision:", round(sm["auto_route_precision"], 4))
print("Unsafe Auto-route Rate:", round(sm["unsafe_auto_route_rate"], 4))
print("Manual Review Rate:", round(sm["manual_review_rate"], 4))
```

Optional plot:

```python
plot_confidence_buckets(selected_run.predictions)
```

## Section text

Interpret the results:

* What share of requests could be automated now?
* How risky is that automated slice?
* Is confidence helping enough to support selective automation?

---

# 9. Cost

## Section title

**Step 7 — Check cost at Candlekeep scale**

## Section text

Even a technically good prototype may not be viable if the expected cost is too high.

Candlekeep already has a meaningful manual cost baseline and delayed-response cost, so cost per message matters as part of the MVP decision.

## User-facing code

```python
cm = selected_run.cost_metrics

print("Cost per Message (USD):", cm["cost_per_message_usd"])
print("Monthly Cost (USD):", cm["monthly_cost_usd"])
print("Cost Source:", cm["cost_source"])
```

## Section text

Interpret the result:

* Is the prototype affordable enough to justify further MVP work?
* Is the cost measured from token usage, or estimated?

---

# 10. Speed

## Section title

**Step 8 — Check speed feasibility**

## Section text

This notebook does not prove end-to-end workflow improvement yet. It only checks whether the system is technically fast enough to support a better process.

Focus on:

* **Median latency**
* optionally **p95 latency**

## User-facing code

```python
lm = selected_run.latency_metrics

print("Median Latency (ms):", lm["median_latency_ms"])
print("p95 Latency (ms):", lm["p95_latency_ms"])
print("Latency Source:", lm["latency_source"])
```

## Section text

Interpret the result:

* Is latency low enough to support a better workflow?
* Was latency measured in this run, or is it unknown?

Candlekeep’s broader product goal is to move from a 4.2-hour first response toward near-instant acknowledgment, so technical speed is an important feasibility check.

---

# 11. Results summary

## Section title

**Step 9 — Review the full evaluation summary**

## Section text

Now combine the results across all four dimensions.

## User-facing code

```python
decision = evaluate_decision(
    quality_metrics=selected_run.quality_metrics,
    safety_metrics=sm,
    cost_metrics=cm,
    latency_metrics=lm,
    thresholds=Thresholds(),
)

decision.to_dict()
```

Optional cleaner display cell:

```python
for dim in decision.dimensions:
    print(dim.name, dim.status, dim.value)
print("Recommendation:", decision.recommendation)
```

## Section text

At this point, you should be able to answer:

* Is routing strong enough?
* Is the automated slice safe enough?
* Is cost acceptable?
* Is speed acceptable?
* What should happen next?

---

# 12. Experiment and improve

## Section title

**Step 10 — Try improvements and re-run**

## Section text

Your first result is not your final result.

This is an important part of the exercise: try to improve the setup and see whether the metrics change.

You can modify:

* model choice
* prompt instructions
* temperature or other parameters

Then re-run the same evaluation flow and compare the results.

## User-facing code

Example: change one thing and rerun.

```python
MODEL_TO_EVALUATE = "openai/gpt-oss-120b"
TEMPERATURE = 0.2

selected_run = evaluate_model_on_dataframe(
    df=df,
    model=MODEL_TO_EVALUATE,
    client=client,
    system_prompt=SYSTEM_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

Or try a modified prompt:

```python
SYSTEM_PROMPT = """<your revised routing prompt here>"""
```

## Section text

Use this part of the notebook to learn:

* which changes actually improve routing
* whether safety improves or degrades
* whether better quality comes with higher cost or latency

The goal is not to maximize one metric in isolation. The goal is to understand trade-offs.

---

# 13. Final decision note

## Section title

**Step 11 — Write your recommendation**

## Section text

Write a short recommendation based on the evidence you generated.

There is no single correct answer. What matters is that your recommendation is justified by the results.

## User-facing template

```text
Decision: [Stop / Improve and re-test / Proceed with guardrails / Proceed to MVP workflow]

Reason:
- Routing:
- Safety:
- Cost:
- Speed:
- Next step:
```

---

# 14. Minimal user-facing code inventory

This is the intended visible code footprint for students.

## Core cells

```python
client = create_client(api_key=TOKENFACTORY_API_KEY)
```

```python
df = load_and_validate_dataset(DATASET_URL)
```

```python
model_runs = await run_model_comparison_async(
    df=df.sample(frac=0.3, random_state=42),
    models=CANDIDATE_MODELS,
    client=client,
    system_prompt=SYSTEM_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

```python
comparison_df = build_comparison_table(model_runs)
comparison_df
```

```python
selected_model = select_best_model(model_runs)
```

```python
selected_run = evaluate_model_on_dataframe(
    df=df,
    model=MODEL_TO_EVALUATE,
    client=client,
    system_prompt=SYSTEM_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

```python
sm = compute_safety_metrics(df, selected_run.predictions)
```

```python
decision = evaluate_decision(
    quality_metrics=selected_run.quality_metrics,
    safety_metrics=sm,
    cost_metrics=selected_run.cost_metrics,
    latency_metrics=selected_run.latency_metrics,
    thresholds=Thresholds(),
)
```

That should be enough for the student-facing notebook.

---

# 15. Implementation notes (hidden from students)

The backing `.py` / library layer may still:

* save `predictions.csv`
* save `run_meta.json`
* save `summary.json`
* save `decision_summary.json`
* save `comparison_table.csv`
* save optional plots and coverage diagnostics

Those are implementation artifacts, not part of the core student experience.

---

# 16. Expected notebook tone

The notebook should feel like:

* a guided evaluation
* a practical experiment
* a product decision exercise

It should not feel like:

* a benchmark harness
* an internal engineering debug tool
* a giant metrics dashboard
