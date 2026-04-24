# Notebook B — Optional Advanced Improvement (Colab practice spec)

## Notebook goal

This notebook helps you answer a follow-up question:

> **If the baseline is not strong enough, what practical changes can improve it?**

You will:

* reuse or rerun the baseline as a reference
* improve the routing prompt
* test few-shot examples
* compare results
* inspect richer diagnostics
* make an improvement-oriented recommendation

This notebook is optional.
It is meant for learners who want to go beyond the core MVP decision and explore how performance can improve.

---

# 1. Title + intro section

## Section title

**Module 3.3 — Notebook B: Improve the baseline**

## Section text

In Notebook A, you evaluated whether Candlekeep’s routing prototype was strong enough for the next MVP step.

In this notebook, you take the next logical step:

> If the baseline is weak, can we improve it in practical ways?

You will test a few realistic levers:

* better prompt design
* few-shot examples
* optional model comparison
* deeper error analysis

The goal is not to chase perfect scores. The goal is to understand what actually improves routing quality, what trade-offs appear, and whether the prototype is moving closer to MVP viability.

---

# 2. What this notebook explores

## Section title

**What this notebook explores**

## Section text

This notebook focuses on improvement, not initial validation.

You will explore four questions:

1. **Prompt refinement** — can clearer instructions improve routing quality?
2. **Few-shot learning** — do example cases help the model route better?
3. **Trade-offs** — do improvements change cost or latency?
4. **Diagnostics** — where does the model still fail, and why?

Notebook B assumes you have already completed Notebook A or at least have a baseline result to compare against.

---

# 3. Setup

## Section title

**Step 1 — Configure the improvement experiment**

## Section text

Start by setting the same API key and core settings you used in Notebook A.

You can:

* reuse the same candidate models
* start with the same baseline prompt
* define a revised prompt
* configure few-shot settings
* choose whether to compare models again or focus on one

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

BASELINE_PROMPT = DEFAULT_SYSTEM_PROMPT
IMPROVED_PROMPT = DEFAULT_ROUTING_SYSTEM_PROMPT
```

Optional few-shot settings:

```python
N_SHOT_VALUES = [0, 2, 4, 8, 16]
```

---

# 4. Load dataset and baseline reference

## Section title

**Step 2 — Load the dataset and baseline reference**

## Section text

Load the same evaluation dataset you used in Notebook A.

Then either:

* rerun the baseline configuration, or
* load baseline results if they are already available

This gives you a reference point for measuring improvement.

## User-facing code

```python
df = load_and_validate_dataset(DATASET_URL)
print(f"Rows loaded: {len(df)}")
```

Option A — rerun baseline:

```python
client = create_client(api_key=TOKENFACTORY_API_KEY)

baseline_run = evaluate_model_on_dataframe(
    df=df,
    model="deepseek-ai/DeepSeek-R1-0528",
    client=client,
    system_prompt=BASELINE_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

## Section text

Keep the baseline metrics visible. You will compare every improvement against this reference.

---

# 5. Baseline reference summary

## Section title

**Step 3 — Review the baseline reference**

## Section text

Before changing anything, confirm the baseline result.

Focus on:

* misroute rate
* department accuracy
* safety metrics
* cost
* latency

## User-facing code

```python
baseline_qm = baseline_run.quality_metrics
baseline_sm = compute_safety_metrics(df, baseline_run.predictions)
baseline_cm = baseline_run.cost_metrics
baseline_lm = baseline_run.latency_metrics

print("Baseline misroute rate:", round(baseline_qm["misroute_rate"], 4))
print("Baseline department accuracy:", round(baseline_qm["department_accuracy"], 4))
print("Baseline unsafe auto-route rate:", round(baseline_sm["unsafe_auto_route_rate"], 4))
print("Baseline cost/message:", baseline_cm["cost_per_message_usd"])
print("Baseline median latency:", baseline_lm["median_latency_ms"])
```

## Section text

This is your reference point. Improvement only matters if you can compare it to something concrete.

---

# 6. Prompt refinement experiment

## Section title

**Step 4 — Try a stronger routing prompt**

## Section text

One of the simplest ways to improve a weak baseline is to make the routing prompt clearer.

In this step, you will test a revised prompt with:

* clearer Candlekeep context
* more explicit department definitions
* sharper routing instructions
* less distraction from non-routing tasks

## User-facing code

```python
improved_prompt_run = evaluate_model_on_dataframe(
    df=df,
    model="deepseek-ai/DeepSeek-R1-0528",
    client=client,
    system_prompt=IMPROVED_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

## User-facing code

Compare baseline vs improved prompt:

```python
compare_prompt_runs(
    baseline_run,
    improved_prompt_run,
)
```

If no helper exists yet, minimal visible code can be:

```python
print("Baseline misroute:", baseline_run.quality_metrics["misroute_rate"])
print("Improved misroute:", improved_prompt_run.quality_metrics["misroute_rate"])

print("Baseline dept acc:", baseline_run.quality_metrics["department_accuracy"])
print("Improved dept acc:", improved_prompt_run.quality_metrics["department_accuracy"])
```

## Section text

Ask:

* Did misroute rate improve?
* Did safety improve?
* Did cost or latency change meaningfully?

This is the first practical lesson in improvement: prompt design can materially change results.

---

# 7. Few-shot experiment

## Section title

**Step 5 — Test few-shot examples**

## Section text

If prompt refinement helps but is not enough, the next lever is few-shot learning.

In this step, you test whether giving the model a few labeled examples improves routing quality.

Important:

* keep the evaluation slice fixed
* do not let examples overlap with the evaluation set

## User-facing code

```python
few_shot_slice = prepare_few_shot_ablation_slice(
    df=df,
    eval_frac=0.68,
    random_state=42,
)
```

```python
few_shot_results = await run_few_shot_ablation_async(
    df=few_shot_slice.eval_df,
    few_shot_pool=few_shot_slice.few_shot_pool,
    model="deepseek-ai/DeepSeek-R1-0528",
    client=client,
    system_prompt=IMPROVED_PROMPT,
    n_shot_values=N_SHOT_VALUES,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

## User-facing code

Show results:

```python
display_few_shot_results(few_shot_results)
```

Or minimally:

```python
for result in few_shot_results:
    print(
        result["n_shot"],
        result["quality_metrics"]["misroute_rate"],
        result["quality_metrics"]["department_accuracy"],
    )
```

## Section text

Ask:

* Does routing improve with examples?
* Where do gains flatten out?
* Do more examples increase cost or latency?

This helps you see whether few-shot learning is worth the extra complexity.

---

# 8. Few-shot trade-off charts

## Section title

**Step 6 — Visualize few-shot trade-offs**

## Section text

Charts help you see whether improvement is worth the added complexity.

Use this section to compare:

* accuracy improvement
* misroute reduction
* cost changes
* latency changes

## User-facing code

```python
plot_metrics_for_n_shot(few_shot_results)
```

```python
plot_latency_for_n_shots(few_shot_results)
```

## Section text

Interpret the curve:

* Is there a meaningful gain from 0 → 2 → 4 → 8 shots?
* Does the gain level off?
* At what point do extra examples stop being worth it?

---

# 9. Optional model comparison with improved prompt

## Section title

**Step 7 — Compare models again with the improved prompt (optional)**

## Section text

Sometimes a weak baseline improves enough with a better prompt that the relative ranking of models changes.

If you want, rerun model comparison with the improved prompt.

## User-facing code

```python
improved_model_runs = await run_model_comparison_async(
    df=df.sample(frac=0.3, random_state=42),
    models=CANDIDATE_MODELS,
    client=client,
    system_prompt=IMPROVED_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

```python
improved_comparison_df = build_model_comparison_dataframe(df, improved_model_runs)
improved_comparison_df
```

Optional plot:

```python
plot_quality_vs_cost(improved_comparison_df)
```

## Section text

Use this only if it helps answer:

* is there now a stronger candidate model?
* is the quality gain worth the cost or latency trade-off?

This section is optional because the main purpose of Notebook B is improvement, not endless benchmarking.

---

# 10. Richer diagnostics

## Section title

**Step 8 — Inspect where the model still fails**

## Section text

Now that you have improved the setup, inspect the remaining failure modes.

This is where you move from “Did it improve?” to:

> “Why is it still failing?”

Useful diagnostics include:

* confusion matrices
* top failure examples
* confidence vs correctness
* class-level error patterns

## User-facing code

```python
display_evaluation_results(improved_prompt_run)
```

Optional examples table:

```python
show_top_errors(
    df=df,
    predictions=improved_prompt_run.predictions,
    n=10,
)
```

## Section text

Look for patterns:

* which departments are still confused?
* which request types are hardest?
* does confidence help separate easy from risky cases?

---

# 11. Compare baseline vs improved setup

## Section title

**Step 9 — Summarize the improvement**

## Section text

Now compare baseline and improved configurations side by side.

The goal is to make the improvement visible and interpretable.

## User-facing code

```python
baseline_vs_improved = compare_runs(
    baseline_run,
    improved_prompt_run,
    include_safety=True,
    include_cost=True,
    include_latency=True,
)
baseline_vs_improved
```

If no helper exists yet, minimal visible code:

```python
print("Baseline misroute:", baseline_run.quality_metrics["misroute_rate"])
print("Improved misroute:", improved_prompt_run.quality_metrics["misroute_rate"])

print("Baseline dept acc:", baseline_run.quality_metrics["department_accuracy"])
print("Improved dept acc:", improved_prompt_run.quality_metrics["department_accuracy"])
```

## Section text

Ask:

* Is the gain substantial?
* Is the prototype now close enough to the MVP threshold?
* What still blocks rollout?

---

# 12. Improvement recommendation

## Section title

**Step 10 — Write an improvement recommendation**

## Section text

This notebook does not replace the core MVP decision from Notebook A. Instead, it helps you answer a second question:

> Given what we tried, what should we improve next?

Possible recommendations:

* keep the improved prompt and continue
* proceed with assist-first workflow only
* add more examples and retest
* test more candidate models
* collect more data before continuing

## User-facing template

```text
Improvement recommendation:

What improved:
- ...

What is still weak:
- ...

Trade-offs observed:
- ...

Suggested next step:
- ...
```

---

# 13. Minimal user-facing code inventory

This is the intended visible code footprint for students.

## Core cells

```python
client = create_client(api_key=TOKENFACTORY_API_KEY)
```

```python
df = load_and_validate_dataset(DATASET_URL)
```

```python
baseline_run = evaluate_model_on_dataframe(
    df=df,
    model=MODEL_TO_EVALUATE,
    client=client,
    system_prompt=BASELINE_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

```python
improved_prompt_run = evaluate_model_on_dataframe(
    df=df,
    model=MODEL_TO_EVALUATE,
    client=client,
    system_prompt=IMPROVED_PROMPT,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

```python
few_shot_slice = prepare_few_shot_ablation_slice(df=df, eval_frac=0.68, random_state=42)
```

```python
few_shot_results = await run_few_shot_ablation_async(
    df=few_shot_slice.eval_df,
    few_shot_pool=few_shot_slice.few_shot_pool,
    model=MODEL_TO_EVALUATE,
    client=client,
    system_prompt=IMPROVED_PROMPT,
    n_shot_values=N_SHOT_VALUES,
    temperature=TEMPERATURE,
    monthly_messages=MONTHLY_MESSAGES,
    use_progress=True,
)
```

```python
display_evaluation_results(improved_prompt_run)
```

That should be enough for the student-facing notebook.

---

# 14. Hidden implementation notes

The library / script layer may still:

* save intermediate runs
* save raw predictions
* save plots
* save metrics JSON
* save ablation outputs

Those are implementation artifacts, not part of the core student experience.

---

# 15. Expected notebook tone

Notebook B should feel like:

* an improvement lab
* a practical follow-up to Notebook A
* a structured way to test better setups

It should not feel like:

* a full research environment
* an unbounded benchmarking notebook
* a production optimization system

If you want, I can next write this as a **cell-by-cell Colab notebook outline**, matching the exact markdown and code cell order.
