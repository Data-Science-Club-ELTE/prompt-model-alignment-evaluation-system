import click
import json
import sys
from runner import generate, run_multiple
from scorer import (
    score_instruction_following,
    score_format_match,
    score_semantic_accuracy,
    score_consistency,
    score_edge_cases,
    compute_final_score,
    score_label,
    WEIGHTS,
)
from sensitivity import test_sensitivity


def print_report(prompt: str, scores: dict, final: float, sensitivity_result: dict | None = None):
    label = score_label(final)
    width = 60

    print("\n" + "=" * width)
    print("  PROMPT-MODEL ALIGNMENT REPORT")
    print("=" * width)
    print(f"  Final Score : {final}%  —  {label}")
    print("=" * width)

    print("\n  DIMENSION BREAKDOWN")
    print("  " + "-" * (width - 2))

    dim_labels = {
        "instruction_following": "Instruction Following",
        "format_match":          "Output Format Match",
        "semantic_accuracy":     "Semantic Accuracy",
        "consistency":           "Consistency",
        "edge_cases":            "Edge Case Handling",
    }

    for key, label_name in dim_labels.items():
        s = scores[key]
        pct = round(s["score"] * 100)
        weight_pct = int(WEIGHTS[key] * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\n  {label_name} (weight: {weight_pct}%)")
        print(f"  [{bar}] {pct}%")
        print(f"  → {s['note']}")

    # ── Sensitivity section (optional) ──
    if sensitivity_result:
        print("\n" + "=" * width)
        print("\n  SENSITIVITY / ROBUSTNESS")
        print("  " + "-" * (width - 2))

        sens = sensitivity_result["sensitivity_score"]
        sens_pct = round(sens * 100)
        bar = "█" * (sens_pct // 5) + "░" * (20 - sens_pct // 5)
        robustness = "Robust ✅" if sens < 0.3 else ("Moderate ⚠️" if sens < 0.6 else "Fragile ❌")

        print(f"\n  Sensitivity Score : {sens_pct}%  —  {robustness}")
        print(f"  [{bar}] {sens_pct}%")
        print(f"  (Higher = more fragile. Lower = more robust to rewording)")

        most_stable   = sensitivity_result.get("most_stable_variation")
        least_stable  = sensitivity_result.get("least_stable_variation")

        if most_stable:
            print(f"\n  Most stable variation  ({round(most_stable['similarity_score']*100)}% similar):")
            print(f"  → \"{most_stable['variation_text'][:80]}{'...' if len(most_stable['variation_text']) > 80 else ''}\"")
        if least_stable:
            print(f"\n  Least stable variation ({round(least_stable['similarity_score']*100)}% similar):")
            print(f"  → \"{least_stable['variation_text'][:80]}{'...' if len(least_stable['variation_text']) > 80 else ''}\"")

    print("\n" + "=" * width)

    # ── Summary ──
    print("\n  SUMMARY")
    print("  " + "-" * (width - 2))
    weak = [k for k in scores if scores[k]["score"] < 0.65]
    if not weak:
        print("  No major issues detected. Prompt is well aligned with this model.")
    else:
        for k in weak:
            print(f"  ⚠️  {dim_labels[k]} is low — {scores[k]['note']}")

    if sensitivity_result and sensitivity_result["sensitivity_score"] >= 0.6:
        print("  ⚠️  Prompt is fragile — small wording changes produce very different outputs.")

    print("\n" + "=" * width + "\n")


@click.command()
@click.option("--prompt",      "-p", required=True,       help="Path to your prompt .txt file")
@click.option("--expected",    "-e", default=None,        help="Path to expected output .txt file (optional)")
@click.option("--model",       "-m", default="llama3.1:8b", help="Ollama model name")
@click.option("--runs",        "-r", default=3,           help="Number of runs for consistency check")
@click.option("--sensitivity", "-s", is_flag=True,        help="Run sensitivity/robustness test")
@click.option("--json-out",    "-j", is_flag=True,        help="Also save results as report.json")
def main(prompt, expected, model, runs, sensitivity, json_out):

    # ── Load files ──
    try:
        with open(prompt, "r") as f:
            prompt_text = f.read().strip()
    except FileNotFoundError:
        print(f"\n❌ Error: Prompt file not found: '{prompt}'")
        print("   Make sure the file exists and the path is correct.\n")
        sys.exit(1)

    expected_text = None
    if expected:
        try:
            with open(expected, "r") as f:
                expected_text = f.read().strip()
        except FileNotFoundError:
            print(f"\n⚠️  Warning: Expected file not found: '{expected}' — semantic and format scoring will be skipped.\n")

    print(f"\nModel  : {model}")
    print(f"Prompt : {prompt}")
    print(f"Runs   : {runs}\n")

    # ── Step 1: Run the prompt ──
    print("Step 1/4 — Running prompt against model...")
    try:
        outputs = run_multiple(prompt_text, model, n=runs)
    except Exception as e:
        print(f"\n❌ Error: Could not reach Ollama — {e}")
        print("   Make sure Ollama is running. Try: ollama serve")
        print(f"   And that the model is pulled. Try: ollama pull {model}\n")
        sys.exit(1)

    if not outputs or not any(outputs):
        print(f"\n❌ Error: Model returned empty output. Check that '{model}' is working correctly.\n")
        sys.exit(1)

    primary_output = outputs[0]
    print(f"  Primary output: {primary_output[:80]}{'...' if len(primary_output) > 80 else ''}")

    # ── Step 2: Score all dimensions ──
    print("\nStep 2/4 — Scoring...")

    print("  → Instruction following (LLM judge)...")
    instr_score = score_instruction_following(prompt_text, primary_output, model)

    print("  → Format match...")
    fmt_score = score_format_match(expected_text, primary_output)

    print("  → Semantic accuracy (cross-encoder)...")
    sem_score = score_semantic_accuracy(expected_text, primary_output)

    print("  → Consistency (embedding similarity)...")
    cons_score = score_consistency(outputs)

    print("  → Edge cases...")
    edge_score = score_edge_cases([])

    scores = {
        "instruction_following": instr_score,
        "format_match":          fmt_score,
        "semantic_accuracy":     sem_score,
        "consistency":           cons_score,
        "edge_cases":            edge_score,
    }

    final = compute_final_score(scores)

    # ── Step 3: Sensitivity test (optional) ──
    sensitivity_result = None
    if sensitivity:
        print("\nStep 3/4 — Running sensitivity test (this makes several extra API calls)...")
        try:
            sensitivity_result = test_sensitivity(prompt_text, model, num_variations=5)
            print(f"  Sensitivity score: {round(sensitivity_result['sensitivity_score'] * 100)}%")
        except Exception as e:
            print(f"  Warning: Sensitivity test failed ({e}) — skipping.")
    else:
        print("\nStep 3/4 — Skipped (use --sensitivity to enable)")

    # ── Step 4: Report ──
    print("\nStep 4/4 — Building report...")
    print_report(prompt_text, scores, final, sensitivity_result)

    # ── Optional JSON output ──
    if json_out:
        result = {
            "model":       model,
            "final_score": final,
            "dimensions":  scores,
            "outputs":     outputs,
        }
        if sensitivity_result:
            result["sensitivity"] = sensitivity_result
        with open("report.json", "w") as f:
            json.dump(result, f, indent=2)
        print("  Saved to report.json\n")


if __name__ == "__main__":
    main()