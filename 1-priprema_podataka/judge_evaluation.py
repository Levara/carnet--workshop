#!/usr/bin/env python3
"""
LLM-as-a-Judge evaluation script.

PURPOSE:
    Automated quality assessment of LLM responses using a powerful judge model.
    Provides objective, consistent evaluation of model answers with detailed feedback.

WHAT IT DOES:
    - Loads previous evaluation results from evaluate_local_model.py or evaluate_openrouter_model.py
    - Uses a powerful judge model (Gemini/Claude) to score each answer
    - Evaluates on multiple criteria: accuracy, completeness, relevance, clarity, grounding
    - Assigns numerical scores (0-10) and verdicts (CORRECT/PARTIALLY_CORRECT/INCORRECT)
    - Generates both JSON results and visual HTML reports
    - Supports two context modes: specific segments (RAG-style) or full document

CONFIGURATION (edit these variables in the script if needed):
    - OPENROUTER_API_KEY: Required for judge model access (reads from environment or hardcoded)
    - JUDGE_MODEL: Default is "google/gemini-2.5-flash-preview-09-2025"
    - JUDGE_TEMPERATURE: 0.2 (low for consistent judgments)
    - JUDGE_MAX_TOKENS: 2048

EVALUATION CRITERIA (each scored 0-10):
    1. Factual Accuracy    : Correctness based on context
    2. Completeness        : Coverage of all question aspects
    3. Relevance           : Directly answers the question
    4. Clarity & Structure : Well-organized, correct Croatian language
    5. Grounding in Context: Derived from context without hallucinations
    6. Overall Score       : Weighted combination of above

VERDICT GUIDELINES:
    - CORRECT            : Overall score ≥ 8.0, all key facts accurate
    - PARTIALLY_CORRECT  : Score 5.0-7.9, mostly correct with minor issues
    - INCORRECT          : Score < 5.0, major errors or hallucinations

USAGE:
    # Interactive mode - select from available evaluation results
    ./judge_evaluation.py

    # Use full FAQ document as context (more lenient evaluation)
    ./judge_evaluation.py --full-context

CONTEXT MODES:
    Default (specific segments):
        - Judge only sees the relevant context segments
        - More strict evaluation
        - Best for verifying answer is grounded in specific passages

    Full context (--full-context):
        - Judge sees entire FAQ document
        - More lenient evaluation
        - Use when evaluated model had access to full document

COMMAND LINE OPTIONS:
    --full-context, -f  : Use full FAQ document as context instead of specific segments

REQUIREMENTS:
    - OPENROUTER_API_KEY environment variable or hardcoded in script
    - Previous evaluation results in results/ directory (from evaluate_*_model.py)
    - Data files:
        * data/loomen_faq.hard_questions.json (for context segments)
        * data/loomen_faq.md (for full context mode)
    - HTML template: assets/judgment_template.html

OUTPUT:
    - Console: Progress display with scores and verdicts
    - JSON file: results/judgment_TIMESTAMP_MODELNAME.json
        Contains: all judgments with scores, feedback, reasoning
    - HTML report: results/judgment_TIMESTAMP_MODELNAME.html
        Visual report with color-coded cards, expandable details
    - Summary statistics: average scores, verdict breakdown, timing

HTML REPORT FEATURES:
    - Color-coded question cards (green/yellow/red by verdict)
    - Expandable details for each question
    - Side-by-side model answer and reference answer
    - Detailed scores breakdown (all 5 criteria + overall)
    - Strengths and weaknesses lists
    - Missing information identification
    - Hallucination detection
    - Reasoning explanations from judge

INTERACTIVE MENU:
    When you run the script, it displays all available evaluation results:
    - Sorted by timestamp (newest first)
    - Shows model name, question count, file size
    - Select by number or 'q' to quit

EXAMPLE WORKFLOW:
    1. Run model evaluation:
        ./evaluate_openrouter_model.py --model anthropic/claude-3.5-sonnet
    2. Run judge evaluation:
        ./judge_evaluation.py
    3. Select the evaluation file from the menu
    4. Wait for judging to complete
    5. Open the HTML report in a browser:
        firefox results/judgment_TIMESTAMP_MODEL.html
    6. Review detailed scores and feedback

INTERPRETING RESULTS:
    - High scores (8-10): Model performs well on this question
    - Medium scores (5-7): Partial understanding, missing details
    - Low scores (0-4): Significant issues, hallucinations, or wrong answer
    - Check "Hallucinations" section for made-up information
    - Check "Missing Information" for important omitted details
    - Read "Reasoning" for judge's overall assessment

USE CASES:
    - Compare different models on the same questions
    - Identify model strengths and weaknesses
    - Track improvements across model versions
    - Generate reports for stakeholders
    - Debug problematic questions or prompts
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from openai import OpenAI


# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# OPENROUTER_API_KEY = "hardcode key here if not exported through the env variable"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Judge model - using a smart model via OpenRouter
# Options: "anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo", "google/gemini-pro-1.5"
# JUDGE_MODEL = "anthropic/claude-3.5-sonnet"
JUDGE_MODEL = "google/gemini-2.5-flash-preview-09-2025"
JUDGE_TEMPERATURE = 0.2  # Low temperature for consistent judgments
JUDGE_MAX_TOKENS = 2048

# Paths
DATA_DIR = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'results'
PROMPTS_DIR = Path(__file__).parent / 'prompts'
ASSETS_DIR = Path(__file__).parent / 'assets'

QUESTIONS_PATH = DATA_DIR / 'loomen_faq.hard_questions.json'
FAQ_DOCUMENT_PATH = DATA_DIR / 'loomen_faq.md'
HTML_TEMPLATE_PATH = ASSETS_DIR / 'judgment_template.html'

# Judge prompt template
JUDGE_PROMPT = """# Task: Evaluate LLM Answer Quality

You are an expert evaluator assessing the quality of an LLM's answer to a question about the Loomen learning management system used by Croatian educational institutions.

The LLM was given a question and context from documentation. You must evaluate if the answer:
1. Is factually accurate based on the provided context
2. Matches the expected reference answer
3. Doesn't hallucinate information beyond the context

## Evaluation Criteria

### 1. Factual Accuracy (0-10)
- Are all facts correct according to the context?
- Does it contradict the reference answer or context?
- Are technical details (URLs, procedures, settings) accurate?

### 2. Completeness (0-10)
- Does it address all parts of the question?
- Are all key points from the reference answer covered?
- Are important details from the context included?

### 3. Relevance (0-10)
- Does it directly answer the question?
- Is information from the context used appropriately?
- Is there irrelevant information?

### 4. Clarity and Structure (0-10)
- Is the answer well-organized and easy to understand?
- Is the Croatian language usage correct?
- For multi-step procedures, are steps clearly presented?

### 5. Grounding in Context (0-10)
- Is all information derived from the provided context?
- Are there hallucinations (facts not in context)?
- Does it avoid making up URLs, procedures, or details?

## Verdict Guidelines

- **CORRECT**: Overall score ≥ 8.0, all key facts accurate, comprehensive, well-grounded
- **PARTIALLY_CORRECT**: Score 5.0-7.9, mostly correct but missing details or minor issues
- **INCORRECT**: Score < 5.0, major errors, hallucinations, or doesn't answer the question

---

<context>
{CONTEXT_SEGMENTS}
</context>

<question>
{QUESTION}
</question>

<reference_answer>
{REFERENCE_ANSWER}
</reference_answer>

<model_answer>
{MODEL_ANSWER}
</model_answer>

---

Provide your evaluation in this exact JSON format:

```json
{{
  "scores": {{
    "factual_accuracy": <0-10>,
    "completeness": <0-10>,
    "relevance": <0-10>,
    "clarity": <0-10>,
    "grounding": <0-10>,
    "overall": <0-10>
  }},
  "verdict": "<CORRECT|PARTIALLY_CORRECT|INCORRECT>",
  "strengths": [
    "<strength 1>",
    "<strength 2>"
  ],
  "weaknesses": [
    "<weakness 1>",
    "<weakness 2>"
  ],
  "missing_information": [
    "<missing detail 1>"
  ],
  "hallucinations": [
    "<hallucinated detail 1>"
  ],
  "reasoning": "<2-3 sentence explanation>"
}}
```"""


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate LLM responses using LLM-as-a-Judge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use specific context segments (default)
  python judge_evaluation.py

  # Use full FAQ document as context
  python judge_evaluation.py --full-context

When to use --full-context:
  - When the evaluated model had access to the full document
  - To judge if the model could extract correct info from the full document
  - For more lenient evaluation (judge sees everything the model saw)

When to use default (specific segments):
  - When you want to verify the answer is grounded in the reference segments
  - For stricter evaluation (judge only sees the relevant parts)
  - To check if the model extracted the right information
        """
    )

    parser.add_argument(
        '--full-context', '-f',
        action='store_true',
        help='Use full FAQ document as context instead of specific segments'
    )

    return parser.parse_args()


def load_faq_document() -> str:
    """Load the full FAQ document."""
    with open(FAQ_DOCUMENT_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def list_result_files() -> List[Path]:
    """List all evaluation result files sorted by timestamp (newest first)."""
    if not RESULTS_DIR.exists():
        return []

    result_files = list(RESULTS_DIR.glob("evaluation_*.json"))
    # Sort by modification time, newest first
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return result_files


def display_results_menu(files: List[Path]) -> Optional[Path]:
    """Display menu of result files and let user select one."""
    if not files:
        print("No evaluation result files found in results/")
        return None

    print("\n" + "="*80)
    print("Available Evaluation Results (newest first):")
    print("="*80)

    for i, file in enumerate(files, 1):
        # Get file info
        mtime = datetime.fromtimestamp(file.stat().st_mtime)
        size_kb = file.stat().st_size / 1024

        # Try to load and show basic info
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = data.get('model', 'unknown')
                # Extract just the filename from full path
                if '/' in model_name:
                    model_name = Path(model_name).name
                num_results = len(data.get('results', []))

            print(f"\n{i}. {file.name}")
            print(f"   Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Model: {model_name}")
            print(f"   Questions: {num_results}")
            print(f"   Size: {size_kb:.1f} KB")
        except Exception as e:
            print(f"\n{i}. {file.name}")
            print(f"   Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Size: {size_kb:.1f} KB")
            print(f"   (Could not read file details: {e})")

    print("\n" + "="*80)

    while True:
        try:
            choice = input(f"\nSelect a file (1-{len(files)}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                return None

            index = int(choice) - 1
            if 0 <= index < len(files):
                return files[index]
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Please enter a valid number or 'q'")


def load_questions_with_context() -> Dict[str, Any]:
    """Load questions with context and reference answers."""
    with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Convert to dict keyed by question text for easy lookup
    questions_dict = {}
    for q in questions:
        questions_dict[q['question']] = {
            'answer': q['answer'],
            'context': q['context']
        }

    return questions_dict


def format_context_segments(context_segments: List[Dict[str, str]]) -> str:
    """Format context segments for the judge prompt."""
    formatted = []
    for i, segment in enumerate(context_segments, 1):
        location = segment['segment_location']
        text = segment['segment_text']
        formatted.append(f"Segment {i} - Location: \"{location}\"\n{text}")

    return "\n\n---\n\n".join(formatted)


def build_judge_prompt(question: str, reference_answer: str,
                       model_answer: str, context_segments: List[Dict[str, str]],
                       full_context: Optional[str] = None) -> str:
    """
    Build the complete judge prompt.

    Args:
        question: The question that was asked
        reference_answer: The expected answer
        model_answer: The model's actual answer
        context_segments: Specific context segments (ignored if full_context is provided)
        full_context: Optional full FAQ document to use as context

    Returns:
        Complete judge prompt
    """
    if full_context:
        # Use full document as context
        context_text = full_context
    else:
        # Use specific segments
        context_text = format_context_segments(context_segments)

    return JUDGE_PROMPT.format(
        CONTEXT_SEGMENTS=context_text,
        QUESTION=question,
        REFERENCE_ANSWER=reference_answer,
        MODEL_ANSWER=model_answer
    )


def query_judge(client: OpenAI, prompt: str, question_num: int, total: int) -> Dict[str, Any]:
    """Query the judge model via OpenRouter."""
    print(f"\n{'='*80}")
    print(f"Judging Question {question_num}/{total}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
        )

        elapsed_time = time.time() - start_time

        # Extract response
        judgment_text = response.choices[0].message.content

        # Try to parse JSON from the response
        # The model should return JSON, but it might be wrapped in markdown
        try:
            # Try to extract JSON if wrapped in code blocks
            if "```json" in judgment_text:
                start = judgment_text.find("```json") + 7
                end = judgment_text.rfind("```")
                json_text = judgment_text[start:end].strip()
            elif "```" in judgment_text:
                start = judgment_text.find("```") + 3
                end = judgment_text.rfind("```")
                json_text = judgment_text[start:end].strip()
            else:
                json_text = judgment_text

            judgment = json.loads(json_text)

            overall_score = judgment['scores']['overall']
            verdict = judgment['verdict']

            print(f"✓ Judgment received in {elapsed_time:.2f}s")
            print(f"  Overall Score: {overall_score}/10")
            print(f"  Verdict: {verdict}")

            return {
                'success': True,
                'judgment': judgment,
                'raw_response': judgment_text,
                'elapsed_time': elapsed_time,
                'error': None
            }

        except json.JSONDecodeError as e:
            print(f"⚠ Could not parse JSON from judge response: {e}")
            print(f"Raw response: {judgment_text[:200]}...")

            return {
                'success': False,
                'judgment': None,
                'raw_response': judgment_text,
                'elapsed_time': elapsed_time,
                'error': f"JSON parse error: {e}"
            }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Error after {elapsed_time:.2f}s: {e}")

        return {
            'success': False,
            'judgment': None,
            'raw_response': None,
            'elapsed_time': elapsed_time,
            'error': str(e)
        }


def generate_question_html(judgment: Dict[str, Any]) -> str:
    """Generate HTML for a single question card."""
    judge_result = judgment['judge_result']

    if not judge_result['success']:
        return f"""
        <div class="question-card">
            <div class="question-header incorrect">
                <span class="question-number">Q{judgment['question_number']}</span>
                <div class="question-title">{judgment['question']}</div>
                <div class="question-score">
                    <span style="color: #ef4444;">Error: {judge_result['error']}</span>
                </div>
            </div>
        </div>
        """

    j = judge_result['judgment']
    scores = j['scores']
    verdict = j['verdict']

    # Determine verdict class
    verdict_class = {
        'CORRECT': 'correct',
        'PARTIALLY_CORRECT': 'partial',
        'INCORRECT': 'incorrect'
    }.get(verdict, 'incorrect')

    # Generate scores HTML
    scores_html = f"""
    <div class="scores-grid">
        <div class="score-item">
            <div class="label">Factual Accuracy</div>
            <div class="score">{scores['factual_accuracy']}/10</div>
        </div>
        <div class="score-item">
            <div class="label">Completeness</div>
            <div class="score">{scores['completeness']}/10</div>
        </div>
        <div class="score-item">
            <div class="label">Relevance</div>
            <div class="score">{scores['relevance']}/10</div>
        </div>
        <div class="score-item">
            <div class="label">Clarity</div>
            <div class="score">{scores['clarity']}/10</div>
        </div>
        <div class="score-item">
            <div class="label">Grounding</div>
            <div class="score">{scores['grounding']}/10</div>
        </div>
    </div>
    """

    # Generate lists
    def make_list(items, css_class):
        if not items:
            return "<p style='color: #9ca3af; font-style: italic;'>None</p>"
        items_html = '\n'.join(f"<li>{item}</li>" for item in items)
        return f"<ul class='list-items {css_class}'>{items_html}</ul>"

    strengths_html = make_list(j.get('strengths', []), 'strengths')
    weaknesses_html = make_list(j.get('weaknesses', []), 'weaknesses')
    missing_html = make_list(j.get('missing_information', []), 'missing')
    hallucinations_html = make_list(j.get('hallucinations', []), 'hallucinations')

    return f"""
    <div class="question-card">
        <div class="question-header {verdict_class}">
            <span class="question-number">Q{judgment['question_number']}</span>
            <div class="question-title">{judgment['question']}</div>
            <div class="question-score">
                <div class="score-circle {verdict_class}">{scores['overall']}/10</div>
                <span class="expand-icon">▼</span>
            </div>
        </div>
        <div class="question-details">
            <div class="detail-section">
                <h4>Model's Answer</h4>
                <div class="content">{judgment['model_answer']}</div>
            </div>

            <div class="detail-section">
                <h4>Reference Answer</h4>
                <div class="content">{judgment['expected_answer']}</div>
            </div>

            <div class="detail-section">
                <h4>Detailed Scores</h4>
                {scores_html}
            </div>

            <div class="detail-section">
                <h4>Strengths</h4>
                {strengths_html}
            </div>

            <div class="detail-section">
                <h4>Weaknesses</h4>
                {weaknesses_html}
            </div>

            <div class="detail-section">
                <h4>Missing Information</h4>
                {missing_html}
            </div>

            <div class="detail-section">
                <h4>Hallucinations</h4>
                {hallucinations_html}
            </div>

            <div class="reasoning">
                <strong>Reasoning:</strong> {j.get('reasoning', 'N/A')}
            </div>
        </div>
    </div>
    """


def generate_html_report(judgments: List[Dict[str, Any]],
                        model_name: str,
                        original_results_file: Path) -> str:
    """Generate HTML report from judgments."""

    # Load template
    with open(HTML_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = f.read()

    # Calculate statistics
    successful = [j for j in judgments if j['judge_result']['success']]

    if not successful:
        avg_score = 0
        min_score = 0
        max_score = 0
        correct_count = 0
        partial_count = 0
        incorrect_count = 0
    else:
        scores = [j['judge_result']['judgment']['scores']['overall'] for j in successful]
        verdicts = [j['judge_result']['judgment']['verdict'] for j in successful]

        avg_score = f"{sum(scores) / len(scores):.1f}"
        min_score = f"{min(scores):.1f}"
        max_score = f"{max(scores):.1f}"

        correct_count = sum(1 for v in verdicts if v == 'CORRECT')
        partial_count = sum(1 for v in verdicts if v == 'PARTIALLY_CORRECT')
        incorrect_count = sum(1 for v in verdicts if v == 'INCORRECT')

    success_rate = f"{len(successful) / len(judgments) * 100:.0f}" if judgments else "0"

    # Generate questions HTML
    questions_html = '\n'.join(generate_question_html(j) for j in judgments)

    # Extract model name from path
    if '/' in model_name:
        model_display = Path(model_name).name
    else:
        model_display = model_name

    # Fill in template
    html = template.replace('{{MODEL_NAME}}', model_display)
    html = html.replace('{{JUDGE_MODEL}}', JUDGE_MODEL)
    html = html.replace('{{EVALUATION_DATE}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    html = html.replace('{{EVALUATED_FILE}}', original_results_file.name)
    html = html.replace('{{AVG_SCORE}}', avg_score)
    html = html.replace('{{MIN_SCORE}}', min_score)
    html = html.replace('{{MAX_SCORE}}', max_score)
    html = html.replace('{{TOTAL_QUESTIONS}}', str(len(judgments)))
    html = html.replace('{{SUCCESSFUL_JUDGMENTS}}', str(len(successful)))
    html = html.replace('{{SUCCESS_RATE}}', success_rate)
    html = html.replace('{{CORRECT_COUNT}}', str(correct_count))
    html = html.replace('{{PARTIAL_COUNT}}', str(partial_count))
    html = html.replace('{{INCORRECT_COUNT}}', str(incorrect_count))
    html = html.replace('{{QUESTIONS_HTML}}', questions_html)
    html = html.replace('{{GENERATION_DATE}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return html


def save_judgment_results(judgments: List[Dict[str, Any]],
                         model_name: str,
                         original_results_file: Path,
                         output_path: Path,
                         use_full_context: bool):
    """Save judgment results to JSON and HTML files."""
    output_data = {
        "judge_model": JUDGE_MODEL,
        "evaluation_date": datetime.now().isoformat(),
        "evaluated_file": str(original_results_file),
        "config": {
            "temperature": JUDGE_TEMPERATURE,
            "max_tokens": JUDGE_MAX_TOKENS
        },
        "judgments": judgments
    }

    # Save JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ JSON saved to: {output_path}")

    # Generate and save HTML
    html_output = generate_html_report(judgments, model_name, original_results_file)
    html_path = output_path.with_suffix('.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    print(f"✓ HTML saved to: {html_path}")


def print_judgment_summary(judgments: List[Dict[str, Any]]):
    """Print summary statistics of judgments."""
    total = len(judgments)
    successful = sum(1 for j in judgments if j['judge_result']['success'])
    failed = total - successful

    if successful == 0:
        print("\n⚠ No successful judgments to summarize")
        return

    # Calculate statistics
    scores = [j['judge_result']['judgment']['scores']['overall']
              for j in judgments if j['judge_result']['success']]

    verdicts = [j['judge_result']['judgment']['verdict']
                for j in judgments if j['judge_result']['success']]

    correct = sum(1 for v in verdicts if v == 'CORRECT')
    partial = sum(1 for v in verdicts if v == 'PARTIALLY_CORRECT')
    incorrect = sum(1 for v in verdicts if v == 'INCORRECT')

    avg_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0

    total_time = sum(j['judge_result']['elapsed_time'] for j in judgments)
    avg_time = total_time / total if total > 0 else 0

    print(f"\n{'='*80}")
    print("JUDGMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions: {total}")
    print(f"Successfully judged: {successful}")
    print(f"Failed: {failed}")

    print(f"\nOverall Scores:")
    print(f"  Average: {avg_score:.2f}/10")
    print(f"  Min: {min_score:.1f}/10")
    print(f"  Max: {max_score:.1f}/10")

    print(f"\nVerdicts:")
    print(f"  CORRECT: {correct} ({correct/successful*100:.1f}%)")
    print(f"  PARTIALLY_CORRECT: {partial} ({partial/successful*100:.1f}%)")
    print(f"  INCORRECT: {incorrect} ({incorrect/successful*100:.1f}%)")

    print(f"\nTiming:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per question: {avg_time:.2f}s")

    # Show worst performing questions
    if successful > 0:
        print(f"\nLowest Scoring Questions:")
        scored_questions = [(j['question_number'], j['question'][:60],
                           j['judge_result']['judgment']['scores']['overall'])
                          for j in judgments if j['judge_result']['success']]
        scored_questions.sort(key=lambda x: x[2])

        for i, (num, q, score) in enumerate(scored_questions[:3], 1):
            print(f"  {i}. Q{num} (Score: {score:.1f}): {q}...")


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    print("="*80)
    print("LLM-as-a-Judge Evaluation")
    print("="*80)
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"OpenRouter API: {OPENROUTER_BASE_URL}")
    print(f"Context Mode: {'Full Document' if args.full_context else 'Specific Segments'}")
    print("="*80)

    # Check for API key
    if not OPENROUTER_API_KEY:
        print("\n❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return 1

    # List and select results file
    result_files = list_result_files()
    selected_file = display_results_menu(result_files)

    if not selected_file:
        print("\nNo file selected. Exiting.")
        return 0

    print(f"\nSelected: {selected_file.name}")

    # Load the results file
    print("\nLoading results...")
    with open(selected_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    model_name = results_data.get('model', 'unknown')
    results = results_data.get('results', [])
    print(f"✓ Loaded {len(results)} results from {model_name}")

    # Load questions with context
    print("\nLoading questions with context...")
    questions_dict = load_questions_with_context()
    print(f"✓ Loaded {len(questions_dict)} questions with context")

    # Load full FAQ document if needed
    faq_full_document = None
    if args.full_context:
        print("\nLoading full FAQ document...")
        faq_full_document = load_faq_document()
        print(f"✓ Loaded full FAQ document: {len(faq_full_document):,} chars")
        print("⚠ Using full document as context (not specific segments)")

    # Initialize OpenRouter client
    print(f"\nInitializing OpenRouter client...")
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )
    print("✓ Client initialized")

    # Process each result
    judgments = []

    for i, result in enumerate(results, 1):
        question = result['question']
        expected_answer = result['expected_answer']

        # Get model's response
        if not result['response']['success']:
            print(f"\nSkipping Q{i}: Model failed to generate answer")
            judgments.append({
                'question_number': result['question_number'],
                'question': question,
                'judge_result': {
                    'success': False,
                    'judgment': None,
                    'error': 'Model did not generate answer',
                    'elapsed_time': 0
                }
            })
            continue

        model_answer = result['response']['answer']

        # Get context for this question
        if question not in questions_dict:
            print(f"\n⚠ Warning: Q{i} not found in questions file, skipping")
            continue

        question_data = questions_dict[question]
        context_segments = question_data['context']

        # Build judge prompt
        judge_prompt = build_judge_prompt(
            question=question,
            reference_answer=expected_answer,
            model_answer=model_answer,
            context_segments=context_segments,
            full_context=faq_full_document
        )

        # Query judge
        judge_result = query_judge(client, judge_prompt, i, len(results))

        # Store judgment
        judgment = {
            'question_number': result['question_number'],
            'question': question,
            'expected_answer': expected_answer,
            'model_answer': model_answer,
            'judge_result': judge_result
        }
        judgments.append(judgment)

        # Small delay to avoid rate limits
        if i < len(results):
            time.sleep(1)

    # Save judgments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract model name from original file
    original_model = selected_file.stem.split('_', 2)[-1] if '_' in selected_file.stem else 'model'

    # Add context mode to filename
    context_suffix = "_fullctx" if args.full_context else ""
    output_file = RESULTS_DIR / f"judgment_{timestamp}_{original_model}{context_suffix}.json"

    save_judgment_results(judgments, model_name, selected_file, output_file, args.full_context)

    # Print summary
    print_judgment_summary(judgments)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
