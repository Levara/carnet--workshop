#!/usr/bin/env python3
"""
Evaluate OpenRouter models on hard questions using the full document context.

PURPOSE:
    Evaluate cloud-based LLM models via OpenRouter API for Croatian language Q&A.
    Supports Claude, GPT-4, Gemini, and other OpenRouter-hosted models.

WHAT IT DOES:
    - Connects to OpenRouter API (supports multiple model providers)
    - Loads the full FAQ document and hard questions
    - Tests model performance on Croatian language Q&A
    - Measures response time, token usage, and answer quality
    - Saves detailed results for later analysis with judge_evaluation.py

CONFIGURATION:
    - API key from environment variable: OPENROUTER_API_KEY
    - Model specified via command line argument (required)
    - Default parameters: temperature=0.1, max_tokens=2048, top_p=0.95
    - All parameters can be overridden via command line options

SUPPORTED MODELS (examples):
    - anthropic/claude-3.5-sonnet (recommended for quality)
    - google/gemini-2.5-flash-preview-09-2025 (fast, cost-effective)
    - openai/gpt-4-turbo
    - openai/gpt-4o
    - meta-llama/llama-3.1-70b-instruct
    - qwen/qwen-2.5-72b-instruct
    - Any other OpenRouter-supported model

USAGE:
    # Set API key (required)
    export OPENROUTER_API_KEY="your-key-here"

    # Basic usage with Claude
    ./evaluate_openrouter_model.py --model anthropic/claude-3.5-sonnet

    # With custom parameters
    ./evaluate_openrouter_model.py \\
        --model google/gemini-2.5-flash-preview-09-2025 \\
        --temperature 0.2 \\
        --max-tokens 3000

    # GPT-4 evaluation
    ./evaluate_openrouter_model.py --model openai/gpt-4-turbo

COMMAND LINE OPTIONS:
    --model, -m         : Model name (required, e.g., anthropic/claude-3.5-sonnet)
    --temperature, -t   : Sampling temperature (default: 0.1)
    --max-tokens        : Maximum response tokens (default: 2048)
    --top-p             : Top-p sampling parameter (default: 0.95)

REQUIREMENTS:
    - OPENROUTER_API_KEY environment variable must be set
    - Data files must be present in data/ directory:
        * data/loomen_faq.md (FAQ document)
        * data/loomen_faq.hard_questions.json (test questions)
    - Prompt template: prompts/llm_odgovor_cijeli_dokument.txt

OUTPUT:
    - Console: Progress display with response previews and timing
    - JSON file: results/evaluation_TIMESTAMP_MODELNAME.json
        Contains: questions, model responses, timing, token usage, config
    - Summary statistics: success rate, average time, total tokens

OUTPUT FORMAT:
    {
        "model": "provider/model-name",
        "evaluation_date": "ISO-timestamp",
        "config": {
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 0.95,
            "provider": "openrouter",
            "api_base": "https://openrouter.ai/api/v1"
        },
        "results": [
            {
                "question_number": 1,
                "question": "...",
                "expected_answer": "...",
                "response": {
                    "success": true,
                    "answer": "...",
                    "elapsed_time": 2.5,
                    "usage": {"prompt_tokens": 150, "completion_tokens": 75, ...},
                    "finish_reason": "stop"
                },
                "timestamp": "ISO-timestamp"
            },
            ...
        ]
    }

NEXT STEPS:
    After running this script, use judge_evaluation.py to score the responses:
        ./judge_evaluation.py

EXAMPLE WORKFLOW:
    1. Set up API key: export OPENROUTER_API_KEY="sk-or-v1-..."
    2. Run evaluation: ./evaluate_openrouter_model.py --model anthropic/claude-3.5-sonnet
    3. Check console for progress
    4. Review results/evaluation_*.json for details
    5. Run judge_evaluation.py for automated scoring
    6. Compare results from different models
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from openai import OpenAI

# Import shared utilities
from evaluation_utils import (
    ensure_results_dir,
    load_prompt_template,
    load_faq_document,
    load_questions,
    build_full_prompt,
    sanitize_model_name_for_filename,
    save_results,
    print_summary,
    format_response_success,
    format_response_error,
    RESULTS_DIR
)


# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model parameters (can be overridden via CLI)
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.95


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate OpenRouter models on hard questions with full document context',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Claude 3.5 Sonnet
  python evaluate_openrouter_model.py --model anthropic/claude-3.5-sonnet

  # Evaluate GPT-4 Turbo with custom parameters
  python evaluate_openrouter_model.py --model openai/gpt-4-turbo --temperature 0.2 --max-tokens 3000

  # Evaluate Gemini Pro
  python evaluate_openrouter_model.py --model google/gemini-pro-1.5

Popular models:
  - anthropic/claude-3.5-sonnet (recommended)
  - anthropic/claude-3-opus
  - openai/gpt-4-turbo
  - openai/gpt-4o
  - google/gemini-pro-1.5
  - google/gemini-flash-1.5
  - meta-llama/llama-3.1-70b-instruct
  - qwen/qwen-2.5-72b-instruct
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='OpenRouter model name (e.g., anthropic/claude-3.5-sonnet)'
    )

    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f'Temperature for sampling (default: {DEFAULT_TEMPERATURE})'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Maximum tokens for response (default: {DEFAULT_MAX_TOKENS})'
    )

    parser.add_argument(
        '--top-p',
        type=float,
        default=DEFAULT_TOP_P,
        help=f'Top-p for nucleus sampling (default: {DEFAULT_TOP_P})'
    )

    return parser.parse_args()


def query_openrouter_model(client: OpenAI, model_name: str, prompt: str,
                           question_num: int, total: int,
                           temperature: float, max_tokens: int, top_p: float) -> Dict[str, Any]:
    """
    Query the OpenRouter model via OpenAI-compatible API.

    Args:
        client: OpenAI client configured for OpenRouter
        model_name: The model name to use
        prompt: The complete prompt to send
        question_num: Current question number (for progress display)
        total: Total number of questions
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        top_p: Top-p sampling parameter

    Returns:
        Dictionary with response data
    """
    print(f"\n{'='*80}")
    print(f"Querying model: Question {question_num}/{total}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False
        )

        elapsed_time = time.time() - start_time

        # Extract response
        answer = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        # Extract token usage if available
        usage = {
            'prompt_tokens': response.usage.prompt_tokens if response.usage else None,
            'completion_tokens': response.usage.completion_tokens if response.usage else None,
            'total_tokens': response.usage.total_tokens if response.usage else None,
        }

        print(f"✓ Response received in {elapsed_time:.2f}s")
        if usage['total_tokens']:
            print(f"  Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total")
        print(f"  Finish reason: {finish_reason}")
        print(f"\nAnswer preview: {answer[:200]}..." if len(answer) > 200 else f"\nAnswer: {answer}")

        return format_response_success(answer, elapsed_time, usage, finish_reason)

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Error after {elapsed_time:.2f}s: {e}")
        return format_response_error(str(e), elapsed_time)


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    print("="*80)
    print("LLM Evaluation: Hard Questions with Full Document Context (OpenRouter)")
    print("="*80)
    print(f"Server: {OPENROUTER_BASE_URL}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Top-p: {args.top_p}")
    print("="*80)

    # Check for API key
    if not OPENROUTER_API_KEY:
        print("\n❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return 1

    # Create results directory if it doesn't exist
    ensure_results_dir()

    # Load files
    print("\nLoading files...")
    prompt_template = load_prompt_template()
    print(f"✓ Loaded prompt template: {len(prompt_template):,} chars")

    faq_document = load_faq_document()
    print(f"✓ Loaded FAQ document: {len(faq_document):,} chars")

    questions = load_questions()
    print(f"✓ Loaded {len(questions)} hard questions")

    # Initialize OpenAI client for OpenRouter
    print(f"\nConnecting to OpenRouter at {OPENROUTER_BASE_URL}...")
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )
    print("✓ Client initialized")

    # Sanitize model name for filename
    model_name_short = sanitize_model_name_for_filename(args.model)
    print(f"Model for filename: {model_name_short}")

    # Process each question
    results = []

    for i, question_obj in enumerate(questions, 1):
        question = question_obj['question']
        expected_answer = question_obj['answer']

        # Build the complete prompt
        full_prompt = build_full_prompt(prompt_template, faq_document, question)

        # Query the model
        response = query_openrouter_model(
            client=client,
            model_name=args.model,
            prompt=full_prompt,
            question_num=i,
            total=len(questions),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p
        )

        # Store result
        result = {
            'question_number': i,
            'question': question,
            'expected_answer': expected_answer,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)

        # Delay between requests to respect rate limits
        if i < len(questions):
            time.sleep(1)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"evaluation_{timestamp}_{model_name_short}.json"

    config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "provider": "openrouter",
        "api_base": OPENROUTER_BASE_URL
    }
    save_results(results, args.model, output_file, config)

    # Print summary
    print_summary(results)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
