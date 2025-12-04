#!/usr/bin/env python3
"""
Evaluate a local LLM on hard questions using the full document context.

PURPOSE:
    Evaluate local LLM models running on llama.cpp server for Croatian language Q&A.
    Tests models on challenging questions from the Loomen FAQ dataset.

WHAT IT DOES:
    - Connects to llama.cpp server via OpenAI-compatible API
    - Loads the full FAQ document as context
    - Tests the model on challenging questions from the hard questions dataset
    - Measures response time, token usage, and answer quality
    - Saves detailed results for later analysis with judge_evaluation.py

CONFIGURATION (edit these variables in the script):
    - LLAMA_CPP_HOST: Server hostname or IP (default: "100.111.222.54")
    - LLAMA_CPP_PORT: Server port (default: 8080)
    - TEMPERATURE: Sampling temperature (default: 0.1 for consistency)
    - MAX_TOKENS: Maximum response tokens (default: 2048)
    - TOP_P: Top-p sampling parameter (default: 0.95)

USAGE:
    # Basic evaluation (uses default configuration)
    ./evaluate_local_model.py

    # To change configuration, edit the script variables:
    # - LLAMA_CPP_HOST: Change to your server IP
    # - LLAMA_CPP_PORT: Change if using different port
    # - TEMPERATURE, MAX_TOKENS, TOP_P: Adjust sampling parameters

REQUIREMENTS:
    - llama.cpp server must be running and accessible
    - Server must expose OpenAI-compatible API at /v1 endpoint
    - Data files must be present in data/ directory:
        * data/loomen_faq.md (FAQ document)
        * data/loomen_faq.hard_questions.json (test questions)
    - Prompt template: prompts/llm_odgovor_cijeli_dokument.txt

OUTPUT:
    - Console: Progress display with response previews and timing
    - JSON file: results/evaluation_TIMESTAMP_MODELNAME.json
        Contains: questions, model responses, timing, token usage
    - Summary statistics: success rate, average time, total tokens

OUTPUT FORMAT:
    {
        "model": "model-name",
        "evaluation_date": "ISO-timestamp",
        "config": {"temperature": 0.1, "max_tokens": 2048, ...},
        "results": [
            {
                "question_number": 1,
                "question": "...",
                "expected_answer": "...",
                "response": {
                    "success": true,
                    "answer": "...",
                    "elapsed_time": 3.14,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, ...},
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
    1. Start llama.cpp server with your model
    2. Run this script: ./evaluate_local_model.py
    3. Check console output for immediate feedback
    4. Review results/evaluation_*.json for details
    5. Run judge_evaluation.py for automated scoring
"""

import time
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
LLAMA_CPP_HOST = "100.111.222.54"
LLAMA_CPP_PORT = 8080
LLAMA_CPP_BASE_URL = f"http://{LLAMA_CPP_HOST}:{LLAMA_CPP_PORT}/v1"

# Model configuration (will be fetched from server)
TEMPERATURE = 0.1  # Low temperature for more consistent answers
MAX_TOKENS = 2048  # Maximum tokens for the response
TOP_P = 0.95


def get_llama_cpp_model_name(client: OpenAI) -> str:
    """
    Fetch the actual model name from the llama.cpp server.

    Returns:
        The full model path/name as reported by the server
    """
    try:
        models = client.models.list()
        if models.data and len(models.data) > 0:
            model_id = models.data[0].id
            print(f"✓ Detected model: {model_id}")
            return model_id
        else:
            print("⚠ No models found, using default name")
            return "unknown-model"
    except Exception as e:
        print(f"⚠ Could not fetch model name: {e}")
        return "unknown-model"


def query_llama_cpp_model(client: OpenAI, model_name: str, prompt: str,
                          question_num: int, total: int) -> Dict[str, Any]:
    """
    Query the llama.cpp model via OpenAI API.

    Args:
        client: OpenAI client configured for llama.cpp
        model_name: The model name to use in the request
        prompt: The complete prompt to send
        question_num: Current question number (for progress display)
        total: Total number of questions

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
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            stream=False
        )

        elapsed_time = time.time() - start_time

        print(response.json())

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
    print("="*80)
    print("LLM Evaluation: Hard Questions with Full Document Context")
    print("="*80)
    print(f"Server: {LLAMA_CPP_BASE_URL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max tokens: {MAX_TOKENS}")
    print("="*80)

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

    # Initialize OpenAI client for llama.cpp
    print(f"\nConnecting to llama.cpp server at {LLAMA_CPP_BASE_URL}...")
    client = OpenAI(
        base_url=LLAMA_CPP_BASE_URL,
        api_key="not-needed"  # llama.cpp doesn't require API key
    )
    print("✓ Client initialized")

    # Fetch the actual model name from the server
    print("\nFetching model information...")
    model_name = get_llama_cpp_model_name(client)
    model_name_short = sanitize_model_name_for_filename(model_name)
    print(f"Model for filename: {model_name_short}")

    # Process each question
    results = []

    for i, question_obj in enumerate(questions, 1):
        question = question_obj['question']
        expected_answer = question_obj['answer']

        # Build the complete prompt
        full_prompt = build_full_prompt(prompt_template, faq_document, question)

        print("===================================")
        print("===================================")
        print("===================================")
        print("===================================")
        print(full_prompt)
        print("===================================")
        print("===================================")
        print("===================================")
        print("===================================")

        # Query the model
        response = query_llama_cpp_model(client, model_name, full_prompt, i, len(questions))

        # Store result
        result = {
            'question_number': i,
            'question': question,
            'expected_answer': expected_answer,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)

        # Small delay between requests to avoid overwhelming the server
        if i < len(questions):
            time.sleep(0.5)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"evaluation_{timestamp}_{model_name_short}.json"

    config = {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "server": LLAMA_CPP_BASE_URL
    }
    save_results(results, model_name, output_file, config)

    # Print summary
    print_summary(results)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
