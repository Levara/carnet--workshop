"""
Shared utilities for model evaluation scripts.

This module contains functions used by both evaluate_local_model.py (llama.cpp)
and evaluate_openrouter_model.py (OpenRouter API).

PURPOSE:
    Provides common functionality for loading data, building prompts, saving results,
    and formatting output for LLM evaluation workflows.

KEY FUNCTIONS:
    - load_faq_document()              : Load and parse FAQ markdown document
    - load_questions()                 : Load hard questions JSON dataset
    - load_prompt_template()           : Load prompt template from prompts directory
    - build_full_prompt()              : Construct complete prompts with context
    - sanitize_model_name_for_filename(): Clean model names for safe file paths
    - save_results()                   : Save evaluation results in JSON format
    - print_summary()                  : Display evaluation statistics
    - format_response_success()        : Format successful API responses
    - format_response_error()          : Format error responses
    - ensure_results_dir()             : Create results directory if needed

FILE PATHS (configurable via module variables):
    - FAQ document: data/loomen_faq.md
    - Questions: data/loomen_faq.hard_questions.json
    - Prompt template: prompts/llm_odgovor_cijeli_dokument.txt
    - Results directory: results/

USAGE:
    from evaluation_utils import (
        load_faq_document,
        load_questions,
        build_full_prompt,
        save_results
    )

    # Load data
    faq_doc = load_faq_document()
    questions = load_questions()

    # Build prompts
    prompt = build_full_prompt(template, faq_doc, question)

    # Save results
    save_results(results, model_name, output_path, config)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


# Paths
DATA_DIR = Path(__file__).parent / 'data'
PROMPTS_DIR = Path(__file__).parent / 'prompts'
RESULTS_DIR = Path(__file__).parent / 'results'

FAQ_PATH = DATA_DIR / 'loomen_faq.md'
QUESTIONS_PATH = DATA_DIR / 'loomen_faq.hard_questions.json'
PROMPT_PATH = PROMPTS_DIR / 'llm_odgovor_cijeli_dokument.txt'


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(exist_ok=True)


def load_prompt_template(prompt_path: Path = PROMPT_PATH) -> str:
    """Load the prompt template."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_faq_document(faq_path: Path = FAQ_PATH) -> str:
    """Load the full FAQ document."""
    with open(faq_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_questions(questions_path: Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    """Load the hard questions."""
    with open(questions_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_full_prompt(instructions: str, document: str, question: str) -> str:
    """
    Build the complete prompt for the model.

    Format: Instructions + <document>...</document> + <question>...</question>
    This ordering enables prompt caching.

    Args:
        instructions: The prompt template with instructions
        document: The full FAQ document
        question: The user's question

    Returns:
        Complete prompt ready to send to the model
    """
    # Build the complete prompt
    full_prompt = instructions.replace(
        "[COMPLETE DOCUMENT CONTENT GOES HERE]",
        document
    ).replace(
        "[USER'S QUESTION GOES HERE]",
        question
    )

    return full_prompt


def sanitize_model_name_for_filename(model_path: str) -> str:
    """
    Convert model path to a safe filename component.

    Args:
        model_path: Full model path like /path/to/model.gguf or provider/model-name

    Returns:
        Sanitized filename
    """
    # Extract just the filename from the path
    filename = Path(model_path).name

    # Remove .gguf extension if present
    if filename.endswith('.gguf'):
        filename = filename[:-5]

    # Replace slashes in provider/model format
    filename = filename.replace('/', '_')

    return filename


def save_results(results: List[Dict[str, Any]], model_name: str,
                output_path: Path, config: Dict[str, Any] = None):
    """
    Save evaluation results to JSON file.

    Args:
        results: List of result dictionaries
        model_name: Full model name/path
        output_path: Path to save JSON file
        config: Optional configuration dictionary
    """
    output_data = {
        "model": model_name,
        "evaluation_date": datetime.now().isoformat(),
        "config": config or {},
        "results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics."""
    total = len(results)
    successful = sum(1 for r in results if r['response']['success'])
    failed = total - successful

    total_time = sum(r['response']['elapsed_time'] for r in results)
    avg_time = total_time / total if total > 0 else 0

    total_tokens = sum(
        r['response']['usage']['total_tokens']
        for r in results
        if r['response']['success'] and r['response'].get('usage')
        and r['response']['usage'].get('total_tokens')
    )

    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total*100:.1f}%")
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per question: {avg_time:.2f}s")

    if total_tokens > 0:
        print(f"Total tokens used: {total_tokens:,}")

    if failed > 0:
        print(f"\nFailed questions:")
        for i, r in enumerate(results, 1):
            if not r['response']['success']:
                print(f"  Question {i}: {r['response']['error']}")


def format_response_success(answer: str, elapsed_time: float, usage: Dict[str, Any],
                           finish_reason: str) -> Dict[str, Any]:
    """
    Format a successful response.

    Args:
        answer: The model's answer text
        elapsed_time: Time taken in seconds
        usage: Token usage dictionary
        finish_reason: Completion finish reason

    Returns:
        Formatted response dictionary
    """
    return {
        'success': True,
        'answer': answer,
        'elapsed_time': elapsed_time,
        'usage': usage,
        'finish_reason': finish_reason,
        'error': None
    }


def format_response_error(error: str, elapsed_time: float) -> Dict[str, Any]:
    """
    Format an error response.

    Args:
        error: Error message
        elapsed_time: Time taken in seconds

    Returns:
        Formatted error response dictionary
    """
    return {
        'success': False,
        'answer': None,
        'elapsed_time': elapsed_time,
        'usage': None,
        'finish_reason': None,
        'error': error
    }
