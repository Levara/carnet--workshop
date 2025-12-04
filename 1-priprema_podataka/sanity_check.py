#!/usr/bin/env python3
"""
Sanity check script to verify data integrity for evaluation datasets.

PURPOSE:
    Validates that all context segments in the hard questions dataset exist in the
    source FAQ document. Ensures data integrity before running evaluations.

WHAT IT DOES:
    - Loads the FAQ document and hard questions JSON
    - Normalizes text (removes whitespace, punctuation, converts to lowercase)
    - Uses n-gram matching to verify each context segment
    - Reports match percentage for each segment
    - Identifies segments that may have been incorrectly extracted or modified
    - Exits with error code if validation fails

TEXT NORMALIZATION:
    To handle formatting differences, text is normalized by:
    1. Removing markdown link syntax [text](url) -> text url
    2. Converting to lowercase
    3. Removing all whitespace
    4. Removing all punctuation (keeps letters and digits)
    5. Preserving Unicode letters (Croatian č, ć, š, ž, đ, etc.)

MATCHING ALGORITHM:
    - Uses 5-character n-grams for fuzzy matching
    - Calculates percentage of segment n-grams found in document
    - Default threshold: 90% match required for validation pass
    - Tolerates minor formatting differences while catching major issues

USAGE:
    # Run validation (no arguments needed)
    ./sanity_check.py

    # Check exit code
    echo $?  # 0 if all pass, 1 if any fail

REQUIREMENTS:
    - Data files must be present:
        * data/loomen_faq.md (source FAQ document)
        * data/loomen_faq.hard_questions.json (test questions with context)

OUTPUT:
    Console display with:
    - Per-segment verification results with match percentages
    - Location identifiers for each segment
    - Summary statistics:
        * Total segments checked
        * Segments found
        * Segments missing/low match
        * Success rate percentage
    - Detailed report of low-match segments (if any):
        * Question text
        * Segment location
        * Match percentage
        * Segment text preview

EXIT CODES:
    0 : All segments passed validation (≥90% match)
    1 : One or more segments failed validation (<90% match)

USE CASES:
    - Verify data integrity after receiving files via email
    - Validate dataset before running expensive LLM evaluations
    - Debug issues with context extraction
    - Ensure no corruption during file transfer
    - CI/CD pipeline validation step

INTERPRETATION:
    - 100% match: Segment exists exactly in document (ideal)
    - 90-99% match: Minor formatting differences (acceptable)
    - 70-89% match: Significant differences (investigate)
    - <70% match: Major issues, segment may not exist (fail)

EXAMPLE OUTPUT:
    ================================================================================
    Question 1/20
    Q: Kako mogu resetirati svoju lozinku?
    Segments to verify: 2
      ✓ Segment 1/2: FOUND (98.5% match)
        Location: "Resetiranje lozinke > Koraci"
      ✓ Segment 2/2: FOUND (100.0% match)
        Location: "Resetiranje lozinke > Važne napomene"

    ================================================================================
    SUMMARY
    ================================================================================
    Total segments checked: 45
    Segments found: 45
    Segments missing: 0
    Success rate: 100.0%

    ✓ VERIFICATION PASSED: All context segments found in the source document!

TROUBLESHOOTING:
    If validation fails:
    1. Check if data files are present in data/ directory
    2. Verify files were not corrupted during transfer
    3. Check encoding (should be UTF-8)
    4. Review low-match segment details in output
    5. Compare segment text with FAQ document manually
    6. Contact data provider if issues persist

TYPICAL WORKFLOW:
    1. Receive data files via email
    2. Place in data/ directory
    3. Run: ./sanity_check.py
    4. If pass: proceed with evaluations
    5. If fail: investigate reported segments
"""

import json
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by:
    1. Removing markdown link syntax [text](url) -> text url
    2. Converting to lowercase
    3. Removing all whitespace
    4. Removing all punctuation (keeps letters and digits)
    5. Keeps Unicode letters (Croatian č, ć, š, ž, đ, etc.)

    Args:
        text: Input text

    Returns:
        Normalized text with only letters and digits (lowercase, no whitespace/punctuation)
    """
    # Remove markdown links: [text](url) -> text url
    # This handles cases like [https://...](https://...) or [text](url)
    text_no_md = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1 \2', text)

    # Convert to lowercase
    text_lower = text_no_md.lower()
    # Remove all whitespace
    no_whitespace = re.sub(r'\s+', '', text_lower)
    # Keep only letters (including Unicode) and digits
    # This removes punctuation but keeps Croatian characters
    letters_only = re.sub(r'[^\w]', '', no_whitespace, flags=re.UNICODE)
    return letters_only


def load_faq_document(faq_path: Path) -> str:
    """
    Load the FAQ markdown file and normalize it.

    Args:
        faq_path: Path to loomen_faq.md

    Returns:
        FAQ content normalized (ASCII only, no whitespace, lowercase)
    """
    with open(faq_path, 'r', encoding='utf-8') as f:
        content = f.read()

    normalized = normalize_text(content)
    print(f"✓ Loaded FAQ document: {len(content):,} chars, {len(normalized):,} chars (normalized)")
    return normalized


def load_hard_questions(json_path: Path) -> list:
    """
    Load the hard questions JSON file.

    Args:
        json_path: Path to loomen_faq.hard_questions.json

    Returns:
        List of question objects
    """
    # Read with UTF-8 encoding and handle potential encoding issues
    with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Parse JSON
    try:
        questions = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Error at position {e.pos}: {content[max(0, e.pos-50):e.pos+50]}")
        raise

    print(f"✓ Loaded {len(questions)} hard questions")
    return questions


def calculate_ngram_match(segment: str, document: str, n: int = 5) -> float:
    """
    Calculate what percentage of n-grams from segment appear in document.

    Args:
        segment: The text segment to search for
        document: The document to search in
        n: N-gram size (default: 5 characters)

    Returns:
        Percentage match (0.0 to 100.0)
    """
    if len(segment) < n:
        # For very short segments, use exact substring match
        return 100.0 if segment in document else 0.0

    # Generate n-grams from the segment
    segment_ngrams = set()
    for i in range(len(segment) - n + 1):
        ngram = segment[i:i+n]
        segment_ngrams.add(ngram)

    if not segment_ngrams:
        return 0.0

    # Count how many segment n-grams appear in the document
    matches = sum(1 for ngram in segment_ngrams if ngram in document)

    # Calculate percentage
    percentage = (matches / len(segment_ngrams)) * 100.0
    return percentage


def verify_contexts(faq_normalized: str, questions: list, threshold: float = 90.0) -> tuple[int, int, list]:
    """
    Verify that context segments exist in the FAQ document using n-gram matching.

    Args:
        faq_normalized: FAQ document normalized
        questions: List of question objects
        threshold: Minimum percentage match to consider as found (default: 90%)

    Returns:
        Tuple of (total_segments, found_segments, missing_segments)
    """
    total_segments = 0
    found_segments = 0
    missing_segments = []

    for i, question_obj in enumerate(questions, 1):
        question = question_obj['question']
        contexts = question_obj['context']

        print(f"\n{'='*80}")
        print(f"Question {i}/{len(questions)}")
        print(f"Q: {question[:80]}..." if len(question) > 80 else f"Q: {question}")
        print(f"Segments to verify: {len(contexts)}")

        for j, context in enumerate(contexts, 1):
            total_segments += 1
            segment_location = context['segment_location']
            segment_text = context['segment_text']

            # Normalize the segment text
            segment_normalized = normalize_text(segment_text)

            # Calculate n-gram match percentage
            match_pct = calculate_ngram_match(segment_normalized, faq_normalized, n=5)

            # Check if match percentage exceeds threshold
            if match_pct >= threshold:
                found_segments += 1
                print(f"  ✓ Segment {j}/{len(contexts)}: FOUND ({match_pct:.1f}% match)")
                print(f"    Location: {segment_location}")
            else:
                print(f"  ✗ Segment {j}/{len(contexts)}: LOW MATCH ({match_pct:.1f}%)")
                print(f"    Location: {segment_location}")
                print(f"    Text preview: {segment_text[:100]}...")

                missing_segments.append({
                    'question_number': i,
                    'question': question,
                    'segment_number': j,
                    'segment_location': segment_location,
                    'segment_text': segment_text,
                    'match_percentage': match_pct
                })

    return total_segments, found_segments, missing_segments


def print_summary(total: int, found: int, missing: list):
    """
    Print summary of verification results.

    Args:
        total: Total number of segments checked
        found: Number of segments found
        missing: List of missing segment details
    """
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total segments checked: {total}")
    print(f"Segments found: {found}")
    print(f"Segments missing: {len(missing)}")
    print(f"Success rate: {found/total*100:.1f}%")

    if missing:
        print(f"\n{'='*80}")
        print("LOW MATCH SEGMENTS DETAILS")
        print(f"{'='*80}")

        for item in missing:
            match_pct = item.get('match_percentage', 0.0)
            print(f"\nQuestion {item['question_number']}: {item['question']}")
            print(f"Segment {item['segment_number']} - Location: {item['segment_location']}")
            print(f"Match percentage: {match_pct:.1f}%")
            print(f"Text:\n{item['segment_text']}\n")
            print("-" * 80)

    # Exit with error code if any segments are missing
    if missing:
        print("\n⚠️  VERIFICATION FAILED: Some context segments are not found in the source document!")
        return 1
    else:
        print("\n✓ VERIFICATION PASSED: All context segments found in the source document!")
        return 0


def main():
    """Main execution function."""
    # Define paths
    data_dir = Path(__file__).parent / 'data'
    faq_path = data_dir / 'loomen_faq.md'
    json_path = data_dir / 'loomen_faq.hard_questions.json'

    # Verify files exist
    if not faq_path.exists():
        print(f"❌ Error: FAQ file not found at {faq_path}")
        return 1

    if not json_path.exists():
        print(f"❌ Error: JSON file not found at {json_path}")
        return 1

    print("Loomen FAQ Sanity Check")
    print("=" * 80)

    # Load and process files
    faq_normalized = load_faq_document(faq_path)
    questions = load_hard_questions(json_path)

    # Verify all contexts
    total, found, missing = verify_contexts(faq_normalized, questions)

    # Print summary and exit
    return print_summary(total, found, missing)


if __name__ == '__main__':
    exit(main())
