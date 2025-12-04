#!/usr/bin/env python3
import sys
import tiktoken

def count_tokens(file_path, model="gpt-4"):
    """Count tokens in a markdown file using tiktoken."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(content)
        return len(tokens)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python token-count.py <markdown_file> [model]")
        print("Example: python token-count.py document.md")
        print("Optional model parameter (default: gpt-4)")
        sys.exit(1)

    file_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4"

    token_count = count_tokens(file_path, model)
    print(f"Token count: {token_count}")
