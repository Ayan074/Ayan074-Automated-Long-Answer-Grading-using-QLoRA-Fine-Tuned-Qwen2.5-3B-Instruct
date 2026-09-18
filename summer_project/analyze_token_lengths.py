import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DATASET_DIR = Path("dataset_split")

FILES = {
    "TRAIN": DATASET_DIR / "train.jsonl",
    "VALIDATION": DATASET_DIR / "validation.jsonl",
    "TEST": DATASET_DIR / "test.jsonl",
}

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# Expected dataset sizes
EXPECTED_COUNTS = {
    "TRAIN": 700,
    "VALIDATION": 150,
    "TEST": 150,
}


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("TOKEN LENGTH ANALYSIS")
print("=" * 70)

print()
print(f"Model: {MODEL_NAME}")
print(f"Cache: {HF_CACHE_DIR}")

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=str(HF_CACHE_DIR),
)

print("Tokenizer loaded successfully.")
print(f"Vocabulary size: {len(tokenizer)}")


# ============================================================
# COUNT TOKENS
# ============================================================

def count_tokens(messages):
    """
    Count tokens using the Qwen chat template.

    This approximates the actual sequence that will be
    presented to the model during instruction fine-tuning.
    """

    if not isinstance(messages, list):
        raise ValueError("'messages' must be a list.")

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    return len(token_ids)


# ============================================================
# PERCENTILE FUNCTION
# ============================================================

def percentile(values, percentile):
    """
    Calculate percentile without requiring NumPy.
    """

    if not values:
        return 0

    sorted_values = sorted(values)

    k = (len(sorted_values) - 1) * (percentile / 100)

    lower = int(k)
    upper = min(lower + 1, len(sorted_values))

    weight = k - lower

    return (
        sorted_values[lower]
        + weight * (
            sorted_values[upper]
            - sorted_values[lower]
        )
    )


# ============================================================
# ANALYZE ONE FILE
# ============================================================

def analyze_file(split_name, path):

    print()
    print("=" * 70)
    print(f"{split_name}: {path}")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            if not line.strip():
                raise ValueError(
                    f"{path}: empty line at line "
                    f"{line_number}"
                )

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"{path}: invalid JSON at line "
                    f"{line_number}: {e}"
                )

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}: line {line_number} "
                    f"is not a JSON object."
                )

            if "messages" not in record:
                raise ValueError(
                    f"{path}: line {line_number} "
                    f"does not contain 'messages'."
                )

            if not isinstance(
                record["messages"],
                list
            ):
                raise ValueError(
                    f"{path}: line {line_number}: "
                    f"'messages' must be a list."
                )

            if len(record["messages"]) != 3:
                raise ValueError(
                    f"{path}: line {line_number}: "
                    f"expected 3 messages, found "
                    f"{len(record['messages'])}."
                )

            records.append(record)

    # --------------------------------------------------------
    # Count check
    # --------------------------------------------------------

    actual_count = len(records)
    expected_count = EXPECTED_COUNTS[split_name]

    print()
    print(f"Records found   : {actual_count}")
    print(f"Expected records: {expected_count}")

    if actual_count != expected_count:
        raise ValueError(
            f"{split_name}: expected "
            f"{expected_count} records but found "
            f"{actual_count}."
        )

    print("Record count: PASS")

    # --------------------------------------------------------
    # Token counting
    # --------------------------------------------------------

    token_lengths = []

    max_record = None
    max_tokens = -1

    min_record = None
    min_tokens = None

    print()
    print("Counting tokens...")

    for index, record in enumerate(
        records,
        start=1
    ):

        messages = record["messages"]

        token_count = count_tokens(messages)

        token_lengths.append(token_count)

        # Track longest
        if token_count > max_tokens:
            max_tokens = token_count
            max_record = index

        # Track shortest
        if (
            min_tokens is None
            or token_count < min_tokens
        ):
            min_tokens = token_count
            min_record = index

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_tokens = statistics.mean(token_lengths)
    median_tokens = statistics.median(token_lengths)

    p90 = percentile(token_lengths, 90)
    p95 = percentile(token_lengths, 95)
    p99 = percentile(token_lengths, 99)

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TOKEN LENGTH STATISTICS")
    print("-" * 70)

    print(f"Minimum tokens : {min_tokens}")
    print(f"Maximum tokens : {max_tokens}")
    print(f"Mean tokens    : {mean_tokens:.2f}")
    print(f"Median tokens  : {median_tokens:.2f}")
    print(f"90th percentile: {p90:.2f}")
    print(f"95th percentile: {p95:.2f}")
    print(f"99th percentile: {p99:.2f}")

    print()
    print(
        f"Shortest example: record {min_record} "
        f"({min_tokens} tokens)"
    )

    print(
        f"Longest example : record {max_record} "
        f"({max_tokens} tokens)"
    )

    # --------------------------------------------------------
    # Check candidate sequence lengths
    # --------------------------------------------------------

    candidate_lengths = [
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
    ]

    print()
    print("-" * 70)
    print("SEQUENCE LENGTH COVERAGE")
    print("-" * 70)

    print(
        f"{'Max Length':>12} | "
        f"{'Examples Fit':>12} | "
        f"{'Examples Truncated':>19} | "
        f"{'Coverage':>10}"
    )

    print("-" * 70)

    for max_length in candidate_lengths:

        fitting = sum(
            1
            for length in token_lengths
            if length <= max_length
        )

        truncated = len(token_lengths) - fitting

        coverage = (
            fitting / len(token_lengths)
        ) * 100

        print(
            f"{max_length:>12} | "
            f"{fitting:>12} | "
            f"{truncated:>19} | "
            f"{coverage:>9.2f}%"
        )

    # --------------------------------------------------------
    # Examples above important limits
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("EXAMPLES EXCEEDING IMPORTANT LIMITS")
    print("-" * 70)

    for limit in [
        512,
        768,
        1024,
        1536,
        2048,
    ]:

        count = sum(
            1
            for length in token_lengths
            if length > limit
        )

        percentage = (
            count / len(token_lengths)
        ) * 100

        print(
            f"> {limit:4d} tokens: "
            f"{count:4d} examples "
            f"({percentage:6.2f}%)"
        )

    return {
        "count": actual_count,
        "lengths": token_lengths,
        "min": min_tokens,
        "max": max_tokens,
        "mean": mean_tokens,
        "median": median_tokens,
        "p90": p90,
        "p95": p95,
        "p99": p99,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    results = {}

    try:

        for split_name, path in FILES.items():

            results[split_name] = analyze_file(
                split_name,
                path
            )

    except Exception as e:

        print()
        print("=" * 70)
        print("ANALYSIS FAILED")
        print("=" * 70)
        print(str(e))

        raise SystemExit(1)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL TOKEN LENGTH SUMMARY")
    print("=" * 70)

    print()
    print(
        f"{'Split':<15}"
        f"{'Count':>8}"
        f"{'Min':>8}"
        f"{'Mean':>10}"
        f"{'Median':>10}"
        f"{'P90':>10}"
        f"{'P95':>10}"
        f"{'P99':>10}"
        f"{'Max':>10}"
    )

    print("-" * 70)

    for split_name, result in results.items():

        print(
            f"{split_name:<15}"
            f"{result['count']:>8}"
            f"{result['min']:>8}"
            f"{result['mean']:>10.1f}"
            f"{result['median']:>10.1f}"
            f"{result['p90']:>10.1f}"
            f"{result['p95']:>10.1f}"
            f"{result['p99']:>10.1f}"
            f"{result['max']:>10}"
        )

    # ========================================================
    # TRAINING-SET RECOMMENDATION
    # ========================================================

    train = results["TRAIN"]

    print()
    print("=" * 70)
    print("INITIAL SEQUENCE LENGTH RECOMMENDATION")
    print("=" * 70)

    print()

    if train["p95"] <= 1024:

        print(
            "Recommendation: START WITH max_seq_length = 1024"
        )

        print(
            "At least 95% of the training examples fit within "
            "1024 tokens."
        )

    elif train["p95"] <= 1536:

        print(
            "Recommendation: CONSIDER max_seq_length = 1536"
        )

        print(
            "More than 95% of the training examples fit within "
            "1536 tokens."
        )

    elif train["p95"] <= 2048:

        print(
            "Recommendation: CONSIDER max_seq_length = 2048"
        )

        print(
            "More than 95% of the training examples fit within "
            "2048 tokens."
        )

    else:

        print(
            "WARNING: A significant number of examples are "
            "longer than 2048 tokens."
        )

        print(
            "We should inspect the long examples before "
            "choosing the training sequence length."
        )

    print()
    print(
        "IMPORTANT: This recommendation is based on token "
        "length only. We will consider GPU memory before "
        "finalizing the training configuration."
    )

    print()
    print("=" * 70)
    print("TOKEN LENGTH ANALYSIS COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()