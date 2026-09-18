import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset_split")

FILES = {
    "TRAIN": {
        "input": DATASET_DIR / "train.json",
        "output": DATASET_DIR / "train.jsonl",
        "expected_count": 700,
    },
    "VALIDATION": {
        "input": DATASET_DIR / "validation.json",
        "output": DATASET_DIR / "validation.jsonl",
        "expected_count": 150,
    },
    "TEST": {
        "input": DATASET_DIR / "test.json",
        "output": DATASET_DIR / "test.jsonl",
        "expected_count": 150,
    },
}

SYSTEM_MESSAGE = (
    "You are an expert academic evaluator. Grade the student's answer "
    "strictly according to the provided question and rubric. Evaluate every "
    "criterion independently and award only the points justified by the "
    "student's answer."
)

REQUIRED_TOP_LEVEL_FIELDS = {
    "response_id",
    "question_id",
    "question",
    "student_answer",
    "rubric",
    "final_score",
    "max_score",
}

REQUIRED_RUBRIC_FIELDS = {
    "criterion",
    "points",
    "points_awarded",
}


# ============================================================
# ERROR CLASS
# ============================================================

class DatasetValidationError(Exception):
    """Raised when a dataset record fails validation."""
    pass


# ============================================================
# LOAD JSON
# ============================================================

def load_json_file(path):
    """Load and validate that the JSON file contains a list."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetValidationError(
            f"Malformed JSON in {path}: "
            f"line {e.lineno}, column {e.colno}: {e.msg}"
        )

    if not isinstance(data, list):
        raise DatasetValidationError(
            f"{path} must contain a JSON list/array of objects."
        )

    return data


# ============================================================
# VALIDATE RECORD
# ============================================================

def validate_record(record, record_number, file_name):
    """
    Validate one original dataset record.

    record_number is 1-based.
    """

    location = f"{file_name}, record {record_number}"

    # --------------------------------------------------------
    # Record must be a dictionary/object
    # --------------------------------------------------------

    if not isinstance(record, dict):
        raise DatasetValidationError(
            f"{location}: expected a JSON object, "
            f"but found {type(record).__name__}."
        )

    # --------------------------------------------------------
    # Required top-level fields
    # --------------------------------------------------------

    missing_fields = REQUIRED_TOP_LEVEL_FIELDS - set(record.keys())

    if missing_fields:
        raise DatasetValidationError(
            f"{location}: missing required field(s): "
            f"{sorted(missing_fields)}"
        )

    # --------------------------------------------------------
    # Validate response_id
    # --------------------------------------------------------

    if not isinstance(record["response_id"], str):
        raise DatasetValidationError(
            f"{location}: 'response_id' must be a string."
        )

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not isinstance(record["question"], str):
        raise DatasetValidationError(
            f"{location}: 'question' must be a string."
        )

    # --------------------------------------------------------
    # Validate student answer
    # --------------------------------------------------------

    if not isinstance(record["student_answer"], str):
        raise DatasetValidationError(
            f"{location}: 'student_answer' must be a string."
        )

    # --------------------------------------------------------
    # Validate rubric
    # --------------------------------------------------------

    if not isinstance(record["rubric"], list):
        raise DatasetValidationError(
            f"{location}: 'rubric' must be a list."
        )

    if len(record["rubric"]) == 0:
        raise DatasetValidationError(
            f"{location}: 'rubric' cannot be empty."
        )

    for rubric_number, criterion in enumerate(
        record["rubric"], start=1
    ):

        rubric_location = (
            f"{location}, rubric criterion {rubric_number}"
        )

        if not isinstance(criterion, dict):
            raise DatasetValidationError(
                f"{rubric_location}: expected a JSON object."
            )

        missing_rubric_fields = (
            REQUIRED_RUBRIC_FIELDS - set(criterion.keys())
        )

        if missing_rubric_fields:
            raise DatasetValidationError(
                f"{rubric_location}: missing required field(s): "
                f"{sorted(missing_rubric_fields)}"
            )

        if not isinstance(criterion["criterion"], str):
            raise DatasetValidationError(
                f"{rubric_location}: 'criterion' must be a string."
            )

        if not isinstance(criterion["points"], (int, float)):
            raise DatasetValidationError(
                f"{rubric_location}: 'points' must be numeric."
            )

        if isinstance(criterion["points"], bool):
            raise DatasetValidationError(
                f"{rubric_location}: 'points' cannot be boolean."
            )

        if not isinstance(
            criterion["points_awarded"],
            (int, float)
        ):
            raise DatasetValidationError(
                f"{rubric_location}: 'points_awarded' must be numeric."
            )

        if isinstance(criterion["points_awarded"], bool):
            raise DatasetValidationError(
                f"{rubric_location}: "
                f"'points_awarded' cannot be boolean."
            )

    # --------------------------------------------------------
    # Validate final_score and max_score
    # --------------------------------------------------------

    if not isinstance(
        record["final_score"],
        (int, float)
    ):
        raise DatasetValidationError(
            f"{location}: 'final_score' must be numeric."
        )

    if isinstance(record["final_score"], bool):
        raise DatasetValidationError(
            f"{location}: 'final_score' cannot be boolean."
        )

    if not isinstance(
        record["max_score"],
        (int, float)
    ):
        raise DatasetValidationError(
            f"{location}: 'max_score' must be numeric."
        )

    if isinstance(record["max_score"], bool):
        raise DatasetValidationError(
            f"{location}: 'max_score' cannot be boolean."
        )


# ============================================================
# BUILD USER MESSAGE
# ============================================================

def build_user_message(record):
    """
    Build the user message.

    IMPORTANT:
    Only question, rubric, and student answer are included.
    """

    question = record["question"]

    rubric_lines = []

    for index, criterion in enumerate(
        record["rubric"],
        start=1
    ):
        criterion_text = criterion["criterion"]
        points = criterion["points"]

        rubric_lines.append(
            f"{index}. {criterion_text} — "
            f"Maximum {points} point(s)"
        )

    rubric_text = "\n".join(rubric_lines)

    student_answer = record["student_answer"]

    user_content = (
        f"Question:\n"
        f"{question}\n\n"
        f"Rubric:\n"
        f"{rubric_text}\n\n"
        f"Student Answer:\n"
        f"{student_answer}"
    )

    return user_content


# ============================================================
# BUILD ASSISTANT MESSAGE
# ============================================================

def build_assistant_message(record):
    """
    Build the assistant message containing the original
    ground-truth rubric scores and final score.
    """

    lines = []

    for index, criterion in enumerate(
        record["rubric"],
        start=1
    ):
        points_awarded = criterion["points_awarded"]

        lines.append(
            f"Criterion {index}: {points_awarded}"
        )

    lines.append(
        f"Total: {record['final_score']}/{record['max_score']}"
    )

    return "\n".join(lines)


# ============================================================
# CONVERT ONE RECORD
# ============================================================

def convert_record(record):
    """
    Convert one original JSON object into one JSONL object.
    """

    user_message = build_user_message(record)
    assistant_message = build_assistant_message(record)

    converted = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": assistant_message,
            },
        ]
    }

    return converted


# ============================================================
# VERIFY JSONL OUTPUT
# ============================================================

def verify_output(
    output_path,
    original_records,
    original_response_ids,
):
    """
    Verify:
    - output line count
    - one JSON object per line
    - response_id positional mapping

    response_id is stored in the original data but deliberately
    NOT included in the training JSONL object. Therefore, this
    verification uses the corresponding line position and checks
    that the generated line was produced from the expected
    original record.
    """

    if not output_path.exists():
        raise DatasetValidationError(
            f"Output file was not created: {output_path}"
        )

    output_records = []

    with output_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                raise DatasetValidationError(
                    f"{output_path}: empty line found at "
                    f"line {line_number}."
                )

            try:
                converted = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetValidationError(
                    f"{output_path}: malformed JSONL at "
                    f"line {line_number}: {e.msg}"
                )

            if not isinstance(converted, dict):
                raise DatasetValidationError(
                    f"{output_path}: line {line_number} "
                    f"is not a JSON object."
                )

            if "messages" not in converted:
                raise DatasetValidationError(
                    f"{output_path}: line {line_number} "
                    f"does not contain 'messages'."
                )

            if len(converted["messages"]) != 3:
                raise DatasetValidationError(
                    f"{output_path}: line {line_number} "
                    f"must contain exactly 3 messages."
                )

            output_records.append(converted)

    input_count = len(original_records)
    output_count = len(output_records)

    if input_count != output_count:
        raise DatasetValidationError(
            f"{output_path}: input/output record count mismatch. "
            f"Input={input_count}, Output={output_count}"
        )

    # --------------------------------------------------------
    # Verify positional mapping through the generated content.
    #
    # response_id itself is intentionally NOT written to JSONL.
    # Therefore, verify that the question and student answer
    # at every position correspond to the original record.
    # --------------------------------------------------------

    for index, (
        original,
        converted
    ) in enumerate(
        zip(original_records, output_records),
        start=1
    ):

        messages = converted["messages"]

        user_content = messages[1]["content"]

        expected_question = original["question"]
        expected_student_answer = original["student_answer"]

        if expected_question not in user_content:
            raise DatasetValidationError(
                f"{output_path}: positional mapping error at "
                f"line {index}. "
                f"Expected response_id="
                f"{original_response_ids[index - 1]}"
            )

        if expected_student_answer not in user_content:
            raise DatasetValidationError(
                f"{output_path}: positional mapping error at "
                f"line {index}. "
                f"Expected response_id="
                f"{original_response_ids[index - 1]}"
            )

    return output_count


# ============================================================
# CONVERT ONE FILE
# ============================================================

def convert_file(split_name, config):
    """
    Independently convert one JSON file to JSONL.
    """

    input_path = config["input"]
    output_path = config["output"]
    expected_count = config["expected_count"]

    print()
    print("=" * 60)
    print(f"PROCESSING: {split_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    records = load_json_file(input_path)

    input_count = len(records)

    print(f"Input file: {input_path}")
    print(f"Input records: {input_count}")
    print(f"Expected records: {expected_count}")

    # --------------------------------------------------------
    # Verify expected dataset size
    # --------------------------------------------------------

    if input_count != expected_count:
        raise DatasetValidationError(
            f"{input_path}: expected {expected_count} records "
            f"but found {input_count}."
        )

    # --------------------------------------------------------
    # Validate ALL records BEFORE writing output.
    #
    # This prevents partially converted files if malformed
    # data is discovered halfway through conversion.
    # --------------------------------------------------------

    print("Validating records...")

    for record_number, record in enumerate(
        records,
        start=1
    ):
        validate_record(
            record,
            record_number,
            input_path.name
        )

    print("Validation: PASSED")

    # --------------------------------------------------------
    # Store original response IDs for positional verification
    # --------------------------------------------------------

    original_response_ids = [
        record["response_id"]
        for record in records
    ]

    # --------------------------------------------------------
    # Convert records in EXACT existing order
    # --------------------------------------------------------

    converted_records = []

    for record in records:
        converted_records.append(
            convert_record(record)
        )

    # --------------------------------------------------------
    # Write JSONL
    # --------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as f:

        for converted_record in converted_records:

            json_line = json.dumps(
                converted_record,
                ensure_ascii=False,
                separators=(",", ":")
            )

            f.write(json_line + "\n")

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    print(f"Output file: {output_path}")
    print("Verifying output...")

    output_count = verify_output(
        output_path,
        records,
        original_response_ids
    )

    print(f"Output records: {output_count}")

    if input_count != output_count:
        raise DatasetValidationError(
            f"{split_name}: input/output count mismatch."
        )

    print(f"{split_name}: SUCCESS")

    return records, converted_records


# ============================================================
# PRINT SAMPLES
# ============================================================

def print_samples(
    split_name,
    original_records,
    converted_records,
    number_of_samples=3
):
    """
    Print a few converted records for manual inspection.
    """

    print()
    print("=" * 60)
    print(f"{split_name} — SAMPLE CONVERTED RECORDS")
    print("=" * 60)

    sample_count = min(
        number_of_samples,
        len(converted_records)
    )

    for index in range(sample_count):

        original = original_records[index]
        converted = converted_records[index]

        print()
        print("-" * 60)
        print(
            f"Sample {index + 1} | "
            f"Original response_id: "
            f"{original['response_id']}"
        )
        print("-" * 60)

        print(
            json.dumps(
                converted,
                ensure_ascii=False,
                indent=2
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("JSON → JSONL DATASET CONVERSION")
    print("=" * 60)

    print()
    print("IMPORTANT:")
    print("- No shuffling")
    print("- No sorting")
    print("- No repartitioning")
    print("- No merging")
    print("- No record movement")
    print("- No record deletion")
    print("- No record duplication")
    print("- Each JSON object → exactly one JSONL line")
    print("- Original order is preserved")

    results = {}

    try:

        # ----------------------------------------------------
        # Process TRAIN independently
        # ----------------------------------------------------

        train_records, train_converted = convert_file(
            "TRAIN",
            FILES["TRAIN"]
        )

        results["TRAIN"] = (
            train_records,
            train_converted
        )

        # ----------------------------------------------------
        # Process VALIDATION independently
        # ----------------------------------------------------

        validation_records, validation_converted = convert_file(
            "VALIDATION",
            FILES["VALIDATION"]
        )

        results["VALIDATION"] = (
            validation_records,
            validation_converted
        )

        # ----------------------------------------------------
        # Process TEST independently
        # ----------------------------------------------------

        test_records, test_converted = convert_file(
            "TEST",
            FILES["TEST"]
        )

        results["TEST"] = (
            test_records,
            test_converted
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("CONVERSION FAILED")
        print("=" * 60)
        print(str(e))

        raise SystemExit(1)

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print()
    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for split_name, config in FILES.items():

        input_count = len(results[split_name][0])
        output_count = len(results[split_name][1])

        print()
        print(split_name)
        print(f"Input records:  {input_count}")
        print(f"Output records: {output_count}")

        if input_count == output_count:
            print("Status: SUCCESS")
        else:
            print("Status: FAILED")

    # --------------------------------------------------------
    # Print samples from each split
    # --------------------------------------------------------

    for split_name in ["TRAIN", "VALIDATION", "TEST"]:

        original_records, converted_records = results[split_name]

        print_samples(
            split_name,
            original_records,
            converted_records,
            number_of_samples=3
        )

    print()
    print("=" * 60)
    print("ALL CONVERSIONS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print()
    print("Generated files:")

    for split_name, config in FILES.items():
        print(f"  {config['output']}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()