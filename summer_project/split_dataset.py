import json
import random
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "complete_dataset.json"

OUTPUT_DIR = Path("dataset_split")

TRAIN_PER_QUESTION = 70
VALIDATION_PER_QUESTION = 15
TEST_PER_QUESTION = 15

RANDOM_SEED = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError("Dataset must be a JSON list of records.")

print(f"Total records found: {len(data)}")


# ============================================================
# GROUP RECORDS BY QUESTION
# ============================================================

groups = {}

for record in data:

    question_id = record["question_id"]

    if question_id not in groups:
        groups[question_id] = []

    groups[question_id].append(record)


print("\nRecords per question:")
print("-" * 60)

for question_id in sorted(groups):
    print(f"{question_id}: {len(groups[question_id])}")


# ============================================================
# CHECK DATASET SIZE
# ============================================================

required_per_question = (
    TRAIN_PER_QUESTION
    + VALIDATION_PER_QUESTION
    + TEST_PER_QUESTION
)

for question_id, records in groups.items():

    if len(records) < required_per_question:
        raise ValueError(
            f"{question_id} has only {len(records)} records. "
            f"At least {required_per_question} are required."
        )


# ============================================================
# RANDOM SPLIT
# ============================================================

random.seed(RANDOM_SEED)

train_data = []
validation_data = []
test_data = []

print("\n")
print("=" * 60)
print("CREATING SPLITS")
print("=" * 60)

for question_id in sorted(groups):

    records = groups[question_id].copy()

    # Shuffle records randomly
    random.shuffle(records)

    # First 70
    train_records = records[
        :TRAIN_PER_QUESTION
    ]

    # Next 15
    validation_records = records[
        TRAIN_PER_QUESTION:
        TRAIN_PER_QUESTION + VALIDATION_PER_QUESTION
    ]

    # Next 15
    test_records = records[
        TRAIN_PER_QUESTION + VALIDATION_PER_QUESTION:
        TRAIN_PER_QUESTION + VALIDATION_PER_QUESTION + TEST_PER_QUESTION
    ]

    train_data.extend(train_records)
    validation_data.extend(validation_records)
    test_data.extend(test_records)

    print(
        f"{question_id}: "
        f"Train={len(train_records)}, "
        f"Validation={len(validation_records)}, "
        f"Test={len(test_records)}"
    )


# ============================================================
# SHUFFLE FINAL DATASETS
# ============================================================

random.shuffle(train_data)
random.shuffle(validation_data)
random.shuffle(test_data)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# SAVE DATASETS
# ============================================================

train_file = OUTPUT_DIR / "train.json"
validation_file = OUTPUT_DIR / "validation.json"
test_file = OUTPUT_DIR / "test.json"


with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)


with open(validation_file, "w", encoding="utf-8") as f:
    json.dump(validation_data, f, indent=2, ensure_ascii=False)


with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)


# ============================================================
# VERIFY SPLITS
# ============================================================

print("\n")
print("=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)

print(f"Training records:   {len(train_data)}")
print(f"Validation records: {len(validation_data)}")
print(f"Test records:       {len(test_data)}")

print("\nExpected:")
print(f"Training records:   {len(groups) * TRAIN_PER_QUESTION}")
print(f"Validation records: {len(groups) * VALIDATION_PER_QUESTION}")
print(f"Test records:       {len(groups) * TEST_PER_QUESTION}")


# ============================================================
# VERIFY QUESTION DISTRIBUTION
# ============================================================

print("\n")
print("=" * 60)
print("QUESTION DISTRIBUTION")
print("=" * 60)

train_counts = Counter(
    record["question_id"]
    for record in train_data
)

validation_counts = Counter(
    record["question_id"]
    for record in validation_data
)

test_counts = Counter(
    record["question_id"]
    for record in test_data
)


for question_id in sorted(groups):

    print(
        f"{question_id}: "
        f"Train={train_counts[question_id]}, "
        f"Validation={validation_counts[question_id]}, "
        f"Test={test_counts[question_id]}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)

print(f"Files saved inside: {OUTPUT_DIR}/")

print(f"  {train_file}")
print(f"  {validation_file}")
print(f"  {test_file}")

print("\nRandom seed:", RANDOM_SEED)
