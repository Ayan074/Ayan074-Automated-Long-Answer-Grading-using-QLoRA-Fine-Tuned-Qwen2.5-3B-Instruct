import json
import re
import torch
import numpy as np

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pearsonr, spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TEST_FILE = "dataset_split/complete_TEST_data.json"

OUTPUT_DIR = Path("baseline_results_newtest")

PREDICTIONS_FILE = OUTPUT_DIR / "predictions.json"
SUMMARY_FILE = OUTPUT_DIR / "metrics.json"

RANDOM_SEED = 42

MAX_NEW_TOKENS = 300


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# PRINT HARDWARE INFORMATION
# ============================================================

print("=" * 70)
print("BASELINE EVALUATION")
print("=" * 70)

print("\nHardware information:")

if torch.cuda.is_available():

    print("CUDA available: YES")
    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"GPU VRAM: {gpu_memory:.2f} GB")

else:

    print("CUDA available: NO")
    print("WARNING: Model will run on CPU.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

if not isinstance(test_data, list):
    raise ValueError("Test dataset must be a JSON list.")

print(f"Test records: {len(test_data)}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BASE MODEL")
print("=" * 70)

if torch.cuda.is_available():

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

model.eval()

print("Model loaded successfully.")

print(
    "Model device:",
    next(model.parameters()).device
)


# ============================================================
# BUILD GRADING PROMPT
# ============================================================

def build_prompt(record):

    question = record["question"]

    student_answer = record["student_answer"]

    rubric = record["rubric"]


    rubric_text = ""

    for i, item in enumerate(rubric, start=1):

        points = item["points"]

        criterion = item["criterion"]

        rubric_text += (
            f"Criterion {i} "
            f"(Maximum {points} points): "
            f"{criterion}\n"
        )


    prompt = f"""
You are an expert academic evaluator.

Your task is to grade a student's answer using the provided question and rubric.

Question:
{question}

Rubric:
{rubric_text}

Student Answer:
{student_answer}

Instructions:

1. Evaluate EVERY rubric criterion independently.
2. Award a score from 0 up to the maximum points specified for that criterion.
3. Never award more points than the maximum points for a criterion.
4. Use only the information in the student's answer when deciding the score.
5. Do not use the expected answer as a separate source of information.
6. Do not provide explanations.
7. Output ONLY one line for each criterion.
8. The output format MUST be:

Criterion 1: [score]
Criterion 2: [score]
Criterion 3: [score]

Continue for every criterion.

Do NOT output a total.
Do NOT output maximum scores.
Do NOT output "/1", "/2", etc.
Do NOT provide explanations.
"""

    return prompt.strip()


# ============================================================
# PARSE MODEL RESPONSE
# ============================================================

def parse_model_response(response, rubric):

    predicted_scores = {}

    # Expected format:
    #
    # Criterion 1: 1
    # Criterion 2: 0
    # Criterion 3: 2

    pattern = re.compile(
        r"Criterion\s*(\d+)\s*:\s*"
        r"(\d+(?:\.\d+)?)",
        re.IGNORECASE
    )

    matches = pattern.findall(response)


    for criterion_number, score in matches:

        criterion_number = int(criterion_number)

        score = float(score)

        if score.is_integer():
            score = int(score)

        predicted_scores[criterion_number] = score


    # --------------------------------------------------------
    # Validate against the actual rubric maximum
    # --------------------------------------------------------

    validated_scores = {}

    invalid_predictions = []

    for i, item in enumerate(rubric, start=1):

        maximum = item["points"]

        if i not in predicted_scores:

            validated_scores[i] = None

            continue


        predicted = predicted_scores[i]


        # Negative score
        if predicted < 0:

            invalid_predictions.append({
                "criterion": i,
                "predicted": predicted,
                "maximum": maximum,
                "reason": "negative_score"
            })

            predicted = 0


        # Score greater than maximum
        if predicted > maximum:

            invalid_predictions.append({
                "criterion": i,
                "predicted": predicted,
                "maximum": maximum,
                "reason": "score_exceeds_maximum"
            })

            predicted = maximum


        validated_scores[i] = predicted


    return validated_scores, invalid_predictions


# ============================================================
# GET GROUND TRUTH RUBRIC SCORES
# ============================================================

def get_ground_truth_scores(record):

    scores = {}

    for i, item in enumerate(
        record["rubric"],
        start=1
    ):

        scores[i] = item["points_awarded"]

    return scores


# ============================================================
# GET MAXIMUM POSSIBLE TOTAL
# ============================================================

def get_maximum_total(record):

    return sum(
        item["points"]
        for item in record["rubric"]
    )


# ============================================================
# GET GROUND TRUTH TOTAL
# ============================================================

def get_ground_truth_total(record):

    return record["final_score"]


# ============================================================
# GENERATE MODEL PREDICTION
# ============================================================

def generate_prediction(prompt):

    messages = [

        {
            "role": "system",
            "content": (
                "You are an expert academic evaluator."
            )
        },

        {
            "role": "user",
            "content": prompt
        }

    ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        text,
        return_tensors="pt"
    )


    if torch.cuda.is_available():

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }


    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id

        )


    # Remove prompt tokens

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]


    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    return response.strip()


# ============================================================
# EVALUATE TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("STARTING BASELINE EVALUATION")
print("=" * 70)


results = []

actual_totals = []
predicted_totals = []

actual_rubric_scores = []
predicted_rubric_scores = []


for index, record in enumerate(
    test_data,
    start=1
):

    response_id = record["response_id"]

    question_id = record["question_id"]


    print(
        f"\n[{index}/{len(test_data)}] "
        f"Evaluating {response_id} ({question_id})..."
    )


    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = build_prompt(record)


    # --------------------------------------------------------
    # Generate model prediction
    # --------------------------------------------------------

    try:

        model_response = generate_prediction(
            prompt
        )

    except Exception as e:

        print(
            f"ERROR while evaluating "
            f"{response_id}: {e}"
        )

        results.append({

            "response_id": response_id,

            "question_id": question_id,

            "status": "generation_error",

            "error": str(e)

        })

        continue


    # --------------------------------------------------------
    # Parse model response
    # --------------------------------------------------------

    predicted_rubrics, invalid_predictions = (
        parse_model_response(
            model_response,
            record["rubric"]
        )
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    actual_rubrics = get_ground_truth_scores(
        record
    )

    actual_total = get_ground_truth_total(
        record
    )

    maximum_total = get_maximum_total(
        record
    )


    # --------------------------------------------------------
    # Calculate predicted total ourselves
    # --------------------------------------------------------

    predicted_total = 0

    missing_criteria = []


    for criterion_number in actual_rubrics:

        predicted_score = predicted_rubrics.get(
            criterion_number
        )


        if predicted_score is None:

            missing_criteria.append(
                criterion_number
            )

        else:

            predicted_total += predicted_score


    # --------------------------------------------------------
    # Absolute error
    # --------------------------------------------------------

    absolute_error = abs(
        predicted_total - actual_total
    )


    # --------------------------------------------------------
    # Rubric-level comparison
    # --------------------------------------------------------

    rubric_correct = 0

    rubric_evaluable = 0

    rubric_total = len(actual_rubrics)


    for criterion_number in actual_rubrics:

        actual_score = actual_rubrics[
            criterion_number
        ]

        predicted_score = predicted_rubrics.get(
            criterion_number
        )


        if predicted_score is not None:

            rubric_evaluable += 1

            actual_rubric_scores.append(
                actual_score
            )

            predicted_rubric_scores.append(
                predicted_score
            )


            if predicted_score == actual_score:

                rubric_correct += 1


    if rubric_evaluable > 0:

        rubric_accuracy = (
            rubric_correct
            / rubric_evaluable
        )

    else:

        rubric_accuracy = 0.0


    # --------------------------------------------------------
    # Store totals for global metrics
    # --------------------------------------------------------

    actual_totals.append(
        actual_total
    )

    predicted_totals.append(
        predicted_total
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {

        "response_id": response_id,

        "question_id": question_id,

        "actual_rubric_scores":
            actual_rubrics,

        "predicted_rubric_scores":
            predicted_rubrics,

        "rubric_maximums": {
            str(i): item["points"]
            for i, item in enumerate(
                record["rubric"],
                start=1
            )
        },

        "actual_total":
            actual_total,

        "predicted_total":
            predicted_total,

        "maximum_total":
            maximum_total,

        "absolute_error":
            absolute_error,

        "rubric_accuracy":
            rubric_accuracy,

        "missing_criteria":
            missing_criteria,

        "invalid_predictions":
            invalid_predictions,

        "model_response":
            model_response,

        "status":
            "success"

    }


    results.append(result)


    # --------------------------------------------------------
    # Print current result
    # --------------------------------------------------------

    print(
        f"    Actual total:      "
        f"{actual_total}/{maximum_total}"
    )

    print(
        f"    Predicted total:   "
        f"{predicted_total}/{maximum_total}"
    )

    print(
        f"    Absolute error:    "
        f"{absolute_error}"
    )

    print(
        f"    Rubric accuracy:   "
        f"{rubric_accuracy:.2%}"
    )


    if invalid_predictions:

        print(
            f"    WARNING: "
            f"{len(invalid_predictions)} "
            f"invalid score(s) detected and clipped."
        )


    if missing_criteria:

        print(
            f"    WARNING: Missing criteria: "
            f"{missing_criteria}"
        )


# ============================================================
# CALCULATE GLOBAL METRICS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING BASELINE METRICS")
print("=" * 70)


actual_array = np.array(
    actual_totals,
    dtype=float
)

predicted_array = np.array(
    predicted_totals,
    dtype=float
)


# ============================================================
# MAE
# ============================================================

mae = np.mean(
    np.abs(
        predicted_array
        - actual_array
    )
)


# ============================================================
# RMSE
# ============================================================

rmse = np.sqrt(
    np.mean(
        (
            predicted_array
            - actual_array
        ) ** 2
    )
)


# ============================================================
# EXACT TOTAL-SCORE ACCURACY
# ============================================================

exact_total_accuracy = np.mean(
    predicted_array == actual_array
)


# ============================================================
# PEARSON CORRELATION
# ============================================================

if (
    len(set(actual_array)) > 1
    and
    len(set(predicted_array)) > 1
):

    pearson_r, pearson_p = pearsonr(
        actual_array,
        predicted_array
    )

else:

    pearson_r = None
    pearson_p = None


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

if (
    len(set(actual_array)) > 1
    and
    len(set(predicted_array)) > 1
):

    spearman_rho, spearman_p = spearmanr(
        actual_array,
        predicted_array
    )

else:

    spearman_rho = None
    spearman_p = None


# ============================================================
# GLOBAL RUBRIC ACCURACY
# ============================================================

if len(actual_rubric_scores) > 0:

    rubric_accuracy = np.mean(
        np.array(actual_rubric_scores)
        ==
        np.array(predicted_rubric_scores)
    )

else:

    rubric_accuracy = None


# ============================================================
# MISSING / INVALID STATISTICS
# ============================================================

total_missing = sum(
    len(result.get("missing_criteria", []))
    for result in results
)


total_invalid = sum(
    len(result.get("invalid_predictions", []))
    for result in results
)


# ============================================================
# CREATE METRICS DICTIONARY
# ============================================================

metrics = {

    "model":
        MODEL_NAME,

    "dataset":
        TEST_FILE,

    "random_seed":
        RANDOM_SEED,

    "num_test_records":
        len(test_data),

    "successful_predictions":
        len(actual_totals),

    "failed_predictions":
        len(test_data)
        - len(actual_totals),

    "mae":
        float(mae),

    "rmse":
        float(rmse),

    "exact_total_score_accuracy":
        float(exact_total_accuracy),

    "rubric_level_accuracy":
        (
            float(rubric_accuracy)
            if rubric_accuracy is not None
            else None
        ),

    "pearson_r":
        (
            float(pearson_r)
            if pearson_r is not None
            else None
        ),

    "pearson_p_value":
        (
            float(pearson_p)
            if pearson_p is not None
            else None
        ),

    "spearman_rho":
        (
            float(spearman_rho)
            if spearman_rho is not None
            else None
        ),

    "spearman_p_value":
        (
            float(spearman_p)
            if spearman_p is not None
            else None
        ),

    "total_missing_criteria":
        total_missing,

    "total_invalid_predictions":
        total_invalid

}


# ============================================================
# SAVE PREDICTIONS
# ============================================================

with open(
    PREDICTIONS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(
    f"Test records:              "
    f"{len(test_data)}"
)

print(
    f"Successful predictions:    "
    f"{len(actual_totals)}"
)

print(
    f"Failed predictions:        "
    f"{len(test_data) - len(actual_totals)}"
)

print(
    f"\nMAE:                       "
    f"{mae:.4f}"
)

print(
    f"RMSE:                      "
    f"{rmse:.4f}"
)

print(
    f"Exact total-score accuracy:"
    f" {exact_total_accuracy:.4f}"
)

if rubric_accuracy is not None:

    print(
        f"Rubric-level accuracy:     "
        f"{rubric_accuracy:.4f}"
    )

else:

    print(
        "Rubric-level accuracy:     N/A"
    )


if pearson_r is not None:

    print(
        f"Pearson r:                 "
        f"{pearson_r:.4f}"
    )

else:

    print(
        "Pearson r:                 N/A"
    )


if spearman_rho is not None:

    print(
        f"Spearman rho:              "
        f"{spearman_rho:.4f}"
    )

else:

    print(
        "Spearman rho:              N/A"
    )


print(
    f"\nMissing rubric predictions: "
    f"{total_missing}"
)

print(
    f"Invalid rubric predictions: "
    f"{total_invalid}"
)


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"Predictions: "
    f"{PREDICTIONS_FILE}"
)

print(
    f"Metrics:     "
    f"{SUMMARY_FILE}"
)

print("\nBaseline evaluation complete.")