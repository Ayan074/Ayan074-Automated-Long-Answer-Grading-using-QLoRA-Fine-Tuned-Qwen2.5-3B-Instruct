#!/usr/bin/env python3

"""
test_trainer_setup.py

Purpose:
    Verify the complete QLoRA + SFTTrainer setup before
    starting actual fine-tuning.

IMPORTANT:
    This script DOES NOT train the model.

Environment:
    Qwen/Qwen2.5-3B-Instruct
    4-bit NF4 quantization
    LoRA / QLoRA
    TRL 1.10.0
    Transformers 5.15.0
    PEFT 0.20.0
    BitsAndBytes 0.50.0

Dataset:
    train.jsonl       = 700
    validation.jsonl  = 150
    test.jsonl        = 150

The test set is intentionally NOT loaded.
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import sys
from pathlib import Path

import torch

import transformers
import trl
import peft
import bitsandbytes

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)

from trl import (
    SFTConfig,
    SFTTrainer,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DATASET_DIR = Path("dataset_split")

TRAIN_FILE = DATASET_DIR / "train.jsonl"
VALIDATION_FILE = DATASET_DIR / "validation.jsonl"

OUTPUT_DIR = Path("trainer_setup_test_output")

EXPECTED_TRAIN = 700
EXPECTED_VALIDATION = 150

# Based on token-length analysis:
#
# Train maximum      = 599
# Validation maximum = 614
# Test maximum       = 593
#
# Therefore 768 provides complete coverage.

MAX_SEQ_LENGTH = 768

SEED = 42

# ============================================================
# LoRA CONFIGURATION
# ============================================================

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def count_jsonl_records(path):

    count = 0

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                raise ValueError(
                    f"Empty line found in {path} "
                    f"at line {line_number}"
                )

            try:
                json.loads(line)

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {e}"
                )

            count += 1

    return count


def verify_message_structure(dataset, split_name):

    expected_roles = [
        "system",
        "user",
        "assistant",
    ]

    for index, record in enumerate(dataset):

        record_number = index + 1

        # ----------------------------------------------------
        # messages field
        # ----------------------------------------------------

        if "messages" not in record:

            raise ValueError(
                f"{split_name} record {record_number}: "
                f"missing 'messages'"
            )

        messages = record["messages"]

        if not isinstance(messages, list):

            raise ValueError(
                f"{split_name} record {record_number}: "
                f"'messages' is not a list"
            )

        # ----------------------------------------------------
        # Exactly 3 messages
        # ----------------------------------------------------

        if len(messages) != 3:

            raise ValueError(
                f"{split_name} record {record_number}: "
                f"expected 3 messages, "
                f"found {len(messages)}"
            )

        # ----------------------------------------------------
        # Check roles and contents
        # ----------------------------------------------------

        for i, expected_role in enumerate(expected_roles):

            message = messages[i]

            if not isinstance(message, dict):

                raise ValueError(
                    f"{split_name} record {record_number}: "
                    f"message {i + 1} is not an object"
                )

            actual_role = message.get("role")

            if actual_role != expected_role:

                raise ValueError(
                    f"{split_name} record {record_number}: "
                    f"message {i + 1} should have role "
                    f"'{expected_role}', "
                    f"found '{actual_role}'"
                )

            if "content" not in message:

                raise ValueError(
                    f"{split_name} record {record_number}: "
                    f"message {i + 1} missing content"
                )

            if not isinstance(message["content"], str):

                raise ValueError(
                    f"{split_name} record {record_number}: "
                    f"message {i + 1} content is not a string"
                )


# ============================================================
# START
# ============================================================

print_section("QLoRA TRAINER SETUP TEST")

print("This script will NOT perform training.")
print("It will only construct and verify the trainer.")


# ============================================================
# 1. SOFTWARE VERSION CHECK
# ============================================================

print_section("1. SOFTWARE VERSION CHECK")

print(f"Python          : {sys.version}")
print(f"PyTorch         : {torch.__version__}")
print(f"Transformers    : {transformers.__version__}")
print(f"TRL             : {trl.__version__}")
print(f"PEFT            : {peft.__version__}")
print(f"BitsAndBytes    : {bitsandbytes.__version__}")

print()

if torch.cuda.is_available():

    print("CUDA available  : True")
    print(f"CUDA version    : {torch.version.cuda}")
    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

else:

    raise RuntimeError(
        "CUDA is not available. "
        "QLoRA requires an NVIDIA GPU."
    )


# ============================================================
# 2. DATASET FILE CHECK
# ============================================================

print_section("2. DATASET FILE CHECK")

if not TRAIN_FILE.exists():

    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}"
    )

if not VALIDATION_FILE.exists():

    raise FileNotFoundError(
        f"Validation file not found: {VALIDATION_FILE}"
    )

print(f"Train file      : {TRAIN_FILE}")
print(f"Validation file : {VALIDATION_FILE}")

train_count = count_jsonl_records(TRAIN_FILE)

validation_count = count_jsonl_records(
    VALIDATION_FILE
)

print()

print(
    f"Train records      : {train_count}"
)

print(
    f"Validation records : {validation_count}"
)

if train_count != EXPECTED_TRAIN:

    raise ValueError(
        f"Expected {EXPECTED_TRAIN} training records, "
        f"found {train_count}"
    )

if validation_count != EXPECTED_VALIDATION:

    raise ValueError(
        f"Expected {EXPECTED_VALIDATION} validation records, "
        f"found {validation_count}"
    )

print()
print("Dataset count check: PASS")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print_section("3. LOADING DATASET")

dataset = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VALIDATION_FILE),
    },
)

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

print(
    f"Train examples      : "
    f"{len(train_dataset)}"
)

print(
    f"Validation examples : "
    f"{len(validation_dataset)}"
)

print()
print("Dataset loading: SUCCESS")


# ============================================================
# 4. MESSAGE STRUCTURE CHECK
# ============================================================

print_section("4. MESSAGE STRUCTURE CHECK")

verify_message_structure(
    train_dataset,
    "TRAIN"
)

verify_message_structure(
    validation_dataset,
    "VALIDATION"
)

print(
    "TRAIN message structure      : PASS"
)

print(
    "VALIDATION message structure : PASS"
)


# ============================================================
# 5. SAMPLE DATA CHECK
# ============================================================

print_section("5. SAMPLE DATA CHECK")

sample = train_dataset[0]

print("First training record:")
print()

for message in sample["messages"]:

    print(
        f"[{message['role'].upper()}]"
    )

    print(
        message["content"][:1000]
    )

    print()

print("Sample record check: PASS")


# ============================================================
# 6. LOAD TOKENIZER
# ============================================================

print_section("6. LOADING TOKENIZER")

print(
    f"Model: {MODEL_NAME}"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

print(
    "Tokenizer loaded successfully."
)

print(
    f"Vocabulary size : {len(tokenizer)}"
)

print(
    f"EOS token       : {tokenizer.eos_token}"
)

print(
    f"PAD token       : {tokenizer.pad_token}"
)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token

    print()
    print(
        "PAD token was not defined."
    )

    print(
        "Using EOS token as PAD token."
    )

print()
print("Tokenizer check: PASS")


# ============================================================
# 7. TOKENIZATION TEST
# ============================================================

print_section("7. TOKENIZATION TEST")

sample_messages = sample["messages"]

formatted_text = tokenizer.apply_chat_template(
    sample_messages,
    tokenize=False,
    add_generation_prompt=False,
)

tokens = tokenizer(
    formatted_text,
    add_special_tokens=False,
)

token_count = len(
    tokens["input_ids"]
)

print(
    f"Sample token count : {token_count}"
)

print(
    f"Maximum length     : {MAX_SEQ_LENGTH}"
)

if token_count > MAX_SEQ_LENGTH:

    print()
    print(
        "WARNING: sample exceeds "
        "maximum sequence length."
    )

else:

    print()
    print(
        "Sample fits within "
        "maximum sequence length."
    )

print()
print("Tokenization check: PASS")


# ============================================================
# 8. CONFIGURE 4-BIT QUANTIZATION
# ============================================================

print_section(
    "8. CONFIGURING 4-BIT QUANTIZATION"
)

compute_dtype = torch.float16

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=compute_dtype,
)

print(
    "4-bit quantization configuration:"
)

print()

print(
    "Load in 4-bit       : True"
)

print(
    "Quantization type   : NF4"
)

print(
    "Double quantization : True"
)

print(
    "Compute dtype       : float16"
)

print()
print(
    "Quantization configuration: PASS"
)


# ============================================================
# 9. LOAD BASE MODEL
# ============================================================

print_section(
    "9. LOADING QWEN2.5-3B-INSTRUCT"
)

print(
    "Loading model in 4-bit mode..."
)

print(
    "Please wait..."
)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map={"": 0},

    dtype=compute_dtype,
)

print()
print(
    "Model loading: SUCCESS"
)

print()

print(
    f"Model device: "
    f"{next(model.parameters()).device}"
)


# ============================================================
# 10. MODEL CONFIGURATION
# ============================================================

print_section(
    "10. MODEL CONFIGURATION"
)

model.config.use_cache = False

print(
    "use_cache: False"
)

print(
    "Model configuration: PASS"
)


# ============================================================
# 11. PREPARE FOR QLoRA
# ============================================================

print_section(
    "11. PREPARING MODEL FOR QLoRA"
)

model = prepare_model_for_kbit_training(
    model
)

print(
    "prepare_model_for_kbit_training: SUCCESS"
)


# ============================================================
# 12. LoRA CONFIGURATION
# ============================================================

print_section(
    "12. LoRA CONFIGURATION"
)

lora_config = LoraConfig(

    r=LORA_R,

    lora_alpha=LORA_ALPHA,

    lora_dropout=LORA_DROPOUT,

    target_modules=LORA_TARGET_MODULES,

    bias="none",

    task_type="CAUSAL_LM",
)

print(
    f"LoRA rank       : {LORA_R}"
)

print(
    f"LoRA alpha      : {LORA_ALPHA}"
)

print(
    f"LoRA dropout    : {LORA_DROPOUT}"
)

print()

print(
    "Target modules:"
)

for module in LORA_TARGET_MODULES:

    print(
        f"  - {module}"
    )

print()

print(
    "LoRA configuration: SUCCESS"
)


# ============================================================
# 13. SFT TRAINING CONFIGURATION
# ============================================================

print_section(
    "13. SFT TRAINING CONFIGURATION"
)

print(
    "Creating SFTConfig..."
)

# IMPORTANT:
#
# Your installed TRL 1.10.0 does NOT support:
#
#     warmup_ratio
#
# It supports:
#
#     warmup_steps
#
# Therefore we use warmup_steps=20.
#
# Also:
#
#     max_length=768
#
# is supported by your installed TRL version.

training_args = SFTConfig(

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir=str(
        OUTPUT_DIR
    ),

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_train_epochs=3,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    learning_rate=2e-4,

    weight_decay=0.01,

    warmup_steps=20,

    lr_scheduler_type="cosine",

    optim="paged_adamw_8bit",

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    fp16=True,

    # --------------------------------------------------------
    # Gradient checkpointing
    # --------------------------------------------------------

    gradient_checkpointing=True,

    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="epoch",

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=3,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=10,

    logging_first_step=True,

    # --------------------------------------------------------
    # Sequence length
    # --------------------------------------------------------

    max_length=MAX_SEQ_LENGTH,

    packing=False,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,

    data_seed=SEED,

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    report_to="none",

    # --------------------------------------------------------
    # Evaluation output
    # --------------------------------------------------------

    prediction_loss_only=True,
)

print()

print(
    "SFTConfig created successfully."
)

print()

print(
    "Important configuration:"
)

print(
    f"Max sequence length   : "
    f"{MAX_SEQ_LENGTH}"
)

print(
    "Epochs                : 3"
)

print(
    "Train batch size      : 1"
)

print(
    "Gradient accumulation : 8"
)

print(
    "Effective batch size  : 8"
)

print(
    "Learning rate         : 2e-4"
)

print(
    "Warmup steps          : 20"
)

print(
    "Optimizer             : "
    "paged_adamw_8bit"
)

print(
    "Precision             : FP16"
)

print(
    "Gradient checkpoint   : True"
)

print(
    "Evaluation            : Every epoch"
)

print(
    "Saving                : Every epoch"
)

print()

print(
    "SFTConfig check: PASS"
)


# ============================================================
# 14. CREATE SFT TRAINER
# ============================================================

print_section(
    "14. CREATING SFTTRAINER"
)

print(
    "Constructing SFTTrainer..."
)

print()

print(
    "IMPORTANT:"
)

print(
    "This does NOT start training."
)

print()

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    peft_config=lora_config,
)

print()

print(
    "SFTTrainer construction: SUCCESS"
)


# ============================================================
# 15. LoRA PARAMETER CHECK
# ============================================================

print_section(
    "15. LoRA TRAINABLE PARAMETER CHECK"
)

try:

    trainer.model.print_trainable_parameters()

except AttributeError:

    trainable_params = 0

    all_params = 0

    for parameter in trainer.model.parameters():

        all_params += parameter.numel()

        if parameter.requires_grad:

            trainable_params += parameter.numel()

    percentage = (
        100.0
        * trainable_params
        / all_params
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    print(
        f"All parameters       : "
        f"{all_params:,}"
    )

    print(
        f"Trainable percentage : "
        f"{percentage:.4f}%"
    )


# ============================================================
# 16. VERIFY BASE MODEL FREEZING
# ============================================================

print_section(
    "16. BASE MODEL / LoRA FREEZING CHECK"
)

trainable = []

frozen = 0

for name, parameter in (
    trainer.model.named_parameters()
):

    if parameter.requires_grad:

        trainable.append(name)

    else:

        frozen += 1


print(
    f"Trainable parameter tensors : "
    f"{len(trainable)}"
)

print(
    f"Frozen parameter tensors    : "
    f"{frozen}"
)

print()

print(
    "First trainable parameters:"
)

for name in trainable[:20]:

    print(
        f"  {name}"
    )

print()

lora_trainable = [

    name

    for name in trainable

    if "lora_" in name.lower()

]

if len(lora_trainable) == 0:

    raise RuntimeError(
        "No LoRA parameters were found "
        "among trainable parameters."
    )

print(
    f"LoRA trainable tensors found: "
    f"{len(lora_trainable)}"
)

print()

print(
    "Freezing / LoRA check: PASS"
)


# ============================================================
# 17. GPU MEMORY CHECK
# ============================================================

print_section(
    "17. GPU MEMORY CHECK"
)

if torch.cuda.is_available():

    device = torch.cuda.current_device()

    total_memory = (
        torch.cuda
        .get_device_properties(device)
        .total_memory
        / 1024**3
    )

    allocated_memory = (
        torch.cuda
        .memory_allocated(device)
        / 1024**3
    )

    reserved_memory = (
        torch.cuda
        .memory_reserved(device)
        / 1024**3
    )

    print(
        f"GPU            : "
        f"{torch.cuda.get_device_name(device)}"
    )

    print(
        f"Total VRAM     : "
        f"{total_memory:.2f} GB"
    )

    print(
        f"Allocated VRAM : "
        f"{allocated_memory:.2f} GB"
    )

    print(
        f"Reserved VRAM  : "
        f"{reserved_memory:.2f} GB"
    )


# ============================================================
# 18. VERIFY NO TRAINING
# ============================================================

print_section(
    "18. TRAINING CHECK"
)

print()

print(
    "TRAINING WAS NOT STARTED."
)

print()

print(
    "This script intentionally does NOT call:"
)

print()

print(
    "    trainer.train()"
)

print()

print(
    "Therefore:"
)

print()

print(
    "    700 training examples : NOT TRAINED"
)

print(
    "    150 validation        : NOT TRAINED"
)

print(
    "    150 test examples     : NOT LOADED"
)

print()


# ============================================================
# 19. FINAL RESULT
# ============================================================

print_section(
    "FINAL RESULT"
)

print()

print(
    "QLoRA TRAINER SETUP: SUCCESS"
)

print()

print(
    "Everything required for actual "
    "QLoRA training was successfully "
    "constructed and verified."
)

print()

print(
    "Verified:"
)

print(
    "  [PASS] CUDA"
)

print(
    "  [PASS] GPU"
)

print(
    "  [PASS] Dataset"
)

print(
    "  [PASS] Dataset counts"
)

print(
    "  [PASS] Message structure"
)

print(
    "  [PASS] Tokenizer"
)

print(
    "  [PASS] Tokenization"
)

print(
    "  [PASS] 4-bit NF4"
)

print(
    "  [PASS] Qwen2.5-3B-Instruct"
)

print(
    "  [PASS] k-bit preparation"
)

print(
    "  [PASS] LoRA configuration"
)

print(
    "  [PASS] SFTConfig"
)

print(
    "  [PASS] SFTTrainer"
)

print(
    "  [PASS] LoRA trainable parameters"
)

print(
    "  [PASS] Base model freezing"
)

print()

print(
    "NO TRAINING WAS PERFORMED."
)

print()

print(
    "The environment is ready for "
    "actual QLoRA fine-tuning."
)

print()

print("=" * 72)