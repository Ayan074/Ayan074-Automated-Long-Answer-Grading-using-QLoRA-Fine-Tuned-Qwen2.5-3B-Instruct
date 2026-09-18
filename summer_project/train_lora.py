import os
import json
import torch

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

from trl import SFTConfig, SFTTrainer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_FILE = "dataset_split/train.jsonl"
VALIDATION_FILE = "dataset_split/validation.jsonl"

OUTPUT_DIR = "./qwen2.5-3b-grader-qlora"

MAX_LENGTH = 768

NUM_EPOCHS = 3

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 2e-4

WARMUP_STEPS = 20

WEIGHT_DECAY = 0.01

LOGGING_STEPS = 10

SEED = 42


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 72)
print("QWEN2.5-3B-INSTRUCT QLoRA FINE-TUNING")
print("=" * 72)

print()
print("Model              :", MODEL_NAME)
print("Train dataset      :", TRAIN_FILE)
print("Validation dataset :", VALIDATION_FILE)
print("Output directory    :", OUTPUT_DIR)
print("Max sequence length:", MAX_LENGTH)
print("Epochs              :", NUM_EPOCHS)
print("Train batch size    :", PER_DEVICE_TRAIN_BATCH_SIZE)
print("Gradient accumulation:", GRADIENT_ACCUMULATION_STEPS)
print("Effective batch size:",
      PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
print("Learning rate       :", LEARNING_RATE)
print("Warmup steps        :", WARMUP_STEPS)
print()


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 72)
print("1. GPU CHECK")
print("=" * 72)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. GPU training cannot continue.")

print("CUDA available :", True)
print("CUDA version   :", torch.version.cuda)
print("GPU            :", torch.cuda.get_device_name(0))

gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f"GPU VRAM       : {gpu_memory:.2f} GB")
print()


# ============================================================
# DATASET CHECK
# ============================================================

print("=" * 72)
print("2. DATASET CHECK")
print("=" * 72)

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation file not found: {VALIDATION_FILE}"
    )

print("Train file      :", TRAIN_FILE)
print("Validation file :", VALIDATION_FILE)


# Load datasets independently.
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

print()
print("Training examples   :", len(train_dataset))
print("Validation examples :", len(validation_dataset))

if len(train_dataset) != 700:
    raise RuntimeError(
        f"Expected 700 training examples, found {len(train_dataset)}"
    )

if len(validation_dataset) != 150:
    raise RuntimeError(
        f"Expected 150 validation examples, found {len(validation_dataset)}"
    )

print("Dataset count check : PASS")
print()


# ============================================================
# MESSAGE STRUCTURE CHECK
# ============================================================

print("=" * 72)
print("3. MESSAGE STRUCTURE CHECK")
print("=" * 72)


def validate_message_structure(dataset, dataset_name):

    for index, example in enumerate(dataset):

        if "messages" not in example:
            raise ValueError(
                f"{dataset_name} record {index + 1}: "
                f"'messages' field missing."
            )

        messages = example["messages"]

        if not isinstance(messages, list):
            raise ValueError(
                f"{dataset_name} record {index + 1}: "
                f"'messages' must be a list."
            )

        if len(messages) != 3:
            raise ValueError(
                f"{dataset_name} record {index + 1}: "
                f"Expected 3 messages, found {len(messages)}."
            )

        expected_roles = ["system", "user", "assistant"]

        for message_index, expected_role in enumerate(expected_roles):

            message = messages[message_index]

            if not isinstance(message, dict):
                raise ValueError(
                    f"{dataset_name} record {index + 1}, "
                    f"message {message_index + 1}: "
                    f"must be an object."
                )

            if message.get("role") != expected_role:
                raise ValueError(
                    f"{dataset_name} record {index + 1}, "
                    f"message {message_index + 1}: "
                    f"expected role '{expected_role}', "
                    f"found '{message.get('role')}'."
                )

            if "content" not in message:
                raise ValueError(
                    f"{dataset_name} record {index + 1}, "
                    f"message {message_index + 1}: "
                    f"'content' missing."
                )

    print(f"{dataset_name} message structure: PASS")


validate_message_structure(train_dataset, "TRAIN")
validate_message_structure(validation_dataset, "VALIDATION")

print()


# ============================================================
# TOKENIZER
# ============================================================

print("=" * 72)
print("4. LOADING TOKENIZER")
print("=" * 72)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print("Tokenizer loaded successfully.")
print("Vocabulary size :", len(tokenizer))
print("EOS token       :", tokenizer.eos_token)
print("PAD token       :", tokenizer.pad_token)

# Qwen normally has a pad token, but make sure one exists.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer check  : PASS")
print()


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

print("=" * 72)
print("5. CONFIGURING 4-BIT QUANTIZATION")
print("=" * 72)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("Load in 4-bit       : True")
print("Quantization type   : NF4")
print("Double quantization : True")
print("Compute dtype       : float16")
print()


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("=" * 72)
print("6. LOADING QWEN2.5-3B-INSTRUCT")
print("=" * 72)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("Model loaded successfully.")

print("Model device:", model.device)

print()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

model.config.use_cache = False

print("=" * 72)
print("7. PREPARING MODEL FOR QLoRA")
print("=" * 72)

model = prepare_model_for_kbit_training(model)

print("prepare_model_for_kbit_training: SUCCESS")
print()


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("=" * 72)
print("8. CONFIGURING LoRA")
print("=" * 72)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",

    task_type="CAUSAL_LM",
)

print("LoRA rank       : 16")
print("LoRA alpha      : 32")
print("LoRA dropout    : 0.05")

print("Target modules:")
for module in lora_config.target_modules:
    print("  -", module)

print()


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print("=" * 72)
print("9. CREATING SFT CONFIGURATION")
print("=" * 72)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Training
    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    warmup_steps=WARMUP_STEPS,

    weight_decay=WEIGHT_DECAY,

    # Optimizer
    optim="paged_adamw_8bit",

    # Precision
    fp16=False,
    bf16=False,

    # Memory
    gradient_checkpointing=True,

    # Sequence length
    max_length=MAX_LENGTH,

    # Dataset processing
    packing=False,

    # Evaluation
    eval_strategy="epoch",

    # Saving
    save_strategy="epoch",
    save_total_limit=2,

    # Logging
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,

    # Reproducibility
    seed=SEED,

    # Do not report to external services
    report_to="none",

    # Keep best model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Disable hub upload
    push_to_hub=False,

    # Dataset contains messages
    dataset_kwargs={
        "skip_prepare_dataset": False,
    },

    # Padding
    # pad_token=tokenizer.pad_token,
)

print("SFT configuration created successfully.")
print()
print("Max sequence length :", MAX_LENGTH)
print("Epochs               :", NUM_EPOCHS)
print("Train batch size     :", PER_DEVICE_TRAIN_BATCH_SIZE)
print("Gradient accumulation:", GRADIENT_ACCUMULATION_STEPS)
print("Effective batch size :",
      PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
print("Learning rate        :", LEARNING_RATE)
print("Warmup steps         :", WARMUP_STEPS)
print("Optimizer            : paged_adamw_8bit")
print("Precision            : FP16")
print("Gradient checkpoint  : True")
print("Evaluation           : Every epoch")
print("Saving               : Every epoch")
print()


# ============================================================
# CREATE TRAINER
# ============================================================

print("=" * 72)
print("10. CREATING SFT TRAINER")
print("=" * 72)

trainer = SFTTrainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    peft_config=lora_config,
)

print("SFTTrainer created successfully.")
print()


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

print("=" * 72)
print("11. TRAINABLE PARAMETER CHECK")
print("=" * 72)

trainable_params = 0
all_params = 0

for param in model.parameters():

    all_params += param.numel()

    if param.requires_grad:
        trainable_params += param.numel()

trainable_percentage = (
    100 * trainable_params / all_params
)

print(f"Trainable parameters : {trainable_params:,}")
print(f"All parameters       : {all_params:,}")
print(f"Trainable percentage : {trainable_percentage:.4f}%")

print()


# ============================================================
# GPU MEMORY BEFORE TRAINING
# ============================================================

print("=" * 72)
print("12. GPU MEMORY BEFORE TRAINING")
print("=" * 72)

torch.cuda.empty_cache()

allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

print(f"Allocated VRAM : {allocated:.2f} GB")
print(f"Reserved VRAM  : {reserved:.2f} GB")

print()


# ============================================================
# START TRAINING
# ============================================================

print("=" * 72)
print("13. STARTING QLoRA TRAINING")
print("=" * 72)

print()
print("IMPORTANT:")
print("Actual training starts now.")
print()
print("Training examples   : 700")
print("Validation examples : 150")
print("Epochs              :", NUM_EPOCHS)
print("Effective batch     : 8")
print("Max sequence length :", MAX_LENGTH)
print()

print("Starting trainer.train()...")
print()

train_result = trainer.train()


# ============================================================
# TRAINING FINISHED
# ============================================================

print()
print("=" * 72)
print("14. TRAINING COMPLETED")
print("=" * 72)

print()

print("Training finished successfully.")

print()
print("Training metrics:")

if train_result.metrics:

    for key, value in train_result.metrics.items():
        print(f"  {key}: {value}")

print()


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

print("=" * 72)
print("15. SAVING LoRA ADAPTER")
print("=" * 72)

FINAL_ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "final_adapter"
)

trainer.save_model(FINAL_ADAPTER_DIR)

tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

print("LoRA adapter saved to:")
print(FINAL_ADAPTER_DIR)

print()


# ============================================================
# FINAL EVALUATION ON VALIDATION SET
# ============================================================

print("=" * 72)
print("16. FINAL VALIDATION EVALUATION")
print("=" * 72)

eval_metrics = trainer.evaluate()

print()

print("Validation metrics:")

for key, value in eval_metrics.items():
    print(f"  {key}: {value}")

print()


# ============================================================
# SAVE TRAINING METRICS
# ============================================================

metrics_file = os.path.join(
    OUTPUT_DIR,
    "training_metrics.json"
)

with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(
        {
            "train_metrics": train_result.metrics,
            "validation_metrics": eval_metrics,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("Training metrics saved to:")
print(metrics_file)

print()


# ============================================================
# FINAL GPU MEMORY
# ============================================================

print("=" * 72)
print("17. FINAL GPU MEMORY")
print("=" * 72)

allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

print(f"Allocated VRAM : {allocated:.2f} GB")
print(f"Reserved VRAM  : {reserved:.2f} GB")

print()


# ============================================================
# FINAL RESULT
# ============================================================

print("=" * 72)
print("QLoRA FINE-TUNING COMPLETE")
print("=" * 72)

print()
print("Base model:")
print("  ", MODEL_NAME)

print()
print("Training:")
print("  Train examples      : 700")
print("  Validation examples : 150")
print("  Epochs              :", NUM_EPOCHS)
print("  Effective batch size:",
      PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)

print()
print("LoRA:")
print("  Rank                : 16")
print("  Alpha               : 32")
print("  Dropout             : 0.05")

print()
print("Output:")
print("  ", OUTPUT_DIR)

print()
print("Final adapter:")
print("  ", FINAL_ADAPTER_DIR)

print()
print("Training completed successfully.")
print("=" * 72)