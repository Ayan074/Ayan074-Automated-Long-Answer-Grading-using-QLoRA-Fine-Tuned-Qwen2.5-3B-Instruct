import os
import sys
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Your existing Hugging Face cache
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

# Small test prompt
TEST_PROMPT = (
    "Evaluate this student answer according to the rubric.\n\n"
    "Question: What is machine learning?\n\n"
    "Rubric:\n"
    "1. Defines machine learning — Maximum 2 point(s)\n"
    "2. Mentions learning from data — Maximum 2 point(s)\n\n"
    "Student Answer:\n"
    "Machine learning is when computers learn from data."
)


# ============================================================
# PRINT SECTION
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 1. PYTHON / PYTORCH CHECK
# ============================================================

print_section("1. PYTHON / PYTORCH CHECK")

print(f"Python version : {sys.version}")
print(f"PyTorch version: {torch.__version__}")

print(f"CUDA available : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "QLoRA cannot be tested on the GPU."
    )

print(f"CUDA version   : {torch.version.cuda}")
print(f"GPU count      : {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}         : {torch.cuda.get_device_name(i)}")

    total_memory = torch.cuda.get_device_properties(i).total_memory
    total_memory_gb = total_memory / (1024 ** 3)

    print(f"VRAM {i}        : {total_memory_gb:.2f} GB")


# ============================================================
# 2. HUGGING FACE CACHE CHECK
# ============================================================

print_section("2. HUGGING FACE MODEL CACHE CHECK")

print(f"HF cache directory:")
print(HF_CACHE_DIR)

model_cache_dir = os.path.join(
    HF_CACHE_DIR,
    "models--Qwen--Qwen2.5-3B-Instruct"
)

if os.path.isdir(model_cache_dir):
    print("Qwen model cache: FOUND")
    print(model_cache_dir)
else:
    print("Qwen model cache: NOT FOUND")
    print(
        "The model may need to be downloaded from Hugging Face."
    )


# ============================================================
# 3. BITSANDBYTES CHECK
# ============================================================

print_section("3. BITSANDBYTES CHECK")

try:
    import bitsandbytes as bnb

    print(f"BitsAndBytes version: {bnb.__version__}")
    print("BitsAndBytes import: SUCCESS")

except Exception as e:
    raise RuntimeError(
        f"BitsAndBytes import failed:\n{e}"
    )


# ============================================================
# 4. PEFT CHECK
# ============================================================

print_section("4. PEFT CHECK")

try:
    import peft

    print(f"PEFT version: {peft.__version__}")
    print("PEFT import: SUCCESS")

except Exception as e:
    raise RuntimeError(
        f"PEFT import failed:\n{e}"
    )


# ============================================================
# 5. LOAD TOKENIZER
# ============================================================

print_section("5. LOADING TOKENIZER")

print(f"Model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=HF_CACHE_DIR,
)

print("Tokenizer loaded successfully.")

print(f"Tokenizer vocabulary size: {len(tokenizer)}")


# ============================================================
# 6. CONFIGURE 4-BIT QUANTIZATION
# ============================================================

print_section("6. CONFIGURING 4-BIT QUANTIZATION")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("4-bit configuration created successfully.")

print("Quantization type : NF4")
print("Double quant      : True")
print("Compute dtype     : float16")


# ============================================================
# 7. LOAD QWEN MODEL
# ============================================================

print_section("7. LOADING QWEN2.5-3B-INSTRUCT IN 4-BIT")

print("Loading model...")
print("This may take some time.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir=HF_CACHE_DIR,
)

print()
print("MODEL LOADING: SUCCESS")


# ============================================================
# 8. CHECK MODEL DEVICE
# ============================================================

print_section("8. MODEL DEVICE CHECK")

try:
    print(f"Model device: {model.device}")
except Exception:
    print("Model uses device_map='auto'.")


# ============================================================
# 9. PRINT GPU MEMORY
# ============================================================

print_section("9. GPU MEMORY AFTER MODEL LOADING")

for i in range(torch.cuda.device_count()):

    allocated = torch.cuda.memory_allocated(i)
    reserved = torch.cuda.memory_reserved(i)

    allocated_gb = allocated / (1024 ** 3)
    reserved_gb = reserved / (1024 ** 3)

    print(f"GPU {i}:")
    print(f"  Allocated: {allocated_gb:.2f} GB")
    print(f"  Reserved : {reserved_gb:.2f} GB")


# ============================================================
# 10. PREPARE MODEL FOR LORA
# ============================================================

print_section("10. PREPARING MODEL FOR LORA")

try:

    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    print("prepare_model_for_kbit_training: SUCCESS")

except Exception as e:

    raise RuntimeError(
        "Failed to prepare model for k-bit training:\n"
        f"{e}"
    )


# ============================================================
# 11. CREATE LORA CONFIGURATION
# ============================================================

print_section("11. CREATING LORA CONFIGURATION")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

print("LoRA configuration created successfully.")

print(f"LoRA rank       : {lora_config.r}")
print(f"LoRA alpha      : {lora_config.lora_alpha}")
print(f"LoRA dropout    : {lora_config.lora_dropout}")

print("Target modules:")
for module in lora_config.target_modules:
    print(f"  - {module}")


# ============================================================
# 12. ATTACH LORA ADAPTERS
# ============================================================

print_section("12. ATTACHING LORA ADAPTERS")

try:

    model = get_peft_model(
        model,
        lora_config
    )

    print("LoRA adapter attachment: SUCCESS")

except Exception as e:

    raise RuntimeError(
        f"Failed to attach LoRA adapters:\n{e}"
    )


# ============================================================
# 13. PRINT TRAINABLE PARAMETERS
# ============================================================

print_section("13. TRAINABLE PARAMETER CHECK")

model.print_trainable_parameters()


# ============================================================
# 14. TOKENIZER TEST
# ============================================================

print_section("14. TOKENIZATION TEST")

inputs = tokenizer(
    TEST_PROMPT,
    return_tensors="pt"
)

print(
    f"Input token count: "
    f"{inputs['input_ids'].shape[1]}"
)


# ============================================================
# 15. MOVE INPUT TO GPU
# ============================================================

print_section("15. GPU INPUT TEST")

device = next(model.parameters()).device

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

print(f"Input device: {inputs['input_ids'].device}")


# ============================================================
# 16. FORWARD PASS TEST
# ============================================================

print_section("16. FORWARD PASS TEST")

model.eval()

with torch.no_grad():

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

print("Forward pass: SUCCESS")

print(f"Logits shape: {outputs.logits.shape}")


# ============================================================
# 17. GENERATION TEST
# ============================================================

print_section("17. GENERATION TEST")

with torch.no_grad():

    generated = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        do_sample=False,
    )

generated_text = tokenizer.decode(
    generated[0],
    skip_special_tokens=True
)

print("Generation: SUCCESS")

print()
print("Generated output:")
print("-" * 70)
print(generated_text)
print("-" * 70)


# ============================================================
# 18. FINAL GPU MEMORY
# ============================================================

print_section("18. FINAL GPU MEMORY")

for i in range(torch.cuda.device_count()):

    allocated = torch.cuda.memory_allocated(i)
    reserved = torch.cuda.memory_reserved(i)

    allocated_gb = allocated / (1024 ** 3)
    reserved_gb = reserved / (1024 ** 3)

    total_memory = torch.cuda.get_device_properties(
        i
    ).total_memory

    total_memory_gb = total_memory / (1024 ** 3)

    print(f"GPU {i}:")
    print(f"  Total VRAM     : {total_memory_gb:.2f} GB")
    print(f"  Allocated      : {allocated_gb:.2f} GB")
    print(f"  Reserved       : {reserved_gb:.2f} GB")
    print(
        f"  Approx. free   : "
        f"{total_memory_gb - reserved_gb:.2f} GB"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print_section("QLORA SETUP TEST: SUCCESS")

print(
    "Qwen2.5-3B-Instruct successfully loaded in 4-bit mode, "
    "LoRA adapters were attached, and a forward/generation "
    "test completed successfully."
)

print()
print("The environment is ready for the next stage:")
print("ACTUAL QLoRA FINE-TUNING")
print()