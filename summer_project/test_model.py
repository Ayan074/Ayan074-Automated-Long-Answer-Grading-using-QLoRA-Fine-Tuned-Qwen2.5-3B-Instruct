import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded successfully.")
print("Device:", next(model.parameters()).device)

messages = [
    {
        "role": "system",
        "content": "You are an expert academic evaluator."
    },
    {
        "role": "user",
        "content": """Question:
What is Gradient descent? Explain Batch, stochastic and mini batch gradient descent comparatively.

Rubric:
1. Defines gradient descent as an optimization algorithm — 1 point
2. States it minimizes the loss or cost function — 1 point
3. States it uses gradients/derivatives to update parameters iteratively — 1 point
4. Mentions learning rate as hyperparameter controlling step size — 1 point
5. Correctly describes batch GD as using entire dataset per update — 1 point
6. Correctly describes SGD as using one sample per update — 1 point
7. Correctly describes mini-batch GD as using a subset of samples per update — 1 point
8. Compares computational cost across the three variants — 1 point
9. Compares convergence behavior — 1 point
10. States mini-batch GD is the most commonly used variant in practice — 1 point

Student Answer:
Learning rate (alpha) decides the size of steps taken during optimization. Batch gives stable gradient estimates, SGD's high variance aids generalization, mini-batch offers reduced variance compared to SGD. With SGD, you randomly pick one sample, compute the gradient, and update immediately. Batch GD is faster because it processes all data at once in parallel. Practically speaking, mini-batch GD is what everyone uses for training neural networks.

Instructions:
Evaluate the student's answer using ONLY the rubric above.

Output ONLY the following 11 lines.

Criterion 1: X/1
Criterion 2: X/1
Criterion 3: X/1
Criterion 4: X/1
Criterion 5: X/1
Criterion 6: X/1
Criterion 7: X/1
Criterion 8: X/1
Criterion 9: X/1
Criterion 10: X/1
Total: X/10

Replace each X with either 0 or 1.

Do NOT provide explanations.
Do NOT provide reasoning.
Do NOT repeat the rubric.
Do NOT output any additional text."""
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
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\n========== MODEL RESPONSE ==========\n")
print(response)
