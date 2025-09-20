"""
LoRA fine-tuning for the text model within Idefics3.

This script extracts the text model from Idefics3, applies LoRA for efficient fine-tuning,
and prepares the model for training on a custom dataset.

Core idea is to embed icelandic language understanding into the text model, before trying to
fine-tune the full Idefics3 model on image-text pairs.
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Idefics3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

model_id = "HuggingFaceTB/SmolVLM-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the original model
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # *attn*implementation="flash_attention_2",
).to(DEVICE)

# Load just the text model (Llama) directly
config = model.config.text_config
text_model = AutoModelForCausalLM.from_config(config)

# The text_model from Idefics3 doesn't have the outer 'model' wrapper
# but AutoModelForCausalLM expects it. So we need to adjust the state dict:
# Get the state dict from Idefics3's text model
source_state = model.model.text_model.state_dict()

# Add the 'model.' prefix to match what AutoModelForCausalLM expects
remapped_state = {}
for key, value in source_state.items():
    remapped_state[f"model.{key}"] = value

# Also add the lm_head weights
remapped_state["lm_head.weight"] = model.lm_head.weight

# Now load everything at once
text_model.load_state_dict(remapped_state, strict=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # Rank of adaptation - higher values = more parameters but potentially better performance
    lora_alpha=32,  # LoRA scaling parameter - typically 2x the rank
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=[
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
        "gate_proj",  # Gate projection (for feedforward)
        "up_proj",  # Up projection (for feedforward)
        "down_proj",  # Down projection (for feedforward)
    ],
    bias="none",  # Whether to train bias parameters
    use_rslora=False,  # Use rank-stabilized LoRA
)

# Apply LoRA to the model
text_model = get_peft_model(text_model, lora_config)

# Move to device
text_model.to(DEVICE)

# Print trainable parameters info
text_model.print_trainable_parameters()


# Inference function for text generation
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = text_model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.2,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Example usage
print("Before training:")
print(generate_text("Once upon a time in a land far, far away,"))

# Load and prepare dataset
ds = load_dataset("arnastofnun/IGC-2024", "wiki", split="train[:1%]")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # False for causal LM (GPT-style), True for masked LM (BERT-style)
)


# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["document"],
        truncation=True,
        padding="max_length",
        max_length=512,  # Adjust based on your needs and memory
    )


train_dataset = ds.map(
    tokenize_function,
    batched=True,
    remove_columns=ds.column_names,  # Remove original columns to save memory
)

# Training arguments - adjusted for LoRA
training_args = TrainingArguments(
    output_dir="./lora_results",
    per_device_train_batch_size=4,  # Can use larger batch size with LoRA
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,  # Slightly higher learning rate is often good for LoRA
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no",
    fp16=True,  # or bf16=True if supported
    dataloader_drop_last=True,
    remove_unused_columns=False,
    report_to="none",  # Disable wandb/tensorboard logging
)

# Create trainer
trainer = Trainer(
    model=text_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Train the model
print("Starting LoRA training...")
trainer.train()

# Save the LoRA adapter
text_model.save_pretrained("./lora_adapter")

print("Training complete!")
print("After training:")
print(generate_text("Once upon a time in a land far, far away,"))

# Optional: Merge LoRA weights back into the base model for inference
# (This creates a single model file but loses the memory efficiency of LoRA)
# merged_model = text_model.merge_and_unload()
# merged_model.save_pretrained("./merged_model")
