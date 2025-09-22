"""
LoRA fine-tuning for the text model within Idefics3.

This script extracts the text model from Idefics3, applies LoRA for efficient fine-tuning,
and prepares the model for training on a custom dataset.

Core idea is to embed icelandic language understanding into the text model, before trying to
fine-tune the full Idefics3 model on image-text pairs.

Before running:

1. activate the uv environment: `source .venv/bin/activate`
2. install requirements: `pip install -r requirements.txt`
3. login to Hugging Face Hub: `huggingface-cli login`
4. configure wandb with `wandb login` then `wandb init` in the root directory of this repo

usage: python train_llm.py push_to_hub=True
"""

import logging
import os
import sys
from dataclasses import dataclass

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Idefics3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    # Dataset configuration
    hf_dataset_id: str = "arnastofnun/IGC-2024"  # Hugging Face dataset ID
    hf_data_directory: str = "wiki"  # Subdirectory within the dataset
    dataset_split: str = "train[:1%]"  # Dataset split to use for
    max_length: int = 512  # Max token length for text sequences
    max_entries: int = 10  # Max entries to process from dataset (for quick testing)
    text_key: str = "document"  # Key in dataset containing text data

    # Model configuration
    model_id: str = "HuggingFaceTB/SmolVLM-Base"  # Base model ID
    push_to_hub: bool = False  # Whether to push trained model to Hugging Face Hub
    hub_repo_id: str = (
        "Sigurdur/SmolVLM-Base-ICELANDIC"  # Hugging Face repo ID to push model
    )

    # LoRA configuration
    lora_r: int = 16  # Rank of adaptation - higher values = more parameters but potentially better performance
    lora_alpha: int = 32  # LoRA scaling parameter - typically 2x the rank
    lora_dropout: float = 0.1  # Dropout for LoRA layers

    # Training arguments
    output_dir: str = (
        "./lora_results"  # Directory to save LoRA adapters and checkpoints
    )
    per_device_train_batch_size: int = 4  # Batch size per device during training
    gradient_accumulation_steps: int = 4  # Number of steps to accumulate gradients
    num_train_epochs: int = 3  # Total number of training epochs
    learning_rate: float = 1e-4  # Learning rate for optimizer
    warmup_steps: int = 100  # Number of warmup steps for learning rate scheduler
    logging_steps: int = 50  # Log every X updates steps
    save_strategy: str = "epoch"  # Save checkpoint every X epochs
    eval_strategy: str = "no"  # Evaluation strategy during training
    fp16: bool = True  # Use mixed precision training if True
    dataloader_drop_last: bool = True  # Drop last incomplete batch if True
    remove_unused_columns: bool = False  # Whether to remove unused columns in dataset
    report_to: str = "wandb"  # Report to wandb


# Inference function for text generation
def generate_text(
    text_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 50,
) -> str:
    """Generate text from the model given a prompt."""

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


def sanity_check(text_model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    """Sanity check to see if the model can generate Icelandic text."""
    # Sanity check - generate text before training
    prompt = "Einu sinni var karl og kerling sem bjuggu í"
    logger.info("Before training:")
    logger.info(generate_text(text_model, tokenizer, prompt))


def prepare_text_dataset(
    cfg: TrainConfig, tokenizer: AutoTokenizer
) -> torch.utils.data.Dataset:
    """
    Loads and tokenizes the text dataset for training.

    Args:
        cfg (TrainConfig): Configuration for dataset and training.
        tokenizer (AutoTokenizer): Tokenizer for the text model.

    Returns:
        torch.utils.data.Dataset: Tokenized dataset ready for training.
    """

    # Load and prepare dataset
    logger.info(f"Loading dataset {cfg.hf_dataset_id}...")
    ds = load_dataset(cfg.hf_dataset_id, cfg.hf_data_directory, split=cfg.dataset_split)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples[cfg.text_key],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
        )

    logger.info("Tokenizing dataset...")
    train_dataset = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=ds.column_names,  # Remove original columns to save memory
    )

    if cfg.max_entries > 0:
        train_dataset = train_dataset.select(range(cfg.max_entries))

    logger.info(f"Dataset size: {len(train_dataset)}")

    return train_dataset


def get_text_model_from_idefics3(
    model: Idefics3ForConditionalGeneration,
) -> AutoModelForCausalLM:
    """Extracts the text model (Llama) from the Idefics3 model.

    note: This function takes around 1 minute to run on a NVIDIA L40s (48 GB VRAM)

    The key steps are:
    - Load just the text model configuration from Idefics3
    - Create a new AutoModelForCausalLM instance with that config
    - Remap the state dict from Idefics3 to match what AutoModelForCausalLM expects
    - Load the remapped state dict into the new text model

    Args:
        model (Idefics3ForConditionalGeneration): The full Idefics3 model.
    Returns:
        AutoModelForCausalLM: The extracted text model.
    """

    # Load just the text model (Llama) directly
    logger.info("Loading text model from Idefics3...")
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
    logger.info("Loading state dict into text model...")
    text_model.load_state_dict(remapped_state, strict=True)

    return text_model


def fine_tune_text_model(cfg: TrainConfig) -> None:
    """
    Main function to fine-tune the text model within Idefics3 using LoRA.
    Args:
        cfg (TrainConfig): Configuration for dataset, model, and training.
    Returns:
        None
    """

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Loading model {cfg.model_id}...")

    # Load the original model
    model = Idefics3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    ).to(DEVICE)

    logger.info("Model loaded.")

    logger.info("Extracting text model from Idefics3...")
    # Extract the text model (Llama)
    text_model = get_text_model_from_idefics3(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.lora_r,  # Rank of adaptation - higher values = more parameters but potentially better performance
        lora_alpha=cfg.lora_alpha,  # LoRA scaling parameter - typically 2x the rank
        lora_dropout=cfg.lora_dropout,  # Dropout for LoRA layers
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

    # sanity check before training
    sanity_check(text_model, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # False for causal LM (GPT-style), True for masked LM (BERT-style)
    )

    train_dataset = prepare_text_dataset(cfg, tokenizer)

    # Training arguments - adjusted for LoRA
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        fp16=cfg.fp16,
        dataloader_drop_last=cfg.dataloader_drop_last,
        remove_unused_columns=cfg.remove_unused_columns,
        report_to=cfg.report_to,
    )

    # Create trainer
    trainer = Trainer(
        model=text_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train the model
    logger.info("Starting LoRA training...")
    trainer.train()

    # Save the LoRA adapter
    text_model.save_pretrained(cfg.output_dir)

    logger.info("Training complete!")
    logger.info("After training:")
    sanity_check(text_model, tokenizer)  # check if Icelandic generation works/improves

    # Merge LoRA weights back into the base model for inference
    # (This creates a single model file but loses the memory efficiency of LoRA)
    merged_model = text_model.merge_and_unload()
    merged_model.save_pretrained("./merged_model")

    # Optionally push to Hugging Face Hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        logger.info(f"Pushing model to the hub at {cfg.hub_repo_id}...")
        text_model.push_to_hub(cfg.hub_repo_id + "-lora")
        logger.info("Model pushed to the hub successfully.")
        text_model.push_to_hub(cfg.hub_repo_id)


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    fine_tune_text_model(cfg)


if __name__ == "__main__":
    main()
