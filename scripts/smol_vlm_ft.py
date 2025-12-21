import logging
import sys
from dataclasses import asdict

import evaluate
import numpy as np
import peft
import torch
import transformers
from datasets import load_dataset
from helpers import TrainConfig
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
)

import wandb

# Load metrics once
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fintune_smolvlm_ocr(cfg: TrainConfig) -> None:
    """Fine-tune SmolVLM on Icelandic OCR dataset."""
    USE_LORA = True
    USE_QLORA = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = AutoProcessor.from_pretrained(
            cfg.model.model_id
        )  # use the processor from the base model

    except Exception as e:
        logger.error(f"Error loading processor for model {cfg.model.model_id}: {e}")
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Base")
        logger.info("Loaded default processor HuggingFaceTB/SmolVLM-Base")

    # our custom model, with pre-trained LLM backbone on Icelandic text
    model_id = cfg.model.model_id  # ./full_idefics3_lora_merged
    logging.info(f"Loading model {model_id} with QLoRA: {USE_QLORA}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=cfg.lora.lora_r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=[
                "down_proj",
                "o_proj",
                "k_proj",
                "q_proj",
                "gate_proj",
                "up_proj",
                "v_proj",
            ],
            # use_dora=False if USE_QLORA else True,
            use_dora=False,
            init_lora_weights="gaussian",
        )
        lora_config.inference_mode = False
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            # _attn_implementation="flash_attention_2",
            device_map="auto",
        )
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        print(
            f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )

        if USE_QLORA:
            model = prepare_model_for_kbit_training(model)

    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2",
        ).to(DEVICE)

        # if you'd like to only fine-tune LLM
        for param in model.model.vision_model.parameters():
            param.requires_grad = False

    ds = load_dataset(cfg.hf_dataset_id)

    # blacklist bad fonts
    font_blacklist = [
        "AppleGothic.ttf",
        "Bodoni Ornaments.ttf",
        "Bodoni 72 Smallcaps Book.ttf",
    ]

    def filter_bad_fonts(example: dict) -> bool:
        """
        Filter out examples with blacklisted fonts.
        Args:
            example (dict): A dataset example containing a "font_path" key.
        Returns:
            bool: True if the example's font is not blacklisted, False otherwise.
        """
        return not any(bad_font in example["font_path"] for bad_font in font_blacklist)

    # filter the dataset to remove bad fonts
    ds = ds.filter(filter_bad_fonts)

    train_ds = ds["train"]

    # limit dataset size for faster experimentation
    validation_ds = ds["validation"].select(range(min(5, len(ds["validation"]))))
    english_validation_ds = load_dataset(
        "Sigurdur/eng_synthetic_ocr", split="validation"
    ).select(range(5))

    multiple_validation_ds = {
        "icelandic": validation_ds,
        "english": english_validation_ds,
    }

    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]

    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            task_desc = "Extract the text from the image."
            answer = example["text"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": task_desc},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    def compute_text_metrics(predictions, labels):
        """Compute comprehensive OCR metrics."""
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Handle logits
        if predictions.ndim > 2:
            predictions = np.argmax(predictions, axis=-1)

        # Decode predictions and labels
        label_ids = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute standard metrics
        wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # BLEU score (references need to be wrapped in lists for multiple references support)
        bleu = bleu_metric.compute(
            predictions=decoded_preds, references=[[ref] for ref in decoded_labels]
        )

        # Character n-gram F-score (chrF)
        chrf = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # Special Icelandic characters
        special_chars = set("þðáéíóúýæö")

        # Initialize custom metrics
        exact_matches = 0
        special_correct = 0
        special_total = 0
        seq_acc_5 = 0
        seq_acc_10 = 0

        for pred, label in zip(decoded_preds, decoded_labels):
            # Exact match
            if pred == label:
                exact_matches += 1

            # Character error rate for this sample
            sample_cer = cer_metric.compute(predictions=[pred], references=[label])

            # Sequence accuracy thresholds
            if sample_cer < 0.05:
                seq_acc_5 += 1
            if sample_cer < 0.10:
                seq_acc_10 += 1

            # Special character accuracy (count-based)
            for char in special_chars:
                label_count = label.lower().count(char)
                pred_count = pred.lower().count(char)
                special_total += label_count
                special_correct += min(label_count, pred_count)

        n = len(decoded_labels)
        return {
            # Standard OCR metrics
            "wer": wer,
            "cer": cer,
            "bleu": bleu["bleu"],  # BLEU returns a dict with multiple scores
            "chrf": chrf["score"],  # chrF returns a dict
            # Custom metrics
            "exact_match": exact_matches / n,
            "special_char_acc": special_correct / max(special_total, 1),
            "seq_acc_5": seq_acc_5 / n,
            "seq_acc_10": seq_acc_10 / n,
        }

    def compute_metrics(eval_preds):
        """Compute metrics function for Trainer."""
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids

        # Handle tuple predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        return compute_text_metrics(predictions, labels)

    # Initialize wandb
    # Convert config to dict and add runtime info
    config_dict = asdict(cfg)
    config_dict.update(
        {
            # Hardware info
            "device": DEVICE,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "CPU",
            # Dataset info
            "total_train_dataset_size": len(train_ds),
            "total_eval_dataset_size": len(validation_ds),
            "effective_batch_size": cfg.training.per_device_train_batch_size
            * cfg.training.gradient_accumulation_steps,
            "total_training_steps": len(train_ds)
            // (
                cfg.training.per_device_train_batch_size
                * cfg.training.gradient_accumulation_steps
            )
            * cfg.training.num_train_epochs,
            # Model architecture
            "model_size": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "lora_target_modules": lora_config.target_modules
            if USE_LORA or USE_QLORA
            else None,
            # Environment
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
        }
    )

    # Initialize wandb
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=config_dict,
    )

    # Use the new config structure to create TrainingArguments cleanly
    training_args = cfg.training.to_training_args(
        output_dir=cfg.training.output_dir,
        hub_model_id=cfg.training.hf_hub_output_model_id,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=multiple_validation_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub()


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()

    # Convert flat CLI args to nested structure
    cli_dict = OmegaConf.to_container(cli_cfg, resolve=True)
    nested_cli = TrainConfig.from_flat_dict(cli_dict)

    # Merge with default config
    cfg = OmegaConf.merge(cfg, nested_cli)
    cfg = OmegaConf.to_container(cfg, resolve=True)  # try not converting to dict

    try:
        cfg = TrainConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error("Error: %s\n\nUsage: python smol_vlm_ft.py", e)
        sys.exit(1)

    fintune_smolvlm_ocr(cfg)


if __name__ == "__main__":
    main()
