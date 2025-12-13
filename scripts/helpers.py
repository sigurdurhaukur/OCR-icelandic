from dataclasses import asdict, dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    hf_dataset_id: str = "arnastofnun/IGC-2024"
    hf_data_directory: str = "wiki"
    max_length: int = 512
    max_entries: int = 0  # 0 = all
    max_eval_entries: int = 10
    text_key: str = "document"


@dataclass
class ModelConfig:
    """Model and deployment configuration."""

    model_id: str = "HuggingFaceTB/SmolVLM-Base"
    push_to_hub: bool = False
    hub_repo_id: str = "Sigurdur/SmolVLM-Base-ICELANDIC"


@dataclass
class LoRAConfig:
    """LoRA-specific configuration."""

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1


@dataclass
class HFTrainingConfig:
    """Configuration that maps directly to HuggingFace TrainingArguments."""

    output_dir: str = "./lora_results"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    logging_steps: int = 50
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    eval_strategy: str = "steps"
    fp16: bool = False
    bf16: bool = True
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "wandb"

    def to_training_args(self, **overrides) -> TrainingArguments:
        """Convert to TrainingArguments with optional overrides."""
        args_dict = asdict(self)
        args_dict.update(overrides)
        return TrainingArguments(**args_dict)


@dataclass
class WandBConfig:
    """Weights & Biases logging configuration."""

    entity: str = "sigurdurhaukur-team"
    project: str = "smolVLM"
    run_name: str = "lora-finetune-icelandic"
    run_description: str = (
        "LoRA fine-tuning of SmolVLM text model on Icelandic text data"
    )
    tags: list = field(
        default_factory=lambda: [
            "LoRA",
            "Idefics3",
            "SmolVLM",
            "Icelandic",
            "Fine-tuning",
            "NLP",
            "Vision-Language Model",
        ]
    )


@dataclass
class TrainConfig:
    """Top-level configuration combining all components."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: HFTrainingConfig = field(default_factory=HFTrainingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def __post_init__(self):
        """Convert dict fields to proper dataclass instances."""
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(**self.dataset)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)
        if isinstance(self.training, dict):
            self.training = HFTrainingConfig(**self.training)
        if isinstance(self.wandb, dict):
            self.wandb = WandBConfig(**self.wandb)

    @staticmethod
    def from_flat_dict(flat_config: dict) -> "TrainConfig":
        """
        Convert flat CLI arguments to nested config structure.
        Supports both flat keys (e.g., 'model_id') and nested keys (e.g., 'model.model_id').
        """
        # Define mapping from flat keys to nested paths
        flat_to_nested = {
            # Dataset fields
            "hf_dataset_id": ("dataset", "hf_dataset_id"),
            "hf_data_directory": ("dataset", "hf_data_directory"),
            "max_length": ("dataset", "max_length"),
            "max_entries": ("dataset", "max_entries"),
            "max_eval_entries": ("dataset", "max_eval_entries"),
            "text_key": ("dataset", "text_key"),
            # Model fields
            "model_id": ("model", "model_id"),
            "push_to_hub": ("model", "push_to_hub"),
            "hub_repo_id": ("model", "hub_repo_id"),
            # LoRA fields
            "lora_r": ("lora", "lora_r"),
            "lora_alpha": ("lora", "lora_alpha"),
            "lora_dropout": ("lora", "lora_dropout"),
            # Training fields
            "output_dir": ("training", "output_dir"),
            "num_train_epochs": ("training", "num_train_epochs"),
            "per_device_train_batch_size": ("training", "per_device_train_batch_size"),
            "per_device_eval_batch_size": ("training", "per_device_eval_batch_size"),
            "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
            "learning_rate": ("training", "learning_rate"),
            "lr_scheduler_type": ("training", "lr_scheduler_type"),
            "warmup_steps": ("training", "warmup_steps"),
            "logging_steps": ("training", "logging_steps"),
            "eval_steps": ("training", "eval_steps"),
            "save_strategy": ("training", "save_strategy"),
            "save_steps": ("training", "save_steps"),
            "save_total_limit": ("training", "save_total_limit"),
            "load_best_model_at_end": ("training", "load_best_model_at_end"),
            "eval_strategy": ("training", "eval_strategy"),
            "fp16": ("training", "fp16"),
            "bf16": ("training", "bf16"),
            "dataloader_drop_last": ("training", "dataloader_drop_last"),
            "remove_unused_columns": ("training", "remove_unused_columns"),
            "metric_for_best_model": ("training", "metric_for_best_model"),
            "greater_is_better": ("training", "greater_is_better"),
            "report_to": ("training", "report_to"),
            # WandB fields
            "entity": ("wandb", "entity"),
            "project": ("wandb", "project"),
            "run_name": ("wandb", "run_name"),
            "run_description": ("wandb", "run_description"),
            "tags": ("wandb", "tags"),
        }

        # Build nested structure
        nested_config = {}
        for key, value in flat_config.items():
            if key in flat_to_nested:
                section, field_name = flat_to_nested[key]
                if section not in nested_config:
                    nested_config[section] = {}
                nested_config[section][field_name] = value
            else:
                # Already nested or unknown key - keep as is
                nested_config[key] = value

        return nested_config

    # Legacy compatibility - expose commonly used fields at top level
    @property
    def model_id(self) -> str:
        return self.model.model_id

    @property
    def hf_dataset_id(self) -> str:
        return self.dataset.hf_dataset_id

    @property
    def hf_data_directory(self) -> str:
        return self.dataset.hf_data_directory

    @property
    def max_length(self) -> int:
        return self.dataset.max_length

    @property
    def max_entries(self) -> int:
        return self.dataset.max_entries

    @property
    def max_eval_entries(self) -> int:
        return self.dataset.max_eval_entries

    @property
    def text_key(self) -> str:
        return self.dataset.text_key

    @property
    def push_to_hub(self) -> bool:
        return self.model.push_to_hub

    @property
    def hub_repo_id(self) -> str:
        return self.model.hub_repo_id

    @property
    def lora_r(self) -> int:
        return self.lora.lora_r

    @property
    def lora_alpha(self) -> int:
        return self.lora.lora_alpha

    @property
    def lora_dropout(self) -> float:
        return self.lora.lora_dropout

    @property
    def output_dir(self) -> str:
        return self.training.output_dir

    @property
    def per_device_train_batch_size(self) -> int:
        return self.training.per_device_train_batch_size

    @property
    def per_device_eval_batch_size(self) -> int:
        return self.training.per_device_eval_batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.training.gradient_accumulation_steps

    @property
    def num_train_epochs(self) -> int:
        return self.training.num_train_epochs

    @property
    def learning_rate(self) -> float:
        return self.training.learning_rate

    @property
    def lr_scheduler_type(self) -> str:
        return self.training.lr_scheduler_type

    @property
    def warmup_steps(self) -> int:
        return self.training.warmup_steps

    @property
    def logging_steps(self) -> int:
        return self.training.logging_steps

    @property
    def eval_steps(self) -> int:
        return self.training.eval_steps

    @property
    def save_strategy(self) -> str:
        return self.training.save_strategy

    @property
    def save_steps(self) -> int:
        return self.training.save_steps

    @property
    def save_total_limit(self) -> int:
        return self.training.save_total_limit

    @property
    def load_best_model_at_end(self) -> bool:
        return self.training.load_best_model_at_end

    @property
    def eval_strategy(self) -> str:
        return self.training.eval_strategy

    @property
    def fp16(self) -> bool:
        return self.training.fp16

    @property
    def bf16(self) -> bool:
        return self.training.bf16

    @property
    def dataloader_drop_last(self) -> bool:
        return self.training.dataloader_drop_last

    @property
    def remove_unused_columns(self) -> bool:
        return self.training.remove_unused_columns

    @property
    def metric_for_best_model(self) -> str:
        return self.training.metric_for_best_model

    @property
    def greater_is_better(self) -> bool:
        return self.training.greater_is_better

    @property
    def report_to(self) -> str:
        return self.training.report_to

    @property
    def entity(self) -> str:
        return self.wandb.entity

    @property
    def project(self) -> str:
        return self.wandb.project
