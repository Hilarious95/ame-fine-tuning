import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from config import BASE_DIR, alpaca_prompt

os.environ["CC"] = "/usr/bin/gcc"

max_seq_length = 2048
local_file = os.path.join(BASE_DIR, "data", "small_sample.json")

""" Available params for load_dataset:
def load_dataset(path: str,
                 name: str | None = None,
                 data_dir: str | None = None,
                 data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
                 split: str | Split | None = None,
                 cache_dir: str | None = None,
                 features: Features | None = None,
                 download_config: DownloadConfig | None = None,
                 download_mode: DownloadMode | str | None = None,
                 verification_mode: VerificationMode | str | None = None,
                 ignore_verifications: Any = "deprecated",
                 keep_in_memory: bool | None = None,
                 save_infos: bool = False,
                 revision: str | Version | None = None,
                 token: bool | str | None = None,
                 use_auth_token: Any = "deprecated",
                 task: str = "deprecated",
                 streaming: bool = False,
                 num_proc: int | None = None,
                 storage_options: dict | None = None,
                 trust_remote_code: bool | None = None,
                 **config_kwargs: Any) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset
"""

def get_dataset(data_files=local_file, path="json", split="train"):
    dataset = load_dataset(
        path=path,
        data_files={"train": data_files},
        split=split
    )
    return dataset


dataset = load_dataset(
    path="json",
    data_files={"train": local_file},
    split="train"
)


def get_model_tokenizer(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    return model, tokenizer


def formatting_prompts_func(dataset, EOS_TOKEN):
    formatted_texts = []
    for example in dataset:
        text = alpaca_prompt.format(  # USING THE ALPACA PROMPT IN HERE DOESN'T SEEM CORRECT, IS IT?
            example['instruction'],
            example['input'],
            example['output']
        ) + EOS_TOKEN
        formatted_texts.append(text)
    return formatted_texts


def fine_tuning_orchestrator(model_name, dataset):
    model, tokenizer = get_model_tokenizer(model_name)
    EOS_TOKEN = tokenizer.eos_token

    formatted_texts = formatting_prompts_func(dataset, EOS_TOKEN)

    dataset = dataset.map(lambda example, idx: {
        'text': formatted_texts[idx]
    }, with_indices=True)

    model = set_model(model)
    trainer = set_trainer(model, tokenizer, dataset)
    trainer_stats = trainer.train()
    show_stats(trainer)

    return trainer_stats


def set_model(model):
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )
    return model


def set_trainer(model, tokenizer, dataset):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )
    return trainer


def show_stats(trainer):
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    trainer_stats = trainer.train()


if __name__ == "__main__":
    model_name = "unsloth/llama-3-8b-bnb-4bit"
    result = fine_tuning_orchestrator(model_name, dataset)

    print(result)
