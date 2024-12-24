import argparse
import os
from pathlib import Path
from typing import Any, List, Literal, Optional

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import TextStreamer, TrainingArguments
from trl import SFTTrainer, DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth import PatchDPOTrainer

# This template doesn’t require additional tokens, which makes it less error-prone (but can slightly impact performance compared to ChatML).
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""


def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    chat_template: str,
) -> tuple:
    # Load the model to fine-tune and its corresponding tokenizer.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # Define LoRA/QLoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )

    return model, tokenizer


def finetune(
    finetuning_type: Literal["sft", "dpo"],
    model_name: str,
    output_dir: str,
    dataset_huggingface_workspace: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "o_proj",
        "gate_proj",
    ],
    chat_template: str = "chatml",
    learning_rate: float = 3e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    beta: float = 0.5,  # Only for DPO
    is_dummy: bool = False,
) -> tuple:
    model, tokenizer = load_model(
        model_name,
        max_seq_length,
        load_in_4bit,
        lora_rank,
        lora_alpha,
        lora_dropout,
        target_modules,
        chat_template,
    )
    # Manually add the end-of-sequence token at the end of each message
    # to ensure that the model learns to output it.
    # Without it, it will keep generating answers without ever stopping.
    EOS_TOKEN = tokenizer.eos_token
    print(f"Setting EOS_TOKEN to {EOS_TOKEN}")

    if is_dummy is True:
        num_train_epochs = 1
        print(
            f"Training in dummy mode. Setting num_train_epochs to '{num_train_epochs}'"
        )
        print(f"Training in dummy mode. Reducing dataset size to '400'.")

    if finetuning_type == "sft":

        def format_samples_sft(examples):
            text = []
            for instruction, output in zip(
                examples["instruction"], examples["output"], strict=False
            ):
                message = alpaca_template.format(instruction, output) + EOS_TOKEN
                text.append(message)

            return {"text": text}

        # We don't have lots of samples in the llm_twin dataset (648 sampels)
        # This is an issue because the model might not correctly learn the chat template
        # Hence, we will upsample it with a high-quality general-purpose dataset called FindTome (only take 10k in the train split)
        # Then, concatenate these 2 datasets to create final set.
        dataset1 = load_dataset(
            f"{dataset_huggingface_workspace}/llm_twin", split="train"
        )
        dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
        dataset = concatenate_datasets([dataset1, dataset2])
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")
        print(f"Loaded dataset with {len(dataset)} samples.")

        # We need to format the data using a chat template
        # Map all the instructions and answers to the chat template
        dataset = dataset.map(
            format_samples_sft, batched=True, remove_columns=dataset.column_names
        )

        # Divide the dataset into train (95%) and test(5%)
        dataset = dataset.train_test_split(test_size=0.05)

        print("Training dataset example:")
        print(dataset["train"][0])

        print("Training dataset example:")
        print(dataset["train"][0])

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=True,
            args=TrainingArguments(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                report_to="comet_ml",
                seed=0,
            ),
        )
    elif finetuning_type == "dpo":
        # Declare here to fix the DPO logs in notebook environment
        # PatchDPOTrainer()

        def format_samples_dpo(example):
            example["prompt"] = alpaca_template.format(example["prompt"], "")
            example["chosen"] = example["chosen"] + EOS_TOKEN
            example["rejected"] = example["rejected"] + EOS_TOKEN

            return {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }

        dataset = load_dataset(
            f"{dataset_huggingface_workspace}/llm_twin_dpo", split="train"
        )
        if is_dummy:
            try:
                dataset = dataset.select(range(400))
            except Exception:
                print("Dummy mode active. Failed to trim the dataset to 400 samples.")
        print(f"Loaded dataset with {len(dataset)} samples.")

        dataset = dataset.map(format_samples_dpo)
        dataset = dataset.train_test_split(test_size=0.05)

        print("Training dataset example:")
        print(dataset["train"][0])

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            beta=beta,  # default beta=0.1
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=max_seq_length // 2,  # prompt + answer
            max_prompt_length=max_seq_length // 2,  # prompt only
            args=DPOConfig(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                per_device_eval_batch_size=per_device_train_batch_size,
                warmup_steps=10,
                output_dir=output_dir,
                eval_steps=0.2,
                logging_steps=1,
                report_to="comet_ml",
                seed=0,
            ),
        )
    else:
        raise ValueError("Invalid finetuning_type. Choose 'sft' or 'dpo'.")

    trainer.train()

    return model, tokenizer


# Test inference to make that there are no obvious errors related to the tokenizer or chat template
def inference(
    model: Any,
    tokenizer: Any,
    prompt: str = "Write a paragraph to introduce supervised fine-tuning.",
    max_new_tokens: int = 256,
) -> None:
    model = FastLanguageModel.for_inference(model)
    message = alpaca_template.format(prompt, "")
    inputs = tokenizer([message], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True
    )


def save_model(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
):
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

    if push_to_hub and repo_id:
        print(f"Saving model to '{repo_id}'")
        model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")


def check_if_huggingface_model_exists(
    model_id: str, default_value: str = "michaelnguyen11/TwinLlama-3.2-3B"
) -> str:
    api = HfApi()

    try:
        api.model_info(model_id)
    except RepositoryNotFoundError:
        print(f"Model '{model_id}' does not exist.")
        model_id = default_value
        print(f"Defaulting to '{model_id}'")
        print("Train your own 'TwinLlama-3.1-8B' to avoid this behavior.")

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--dataset_huggingface_workspace", type=str, default="michaelnguyen11"
    )
    parser.add_argument(
        "--model_output_huggingface_workspace", type=str, default="michaelnguyen11"
    )
    parser.add_argument(
        "--is_dummy",
        type=bool,
        default=False,
        help="Flag to reduce the dataset size for testing",
    )
    parser.add_argument(
        "--finetuning_type",
        type=str,
        choices=["sft", "dpo"],
        default="sft",
        help="Parameter to choose the finetuning stage [sft or dpo].",
    )

    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    print(f"Num training epochs: '{args.num_train_epochs}'")
    print(f"Per device train batch size: '{args.per_device_train_batch_size}'")
    print(f"Learning rate: {args.learning_rate}")
    print(
        f"Datasets will be loaded from Hugging Face workspace: '{args.dataset_huggingface_workspace}'"
    )
    print(
        f"Models will be saved to Hugging Face workspace: '{args.model_output_huggingface_workspace}'"
    )
    print(f"Training in dummy mode? '{args.is_dummy}'")
    print(f"Finetuning type: '{args.finetuning_type}'")

    print(f"Output data dir: '{args.output_data_dir}'")
    print(f"Model dir: '{args.model_dir}'")
    print(f"Number of GPUs: '{args.n_gpus}'")

    if args.finetuning_type == "sft":
        print("Starting SFT training...")
        base_model_name = "unsloth/Llama-3.2-3B"
        print(f"Training from base model '{base_model_name}'")

        output_dir_sft = Path(args.model_dir) / "output_sft"
        model, tokenizer = finetune(
            finetuning_type="sft",
            model_name=base_model_name,
            output_dir=str(output_dir_sft),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            learning_rate=args.learning_rate,
            is_dummy=args.is_dummy,
        )
        inference(model, tokenizer)

        sft_output_model_repo_id = (
            f"{args.model_output_huggingface_workspace}/TwinLlama-3.2-3B"
        )
        save_model(
            model,
            tokenizer,
            "model_sft",
            push_to_hub=True,
            repo_id=sft_output_model_repo_id,
        )
    elif args.finetuning_type == "dpo":
        print("Starting DPO training...")
        sft_base_model_repo_id = (
            f"{args.model_output_huggingface_workspace}/TwinLlama-3.2-3B"
        )
        sft_base_model_repo_id = check_if_huggingface_model_exists(
            sft_base_model_repo_id
        )
        print(f"Training from base model '{sft_base_model_repo_id}'")  # noqa

        output_dir_dpo = Path(args.model_dir) / "output_dpo"
        model, tokenizer = finetune(
            finetuning_type="dpo",
            model_name=sft_base_model_repo_id,
            output_dir=str(output_dir_dpo),
            dataset_huggingface_workspace=args.dataset_huggingface_workspace,
            num_train_epochs=1,  # train for 1 epoch instead of 3 in SFT
            per_device_train_batch_size=args.per_device_train_batch_size,
            lora_rank=32,  # increase from 32 in SFT to 64 in DPO
            lora_alpha=32,  # increase from 32 in SFT to 64 in DPO
            lora_dropout=0.0,
            learning_rate=2e-6,  # learning rate is also lower (from 3e-4 for SFT to 2e-6 here)
            is_dummy=args.is_dummy,
        )
        inference(model, tokenizer)

        dpo_output_model_repo_id = (
            f"{args.model_output_huggingface_workspace}/TwinLlama-3.2-3B-DPO"
        )
        save_model(
            model,
            tokenizer,
            "model_dpo",
            push_to_hub=True,
            repo_id=dpo_output_model_repo_id,
        )
    else:
        raise ValueError("Invalid finetuning_type. Choose 'sft' or 'dpo'.")
