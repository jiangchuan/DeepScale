import os
import argparse
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    get_scheduler,
    SchedulerType
)
from datasets import load_dataset
import deepspeed
from deepspeed.accelerator import get_accelerator

# --- Configuration ---
MODEL_NAME_OR_PATH = "mistralai/Mistral-7B-v0.1"
# Or define a config for an ~8B model (example using Llama-like structure)
# config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
# config.num_hidden_layers = 40 # Adjust these to reach ~8B
# config.hidden_size = 4096    # Adjust these
# config.intermediate_size = 11008 # Adjust these
# config.num_attention_heads = 32
# config.num_key_value_heads = 8 # For GQA if applicable

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"


# Training args (use argparse for flexibility)
def parse_args():
    parser = argparse.ArgumentParser(description="Train an LLM with DeepSpeed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by deepspeed")
    parser.add_argument("--model_name_or_path", type=str, default=MODEL_NAME_OR_PATH, help="Model identifier or path")
    parser.add_argument("--tokenizer_name", type=str, default=TOKENIZER_NAME, help="Tokenizer identifier or path")
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Dataset name (from datasets library or path)")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-103-raw-v1", help="Dataset config")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length for Flash Attention")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"], help="LR scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="Warmup steps for LR scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Output directory for checkpoints")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")

    # Add DeepSpeed configuration
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


# --- Dummy Dataset ---
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, max_seq_length=4096, vocab_size=32000):
        self.num_samples = num_samples
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random token IDs
        input_ids = torch.randint(0, self.vocab_size, (self.max_seq_length,), dtype=torch.long)
        # Create attention mask (all ones for this dummy data)
        attention_mask = torch.ones_like(input_ids)
        # Labels are usually shifted input_ids for Causal LM
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- Initialize DeepSpeed ---
    # DeepSpeed requires local_rank parameter for distributed setup
    deepspeed.init_distributed()
    args.local_rank = int(os.environ['LOCAL_RANK'])  # Get local rank from environment
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    world_size = torch.distributed.get_world_size()

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    # Set padding token if not present (needed for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Common practice

    # --- Load Model Configuration ---
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # Modify config if needed (e.g., for a custom 8B model)
    # config.vocab_size = tokenizer.vocab_size # Ensure vocab size matches tokenizer
    config.use_cache = False  # Important for training

    # --- Load Model with Flash Attention 2 ---
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,  # Use bfloat16 on A100s
        attn_implementation="flash_attention_2"  # Enable Flash Attention 2
    )
    print(f"Model loaded. Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Load Dataset ---
    print("Loading dataset...")
    # Example using HF datasets (adjust as needed)
    # raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    # # This usually involves mapping a tokenization function over the dataset
    # def tokenize_function(examples):
    #     # Ensure output fits max_seq_length, handle truncation/padding
    #     pass
    # tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, ...)
    # train_dataset = tokenized_datasets["train"]

    # Using dummy dataset for demonstration
    train_dataset = DummyDataset(max_seq_length=args.max_seq_length, vocab_size=tokenizer.vocab_size)
    print("Dataset loaded.")

    # --- Prepare DeepSpeed Engine ---
    # Effective batch size = per_device_train_batch_size * world_size * gradient_accumulation_steps
    effective_batch_size = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    print(f"Effective Batch Size: {effective_batch_size}")

    # DeepSpeed figures out optimizer/scheduler from config if 'auto', otherwise uses passed values
    # We pass None for optimizer and lr_scheduler, letting DeepSpeed handle them based on ds_config.json
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config_params=args.deepspeed_config,  # Path to ds_config.json provided via CLI
        model_parameters=[p for p in model.parameters() if p.requires_grad],  # Pass trainable parameters
        optimizer=None,  # Let DeepSpeed handle optimizer creation from config
        lr_scheduler=None  # Let DeepSpeed handle scheduler creation from config
    )
    print(f"DeepSpeed Engine Initialized (Rank {args.local_rank})")

    # --- Dataloader ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,  # Micro-batch size
        # Use DistributedSampler if needed, but DeepSpeed DataLoader often handles it
        # sampler=torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=args.local_rank),
        pin_memory=True
    )

    # Calculate total steps if needed (DeepSpeed might do this based on 'auto' config)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # --- Training Loop ---
    global_step = 0
    print("Starting Training...")
    for epoch in range(args.num_train_epochs):
        model_engine.train()
        for step, batch in enumerate(train_dataloader):
            # Move batch to GPU (DeepSpeed handles model placement)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss

            # Backward pass (DeepSpeed handles accumulation and scaling)
            model_engine.backward(loss)

            # Optimizer step (DeepSpeed handles gradient clipping, etc.)
            model_engine.step()

            global_step += 1

            # Logging (only on rank 0)
            if args.local_rank == 0 and global_step % args.logging_steps == 0:
                print(
                    f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

            # Checkpointing (DeepSpeed handles saving across ranks)
            if global_step % args.save_steps == 0:
                print(f"Saving checkpoint at step {global_step}...")
                save_path = os.path.join(args.output_dir, f"step_{global_step}")
                # model_engine.save_checkpoint saves model, optimizer, scheduler states
                # tag argument is useful to identify checkpoints
                model_engine.save_checkpoint(save_path, tag=f"step_{global_step}")
                print(f"Checkpoint saved to {save_path} (Rank {args.local_rank})")

            # Optional: Add evaluation loop here

        print(f"Epoch {epoch} finished.")

    print("Training finished.")
    # Final save
    if args.output_dir is not None:
        print("Saving final model...")
        final_save_path = os.path.join(args.output_dir, "final_model")
        model_engine.save_checkpoint(final_save_path, tag="final")
        print(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()
