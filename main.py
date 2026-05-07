import torch
from torch.optim import lr_scheduler
import os
import time
import json
from tqdm import tqdm

from vpr_model import VPRModel
from dataloaders.OpenHotelsDataloader import OpenHotelsDataModule


def get_optimizer_and_scheduler(model, config):
    optimizer_name = config.get("optimizer", "sgd")
    lr = config.get("lr", 0.03)
    weight_decay = config.get("weight_decay", 1e-3)
    momentum = config.get("momentum", 0.9)

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

    lr_sched_name = config.get("lr_sched", "linear")
    lr_sched_args = config.get("lr_sched_args", {})

    if lr_sched_name.lower() == "multistep":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_sched_args["milestones"],
            gamma=lr_sched_args["gamma"],
        )
    elif lr_sched_name.lower() == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, lr_sched_args["T_max"])
    elif lr_sched_name.lower() == "linear":
        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=lr_sched_args.get("start_factor", 1.0),
            end_factor=lr_sched_args.get("end_factor", 0.1),
            total_iters=lr_sched_args.get("total_iters", 100),
        )
    else:
        scheduler = None  # Or raise error

    return optimizer, scheduler


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VPR model.")
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=None,
        help="Path to a JSON file to overwrite the default model_config",
    )
    args = parser.parse_args()

    # Configuration
    batch_size = 32
    img_per_place = 4
    num_workers = 32
    max_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataModule
    datamodule = OpenHotelsDataModule(
        batch_size=batch_size,
        img_per_place=img_per_place,
        min_img_per_place=1,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=num_workers,
    )

    # Setup Dataloader
    print("Setting up dataloader...")
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    print(f"Dataloader ready. Steps per epoch: {len(train_loader)}")

    # Model Configuration
    model_config = {
        "backbone_arch": "dinov2_vitb14",
        "backbone_config": {
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True
        },
        "agg_arch": "salad",
        "agg_config": {
            "num_channels": 768,
            "num_clusters": 64,
            "cluster_dim": 128,
            "token_dim": 256
        },
        "lr": 6e-05,
        "optimizer": "adamw",
        "weight_decay": 9.5e-09,
        "momentum": 0.9,
        "lr_sched": "linear",
        "lr_sched_args": {
            "start_factor": 1,
            "end_factor": 0.2,
            "total_iters": 4000
        },
        "loss_name": "MultiSimilarityLoss",
        "miner_name": None,
        "miner_margin": 0.1,
        "faiss_gpu": False,
    }

    if args.model_config_path:
        with open(args.model_config_path, "r") as f:
            override_config = json.load(f)
        model_config.update(override_config)
        print(f"📄 Updated model_config with {args.model_config_path}")

    # Model
    print("Initializing model...")
    model = VPRModel(**model_config)
    model.to(device)

    # Optimizer and Scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, model_config)

    # Output directory
    output_dir = "logs/openhotels/dinov2_vitb14_salad"
    os.makedirs(output_dir, exist_ok=True)

    # Save model config
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=4)

    # Training Loop
    print("Starting training...")
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        # Wrap train_loader with tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}", unit="batch")
        for batch_idx, (places, labels) in enumerate(pbar):
            # Move to device
            places = places.to(device)
            labels = labels.to(device)

            # Reshape places: [BS, K, C, H, W] -> [BS*K, C, H, W]
            BS, K, C, H, W = places.shape
            images = places.view(BS * K, C, H, W)
            labels = labels.view(-1)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            descriptors = model(images)

            # Check for NaNs
            desc_for_check = (
                descriptors[0] if isinstance(descriptors, tuple) else descriptors
            )
            if torch.isnan(desc_for_check).any():
                print("NaNs in descriptors! Stopping.")
                break

            # Loss
            loss = model.loss_function(descriptors, labels)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()
            if scheduler:
                scheduler.step()

            # Logging
            running_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{max_epochs}] Finished. Avg Loss: {epoch_loss:.4f}. Time: {time.time() - start_time:.2f}s"
        )

        # Save Checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete.")
