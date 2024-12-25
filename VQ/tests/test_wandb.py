import random
import wandb
# This is secret and shouldn't be checked into version control
WANDB_API_KEY="532007cd7a07c1aa0d1194049c3231dadd1d418e"
# Name and notes optional
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
wandb.login(key=WANDB_API_KEY)

# Launch 5 simulated experiments
total_runs = 1
for run in range(total_runs):
    # 🐝 1️⃣ Start a new run to track this script
    wandb.init(
        # Set the project where this run will be logged
        project="test2",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{run}",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        })

    # This simple block simulates a training loop logging metrics
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # 🐝 2️⃣ Log metrics from your script to W&B
        wandb.log({"acc": acc, "loss": loss})

    # Mark the run as finished
    wandb.finish()

# Retrieve the history of the 'loss' metric from wandb
print("aaa")
