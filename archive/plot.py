import csv

import matplotlib.pyplot as plt


csv_path = "/data3/junhaohu/checkpoints/Comb-Qwen3-1B/training_diagnostics.csv"
output_path = "/data3/junhaohu/checkpoints/Comb-Qwen3-1B/training_loss_plot.png"

steps = []
losses = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["global_step"]))
        losses.append(float(row["loss"]))

fontsize = 20
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label="Training Loss", linewidth=1.8)
plt.xlabel("Training Steps", fontsize=fontsize)
plt.ylabel("Loss", fontsize=fontsize)
plt.title("Comb-Qwen3-1B Training Loss", fontsize=fontsize)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
