import matplotlib.pyplot as plt

# Valorile reale preluate din log-ul tău
train_losses = [
    4.6500, 2.6585, 1.6160, 1.0675, 0.7494, 0.5907, 0.4896, 0.3955, 0.3608, 0.3203,
    0.2999, 0.2545, 0.2624, 0.2350, 0.2279, 0.2346, 0.2068, 0.1840, 0.2048, 0.1877
]

val_losses = [
    3.3356, 2.1230, 1.4713, 1.1606, 0.8727, 0.8185, 0.8627, 0.7494, 0.7261, 0.6630,
    0.6811, 0.6656, 0.6567, 0.6249, 0.6398, 0.6177, 0.5995, 0.6065, 0.6070, 0.6272
]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, 21), val_losses, marker='x', label='Validation Loss')

plt.title("Evoluția Loss-ului (ResNet-50, 20 epoci)")
plt.xlabel("Epocă")
plt.ylabel("Loss")
plt.xticks(range(1, 21))
plt.grid(True)
plt.legend()

# Salvare imagine
plt.savefig("resnet50_loss_plot.png")
plt.close()

print("✅ Plot salvat: resnet50_loss_plot.png")
