import numpy as np
import matplotlib.pyplot as plt

data = np.load("results/salience_sweep/summary.npz", allow_pickle=True)
summary = data["summary"]  # shape [n, 2] columns: salience, mean_FE
salience = summary[:, 0]
mean_fe = summary[:, 1]

plt.figure()
plt.plot(salience, mean_fe, marker="o")
plt.xlabel("salience")
plt.ylabel("mean FE")
plt.title("Effect of monument salience on free energy")
plt.tight_layout()
