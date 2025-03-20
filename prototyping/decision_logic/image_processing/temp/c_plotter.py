import numpy as np
import matplotlib.pyplot as plt

# File path to the logged horizon data
log_file = "/home/gijs/paparazzi/horizon_log.txt"

# Load the data (each row is one detection frame)
data = np.loadtxt(log_file, dtype=int)

# Check if multiple frames exist
if data.ndim == 1:
    data = data.reshape(1, -1)  # Ensure it's 2D even if one frame exists

# Select the latest frame
horizon = data[-1]

# Plot the detected horizon
plt.figure(figsize=(10, 5))
plt.plot(horizon, label="Horizon Line", color="red", linewidth=2)
plt.ylim([40, 0])  # Invert y-axis (0 at top)
plt.xlim([0, 160]) # Match the column count
plt.xlabel("Column Index")
plt.ylabel("Row (Height)")
plt.title("Horizon Detection Output")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
