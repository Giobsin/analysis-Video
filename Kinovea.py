import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os


DATA_FILE = r"c:\Users\TRIOS\OneDrive - KTH\Thesis - Giorgio Coraglia\CSV trajectory\kinovea_exportBardonecchia2.csv"
df = pd.read_csv(DATA_FILE, sep=';', quotechar='"')


df = df.rename(columns={
    "Trajectory 1/0/X": "X",
    "Trajectory 1/0/Y": "Y"
})


for col in ["X", "Y", "Time"]:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)


real_length_cm = float(input("Enter the real ski length in centimeters: "))
real_length_m = real_length_cm / 100  # Convert cm to meters
pixel_length = float(input("Enter the measured ski length in pixels (from Kinovea): "))
scaling_factor = real_length_m / pixel_length  # meters per pixel


df["X"] *= scaling_factor
df["Y"] *= scaling_factor

# === Apply Savitzky-Golay filter ===
window_length = 11
if window_length >= len(df):
    window_length = len(df) - (1 - len(df) % 2)
df["Xs"] = savgol_filter(df["X"], window_length=window_length, polyorder=2)
df["Ys"] = savgol_filter(df["Y"], window_length=window_length, polyorder=2)


df["dX"] = df["Xs"].diff()
df["dY"] = df["Ys"].diff()
df["dt"] = df["Time"].diff()


df["v_tot"] = np.sqrt(df["dX"]**2 + df["dY"]**2) / df["dt"]
df["v_perp"] = df["dY"] / df["dt"]


slope_degrees = float(input("Enter the slope angle in degrees: "))
slope_radians = np.radians(slope_degrees)


df["v_parallel"] = df["v_tot"] * np.cos(slope_radians)
df["v_perpendicular_slope"] = df["v_perp"] * np.sin(slope_radians)
df["v_total_with_slope"] = np.sqrt(df["v_parallel"]**2 + df["v_perpendicular_slope"]**2)

df["v_horizontal_slope"] = df["v_tot"] * np.cos(slope_radians)
df["v_vertical_slope"] = df["v_perp"] * np.sin(slope_radians)


print(df[["Time", "v_tot", "v_perp", "v_total_with_slope", "v_horizontal_slope", "v_vertical_slope"]].dropna().head())

output_directory = os.path.dirname(DATA_FILE)
output_file = os.path.join(output_directory, "output_velocities_with_slope.xlsx")

with pd.ExcelWriter(output_file) as writer:
    df.to_excel(writer, sheet_name='Original Data', index=False)
    df[["Time", "v_tot", "v_perp", "v_total_with_slope", "v_horizontal_slope", "v_vertical_slope"]].dropna().to_excel(writer, sheet_name='Processed Data', index=False)

print(f"Results saved to {output_file}")
