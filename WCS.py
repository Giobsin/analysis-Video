import pandas as pd

# Correct file path and sheet name
DATA_FILE = r"C:\Users\TRIOS\OneDrive - KTH\Skrivbordet\Collection_data1.xlsx"
DATA_SHEET = "WCS"

# Load the correct sheet
data = pd.read_excel(DATA_FILE, sheet_name=DATA_SHEET)

# Calculate percentage errors for each WCS compared to VI
data['Error_VL2'] = (data['VL2'] - data['VI']) / data['VI'] * 100
data['Error_VI+10'] = (data['V L1 plus10%'] - data['VI']) / data['VI'] * 100
data['Error_VI-10'] = (data['V L1 minus10%'] - data['VI']) / data['VI'] * 100
data['Error_VL2+10'] = (data['V L2 plus10%'] - data['VI']) / data['VI'] * 100
data['Error_VL2-10'] = (data['V L2 minus10%'] - data['VI']) / data['VI'] * 100

# Display the updated table
print(data)

summary = data[['Error_VL2','Error_VI+10','Error_VI-10','Error_VL2+10','Error_VL2-10']].agg(['mean','std','min','max'])
print(summary)

import matplotlib.pyplot as plt

data[['Error_VL2','Error_VI+10','Error_VI-10','Error_VL2+10','Error_VL2-10']].boxplot()
plt.ylabel("Percentage Error (%)")
plt.title("Worst Case Scenario Error Distribution")
plt.show()

# Optionally, save to a new Excel file
output_path = r"C:\Users\TRIOS\OneDrive - KTH\Skrivbordet\WCS_analysis_results.xlsx"
data.to_excel(output_path, index=False)

print(f"\nResults saved to {output_path}")
