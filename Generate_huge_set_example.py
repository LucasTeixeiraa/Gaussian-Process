import os
import numpy as np
import pandas as pd
from modelBASE import TTCellModel as modelA
from ModelB import TTCellModelExt as modelB
from ModelC import TTCellModelChannel as modelC
import chaospy as cp

# Configuration
ti = 0
tf = 500
dt = 0.01
dtS = 1
sample_size = 5000  # Increase sample size

# Function to extract QOIs from model output
def extract_qois(results):
    return [{key: value for key, value in result.items() if key != 'Wf'} for result in results]

# Initialize models
modelA.setSizeParameters(ti, tf, dt, dtS)
modelB.setSizeParameters(ti, tf, dt, dtS)
modelC.setSizeParameters(ti, tf, dt, dtS)

# Define distribution
low, high = 0, 1

# Sample parameters
dist = modelA.getDist(low=low, high=high)
samplesA = dist.sample(sample_size, rule="latin_hypercube")

dist = modelB.getDist(low=low, high=high)
samplesB = dist.sample(sample_size, rule="latin_hypercube")

dist = modelC.getDist(low=low, high=high)
samplesC = dist.sample(sample_size, rule="latin_hypercube")

# Generate QOIs for each model
results_A = modelA.run(samplesA.T, use_gpu=True, regen=True, name="tS.txt")
results_B = modelB.run(samplesB.T, use_gpu=True, regen=True, name="tS.txt")
results_C = modelC.run(samplesC.T, use_gpu=True, regen=True, name="tS.txt")

# Extract QOIs
qois_A = extract_qois(results_A)
qois_B = extract_qois(results_B)
qois_C = extract_qois(results_C)

# Create directory for saving results
output_dir = "Generated_Data_5K"
os.makedirs(output_dir, exist_ok=True)

# Save data for Model A
modelA_dir = os.path.join(output_dir, "ModelA")
os.makedirs(modelA_dir, exist_ok=True)
pd.DataFrame(samplesA.T, columns=[f"Input_{i+1}" for i in range(samplesA.shape[0])]).to_csv(
    os.path.join(modelA_dir, "X.csv"), index=False)
pd.DataFrame(qois_A).to_csv(os.path.join(modelA_dir, "Y.csv"), index=False)

# Save data for Model B
modelB_dir = os.path.join(output_dir, "ModelB")
os.makedirs(modelB_dir, exist_ok=True)
pd.DataFrame(samplesB.T, columns=[f"Input_{i+1}" for i in range(samplesB.shape[0])]).to_csv(
    os.path.join(modelB_dir, "X.csv"), index=False)
pd.DataFrame(qois_B).to_csv(os.path.join(modelB_dir, "Y.csv"), index=False)

# Save data for Model C
modelC_dir = os.path.join(output_dir, "ModelC")
os.makedirs(modelC_dir, exist_ok=True)
pd.DataFrame(samplesC.T, columns=[f"Input_{i+1}" for i in range(samplesC.shape[0])]).to_csv(
    os.path.join(modelC_dir, "X.csv"), index=False)
pd.DataFrame(qois_C).to_csv(os.path.join(modelC_dir, "Y.csv"), index=False)

print(f"Data saved to directory: {output_dir}")
