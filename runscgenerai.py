import pandas as pd
import numpy as np
import scGeneRAI
import os

# Load example data
ex_data_path = '../scGeneRAI/example_data/example_data.csv'
ex_data = pd.read_csv(ex_data_path).iloc[:, 4:]
ex_data_descriptors = pd.read_csv(ex_data_path).iloc[:, 1:3]

# Normalize the data
means = ex_data.mean(axis=0)
sds = ex_data.std(axis=0)
ex_data = (ex_data - means) / sds

# Initialize and fit the model
model = scGeneRAI.scGeneRAI()
model.fit(ex_data, nepochs=100, model_depth=2, descriptors=ex_data_descriptors, early_stopping=True, device_name='cpu')

# Predict networks
model.predict_networks(ex_data.iloc[:50, :], descriptors=ex_data_descriptors.iloc[:50, :], PATH='.')

# Read and combine the result files
results_dir = './results'
files = os.listdir(results_dir)
network_data = pd.concat([pd.read_csv(os.path.join(results_dir, file)) for file in files])

# Process network data
network_data['LRP'] = np.abs(network_data['LRP'])
network_data = network_data[network_data['source_gene'] != network_data['target_gene']]
average_network = network_data[['LRP', 'source_gene', 'target_gene']].groupby(['source_gene', 'target_gene']).mean().reset_index()

# Sort and save the edges
edges = average_network.sort_values(by='LRP', ascending=False)
edges.to_csv('sorted_edges.csv', index=False)