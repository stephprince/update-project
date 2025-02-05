#DC created to check for units / region for BOTH regions and record as a parameter to be used by decoder_analyzer. allow comparing strength of encoding of location across regions

import os
import pynwb
import numpy as np
import csv
#not sure if i need a class. check

# Define the regions of interest
regions_of_interest = ['CA1', 'PFC']

# Initialize a dictionary to store the unit counts per region per file
unit_counts = {}

# Path to the folder with NWB files
nwb_directory = 'Y:\singer\NWBData\UpdateTask'

# Function to count units in each region
def count_units_in_region(units, regions_of_interest):
    region_counts = {region: 0 for region in regions_of_interest}
    
    # Assuming 'region' is a column in units table (check NWB schema)
    if 'region' in units.colnames:
        for region in regions_of_interest:
            region_counts[region] = np.sum(units['region'].data[:] == region)
    return region_counts

# Loop through all NWB files in the directory
for nwb_file in os.listdir(nwb_directory):
    if nwb_file.endswith('.nwb'):
        file_path = os.path.join(nwb_directory, nwb_file)
        
        # Open the NWB file
        with pynwb.NWBHDF5IO(file_path, 'r') as io:
            nwb_data = io.read()
            
            # Check if units are present
            if nwb_data.units:
                units = nwb_data.units
                # Count units for each region
                region_counts = count_units_in_region(units, regions_of_interest)
                # Store the counts
                unit_counts[nwb_file] = region_counts
                    
# write the results to a CSV file

with open('unit_counts_per_region.csv', 'w', newline='') as csvfile:
    fieldnames = ['File', 'CA1', 'mPFC']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for file, counts in unit_counts.items():
        row = {'File': file, 'CA1': counts['CA1'], 'mPFC': counts['mPFC']}
        writer.writerow(row)