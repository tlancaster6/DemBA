#!/usr/bin/env python3
"""
Script to remove empty individual3 columns from DeepLabCut annotation files.
Processes both CSV and H5 files in all subdirectories of labeled-data folder.
"""

import os
import pandas as pd
import shutil
from pathlib import Path

def check_individual3_data(df, file_type):
    """Check if individual3 columns contain any data."""
    individual3_mask = df.columns.get_level_values(1) == 'individual3'
    individual3_data = df.loc[:, individual3_mask]
    has_data = individual3_data.notna().any().any()
    
    if has_data:
        print(f"  WARNING: Found data in individual3 columns in {file_type} file!")
        non_empty_rows = individual3_data.dropna(how='all', axis=0)
        non_empty_cols = individual3_data.dropna(how='all', axis=1)
        print(f"  Non-empty rows: {len(non_empty_rows)}")
        print(f"  Non-empty columns: {len(non_empty_cols.columns)}")
        return True
    return False

def remove_individual3_columns(df):
    """Remove all columns belonging to individual3."""
    # Create mask for columns that are NOT individual3
    keep_mask = df.columns.get_level_values(1) != 'individual3'
    return df.loc[:, keep_mask]

def process_csv_file(csv_path):
    """Process a CSV file to remove individual3 columns."""
    print(f"  Processing CSV: {csv_path}")
    
    # Read CSV as raw text to preserve header formatting
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header to find individual3 columns
    if len(lines) < 4:
        print(f"    ERROR: CSV has fewer than 4 header rows")
        return False
    
    # Split header rows
    header_rows = [line.strip().split(',') for line in lines[:4]]
    
    # Find individual3 columns (check row 1 - individuals row)
    individual3_indices = []
    for i, value in enumerate(header_rows[1]):  # individuals row
        if value == 'individual3':
            individual3_indices.append(i)
    
    print(f"    Original columns: {len(header_rows[0])}")
    print(f"    Found {len(individual3_indices)} individual3 columns to remove")
    
    # Check for data in individual3 columns using pandas
    df = pd.read_csv(csv_path, header=[0,1,2,3])
    has_data = check_individual3_data(df, "CSV")
    if has_data:
        return False
    
    # Create backup
    backup_path = csv_path.with_suffix('.csv.backup')
    if not backup_path.exists():  # Only create backup if it doesn't exist
        shutil.copy2(csv_path, backup_path)
        print(f"    Created backup: {backup_path}")
    
    # Remove individual3 columns from all rows
    cleaned_lines = []
    for line in lines:
        row = line.strip().split(',')
        # Keep columns that are NOT in individual3_indices
        cleaned_row = [row[i] for i in range(len(row)) if i not in individual3_indices]
        cleaned_lines.append(','.join(cleaned_row) + '\n')
    
    print(f"    New columns: {len(cleaned_lines[0].strip().split(','))}")
    
    # Write cleaned CSV
    with open(csv_path, 'w') as f:
        f.writelines(cleaned_lines)
    
    print(f"    Saved cleaned CSV")
    
    return True

def process_h5_file(h5_path):
    """Process an H5 file to remove individual3 columns."""
    print(f"  Processing H5: {h5_path}")
    
    # Load H5 file
    df = pd.read_hdf(h5_path)
    
    print(f"    Original shape: {df.shape}")
    
    # Check for data in individual3 columns
    has_data = check_individual3_data(df, "H5")
    if has_data:
        return False
    
    # Remove individual3 columns
    df_cleaned = remove_individual3_columns(df)
    print(f"    New shape: {df_cleaned.shape}")
    
    # Create backup
    backup_path = h5_path.with_suffix('.h5.backup')
    shutil.copy2(h5_path, backup_path)
    print(f"    Created backup: {backup_path}")
    
    # Save cleaned H5
    df_cleaned.to_hdf(h5_path, key='keypoints', mode='w')
    print(f"    Saved cleaned H5")
    
    return True

def main():
    """Main function to process all folders in labeled-data directory."""
    labeled_data_path = Path("../../projects/demasoni_singlenuc-tucker-2025-09-11/labeled-data")
    
    if not labeled_data_path.exists():
        print(f"ERROR: Path {labeled_data_path} does not exist!")
        return
    
    # Get all subdirectories
    subdirs = [d for d in labeled_data_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} subdirectories to process")
    
    success_count = 0
    warning_count = 0
    
    for subdir in subdirs:
        print(f"\nProcessing directory: {subdir.name}")
        
        # Look for CSV and H5 files
        csv_files = list(subdir.glob("*.csv"))
        h5_files = list(subdir.glob("*.h5"))
        
        if not csv_files and not h5_files:
            print(f"  No CSV or H5 files found in {subdir.name}")
            continue
        
        folder_success = True
        
        # Process CSV files
        for csv_file in csv_files:
            if not process_csv_file(csv_file):
                folder_success = False
                warning_count += 1
        
        # Process H5 files
        for h5_file in h5_files:
            if not process_h5_file(h5_file):
                folder_success = False
                warning_count += 1
        
        if folder_success:
            success_count += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Directories processed successfully: {success_count}")
    print(f"Directories with warnings (data found in individual3): {warning_count}")
    print(f"Total directories: {len(subdirs)}")
    
    if warning_count > 0:
        print(f"\nWARNING: {warning_count} directories had data in individual3 columns!")
        print("These files were NOT modified. Please review manually.")

if __name__ == "__main__":
    main()