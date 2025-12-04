import pandas as pd
import glob
import os

def consolidate_bank_patterns(directory="."):
    """
    Reads all CSVs matching '*_DRecurringPattern*.csv', extracts bank names,
    and consolidates patternId and patternCycle columns.
    """
    # 1. Find all matching files
    # We look for files ending in .csv that contain '_DRecurringPattern'
    search_path = os.path.join(directory, "*_DRecurringPattern*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"No files found matching pattern: {search_path}")
        return

    print(f"Found {len(csv_files)} files to process.\n")

    all_data = []

    for filepath in csv_files:
        try:
            # 2. Parse Bank Name from Filename
            # Example: "path/to/Chase_DRecurringPattern_v2.csv"
            filename = os.path.basename(filepath)
            
            # Logic: Split by '_DRecurringPattern' and take the first part
            # This handles cases like "Chase_DRecurringPattern.csv" -> "Chase"
            # And "Wells_Fargo_DRecurringPattern_2024.csv" -> "Wells_Fargo"
            if "_DRecurringPattern" in filename:
                bank_name = filename.split("_DRecurringPattern")[0]
            else:
                # Fallback if the file matches the glob but casing is different
                bank_name = filename.split("_")[0]

            print(f"Processing: {filename} -> Bank: {bank_name}")

            # 3. Read specific columns
            # using usecols is faster and uses less memory
            df = pd.read_csv(filepath, usecols=['patternId', 'patternCycle'])
            
            # Add the parsed bank name
            df['bank_name'] = bank_name
            
            all_data.append(df)

        except ValueError as ve:
            print(f"  [!] Skipped {filename}: Missing required columns (patternId or patternCycle).")
        except Exception as e:
            print(f"  [!] Error reading {filename}: {e}")

    # 4. Concatenate and Save
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        output_filename = f"{directory}/all_DRecurringPattern3.csv"

        initial_count = len(final_df)
        final_df = final_df.drop_duplicates(subset=['patternId'])
        dup_count = initial_count - len(final_df)

        final_df.to_csv(output_filename, index=False)

        print(f"\nSuccess! Consolidated {len(final_df)} rows to '{output_filename}'.")
        print(f"Removed {dup_count} duplicate transaction IDs.")
        print(final_df.head())

    else:
        print("\nNo valid data could be extracted.")

import sys
if __name__ == "__main__":
    consolidate_bank_patterns(sys.argv[1]) # Work/Data/selected_patterns
