import pandas as pd
import glob
import os

def consolidate_transactions(directory="."):
    """
    Reads all CSVs matching '*_DTransaction*.csv', filters for valid/cleared transactions,
    renames columns, and consolidates them into a single file.
    """
    # 1. Find all matching files
    search_path = os.path.join(directory, "*_DTransaction*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"No files found matching pattern: {search_path}")
        return

    print(f"Found {len(csv_files)} files to process.\n")

    all_data = []

    # Columns that MUST exist and MUST NOT be null
    required_cols = [
        'trId', 'accountId', 'date', 'amount', 'bankRawDescription', 'isRecurring'
    ]
    
    # Columns that we want to keep if they exist (can be null)
    optional_cols = ['counter_party', 'patternId', 'p_categoryGroupId', 'p_subCategoryId']

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        try:
            # 2. Parse Bank Name
            if "_DTransaction" in filename:
                bank_name = filename.split("_DTransaction")[0]
            else:
                bank_name = filename.split("_")[0]

            print(f"Processing: {filename} -> Bank: {bank_name}")

            # Read the file
            # We read all columns first to safely handle filtering and renaming
            df = pd.read_csv(filepath, low_memory=False)
            
            # 3. Filter Data
            # mask = (isForeignCurrency == False) & (status == 'Cleared')
            # Check if columns exist to avoid errors
            if 'isForeignCurrency' in df.columns and 'status' in df.columns:
                # Handle potential string vs bool differences
                is_foreign = df['isForeignCurrency'].astype(str).str.lower() == 'true'
                is_cleared = df['status'] == 'Cleared'
                
                df = df[(~is_foreign) & (is_cleared)].copy()
            else:
                print(f"  [!] Skipped filtering for {filename}: Missing 'isForeignCurrency' or 'status' columns.")


            def violations(dt):
                return len(dt[
                    ((dt['amount'] > 0) & (dt['direction'] != 'D')) |
                    ((dt['amount'] < 0) & (dt['direction'] != 'C'))
                ])

            assert len(df[df['amount']==0])==0
            assert violations(df)==0


            # 4. Rename Columns
            # 'id' -> 'trId', 'deviceId' -> 'counter_party'
            rename_map = {
                'id': 'trId',
                'deviceId': 'counter_party',
                'personeticsCategoryGroupId': 'p_categoryGroupId',
                'personeticsSubCategoryId': 'p_subCategoryId'
            }
            df = df.rename(columns=rename_map)
            
            # 5. Add Bank Name
            df['bank_name'] = bank_name

            # 6. Validate & Select Columns
            # Ensure required columns exist in the DataFrame (renaming happened above)
            missing_reqs = [c for c in required_cols if c not in df.columns]
            if missing_reqs:
                print(f"  [!] Skipped {filename}: Missing required columns after rename: {missing_reqs}")
                continue
            
            # Select only the columns we want (union of required + optional + bank_name)
            cols_to_keep = required_cols + [c for c in optional_cols if c in df.columns] + ['bank_name']
            df = df[cols_to_keep]

            # 7. Drop Rows with Missing Values in Required Columns
            # "filter rows that have missing values for any of the other columns"
            before_drop = len(df)
            df = df.dropna(subset=required_cols)
            dropped_count = before_drop - len(df)
            
            if dropped_count > 0:
                print(f"      Dropped {dropped_count} rows due to missing required values.")

            if not df.empty:
                all_data.append(df)

        except Exception as e:
            print(f"  [!] Error processing {filename}: {e}")

    # 8. Concatenate, Deduplicate, and Save
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 9. Drop Duplicates by trId
        initial_count = len(final_df)
        final_df = final_df.drop_duplicates(subset=['trId'])
        dup_count = initial_count - len(final_df)
        
        output_filename = f"{directory}/all_transactions3.csv"
        final_df.to_csv(output_filename, index=False)
        
        print(f"\nSuccess! Consolidated {len(final_df)} rows to '{output_filename}'.")
        print(f"Removed {dup_count} duplicate transaction IDs.")
        print(final_df.head())
    else:
        print("\nNo valid transactions found.")

import sys
if __name__ == "__main__":
    consolidate_transactions(sys.argv[1]) # Work/Data/selected_transactions
