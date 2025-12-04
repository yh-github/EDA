from pathlib import Path
import pandas as pd

all_patterns = pd.read_csv(Path('C:/Work/Data/selected_patterns/all_DRecurringPattern3.csv'), low_memory=False)
all_transactions = pd.read_csv(Path('C:/Work/Data/selected_transactions/all_transactions3.csv'), low_memory=False)

all_merged = pd.merge(
    left=all_transactions,
    right=all_patterns,
    left_on=['patternId', 'bank_name'],
    right_on=['patternId', 'bank_name'],
    how='left'
    # suffixes=('', '_pattern')
)

all_merged.to_csv(Path('C:/Work/Data/data.csv'), index=False)

print(all_merged.head())