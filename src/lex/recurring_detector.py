import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from common.config import FieldConfig


class RecurringDetector:
    def __init__(self, field_config: FieldConfig = None,
                 interval_tolerance=10, min_transactions=2,
                 amount_cv_threshold=0.2, dom_std_threshold=1.5, eps=0.29):
        self.fc = field_config or FieldConfig()
        self.interval_tolerance = interval_tolerance
        self.min_transactions = min_transactions
        self.amount_cv_threshold = amount_cv_threshold
        self.dom_std_threshold = dom_std_threshold
        self.eps = eps

        self.STOPWORDS = {
            'pos', 'debit', 'credit', 'purchase', 'payment', 'transfer',
            'check', 'withdrawal', 'deposit', 'atm', 'fee', 'charge',
            'bill', 'pay', 'online', 'mobile', 'transaction', 'authorized'
        }

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Remove digits and special chars
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        words = text.split()
        words = [w for w in words if w not in self.STOPWORDS]
        return ' '.join(words)

    def detect(self, df, return_candidates=False):
        print("Detecting recurring transactions...")
        df = df.copy()
        df['recurring_group_id'] = -1

        candidates = []

        # Use config names
        col_acc = self.fc.accountId
        col_amt = self.fc.amount
        col_desc = self.fc.text
        col_date = self.fc.date
        col_label = self.fc.label

        # Group by account
        grouped = df.groupby(col_acc)
        group_id_counter = 0

        for account_id, account_df in grouped:
            if len(account_df) < self.min_transactions:
                continue

            # Sort by amount for sliding window logic
            account_df = account_df.sort_values(col_amt)

            window_size = 200
            stride = 200

            for i in range(0, len(account_df), stride):
                window_df = account_df.iloc[i: i + window_size].copy()

                if len(window_df) < self.min_transactions:
                    continue

                raw_descriptions = window_df[col_desc].fillna('').tolist()
                cleaned_descriptions = [self._clean_text(d) for d in raw_descriptions]

                # Fallback to raw if cleaning strips everything
                final_descriptions = [c if len(c) > 2 else r for c, r in zip(cleaned_descriptions, raw_descriptions)]

                if len(final_descriptions) < 2:
                    continue

                try:
                    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                    tfidf_matrix = vectorizer.fit_transform(final_descriptions)
                    clustering = DBSCAN(eps=self.eps, min_samples=self.min_transactions, metric='cosine').fit(
                        tfidf_matrix)
                    labels = clustering.labels_
                except Exception:
                    labels = pd.factorize(window_df[col_desc])[0]

                window_df['desc_cluster'] = labels

                # Process clusters
                for cluster_id, desc_df in window_df.groupby('desc_cluster'):
                    if cluster_id == -1 or len(desc_df) < self.min_transactions:
                        continue

                    dates = desc_df[col_date].sort_values()
                    intervals = dates.diff().dt.days.dropna()

                    if len(intervals) == 0: continue

                    median_interval = intervals.median()
                    std_interval = intervals.std()

                    # Fill NaNs for safety
                    if np.isnan(std_interval): std_interval = 0.0

                    std_dom = dates.dt.day.std()
                    if np.isnan(std_dom): std_dom = 0.0

                    std_dow = dates.dt.dayofweek.std()
                    if np.isnan(std_dow): std_dow = 0.0

                    amount_std = desc_df[col_amt].std()
                    amount_mean = desc_df[col_amt].abs().mean()
                    amount_cv = (amount_std / amount_mean) if amount_mean > 0 else 0.0

                    # Feature Extraction
                    if return_candidates:
                        is_recurring_label = False
                        if col_label in desc_df.columns:
                            # Ground truth check
                            if desc_df[col_label].mean() > 0.5:
                                is_recurring_label = True

                        candidates.append({
                            'interval_std': std_interval,
                            'interval_median': median_interval,
                            'amount_cv': amount_cv,
                            'amount_std': amount_std,
                            'dom_std': std_dom,
                            'dow_std': std_dow,
                            'count': len(desc_df),
                            'days_span': (dates.max() - dates.min()).days,
                            'description_length': np.mean([len(d) for d in final_descriptions]),
                            'unique_descriptions': desc_df[col_desc].nunique(),
                            'label': is_recurring_label,
                            'group_id': group_id_counter,
                            'indices': desc_df.index.tolist()
                        })
                        group_id_counter += 1
                        continue

                    # Fallback Heuristic
                    is_periodic_interval = std_interval < self.interval_tolerance
                    is_periodic_dom = std_dom < self.dom_std_threshold
                    is_periodic_dow = std_dow < 1.0
                    is_stable_amount = amount_cv < self.amount_cv_threshold

                    is_recurring = False
                    if len(desc_df) < 3:
                        if is_periodic_interval and is_stable_amount:
                            is_recurring = True
                    else:
                        if (is_periodic_interval or is_periodic_dom or is_periodic_dow) and is_stable_amount:
                            is_recurring = True

                    if is_recurring:
                        df.loc[desc_df.index, 'recurring_group_id'] = group_id_counter
                        group_id_counter += 1

        if return_candidates:
            return pd.DataFrame(candidates)

        print(f"Detected {group_id_counter} recurring groups.")
        return df