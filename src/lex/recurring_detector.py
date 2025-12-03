import pandas as pd
import numpy as np

# best params {"eps": 0.2897552108568082, "min_transactions": 2, "prob_threshold": 0.6284580666235424}

class RecurringDetector:
    def __init__(self, interval_tolerance=10, min_transactions=2, amount_cv_threshold=0.2, dom_std_threshold=1.5,
                 eps=0.5):
        self.interval_tolerance = interval_tolerance
        self.min_transactions = min_transactions
        self.amount_cv_threshold = amount_cv_threshold
        self.dom_std_threshold = dom_std_threshold
        self.eps = eps

    def detect(self, df, return_candidates=False):
        """
        Detects recurring transactions in the dataframe.
        If return_candidates is True, returns a DataFrame of candidate groups with features.
        Otherwise, returns the original dataframe with a 'recurring_group_id' column.
        """
        print("Detecting recurring transactions...")
        df = df.copy()
        df['recurring_group_id'] = -1

        candidates = []

        # Group by account
        grouped = df.groupby('accountId')

        group_id_counter = 0

        import re
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import DBSCAN

        STOPWORDS = {
            'pos', 'debit', 'credit', 'purchase', 'payment', 'transfer',
            'check', 'withdrawal', 'deposit', 'atm', 'fee', 'charge',
            'bill', 'pay', 'online', 'mobile', 'transaction', 'authorized'
        }

        def clean_text(text):
            if not isinstance(text, str):
                return ""
            # Lowercase
            text = text.lower()
            # Remove digits and special chars (keep only letters and spaces)
            text = re.sub(r'[^a-z\s]', ' ', text)
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()

            # Remove stopwords
            words = text.split()
            words = [w for w in words if w not in STOPWORDS]
            text = ' '.join(words)

            return text

        for account_id, account_df in grouped:
            if len(account_df) < self.min_transactions:
                continue

            # Optimization: Sort by amount and process in sliding windows
            # Hypothesis: Recurring transactions are close in amount rank
            account_df = account_df.sort_values('amount')

            # Non-overlapping windows of 200
            window_size = 200
            stride = 200

            for i in range(0, len(account_df), stride):
                window_df = account_df.iloc[i: i + window_size].copy()

                if len(window_df) < self.min_transactions:
                    continue

                # Fuzzy clustering of descriptions WITHIN the window
                # Apply aggressive cleaning
                raw_descriptions = window_df['bankRawDescription'].fillna('').tolist()
                cleaned_descriptions = [clean_text(d) for d in raw_descriptions]

                # If cleaning results in empty strings (e.g. only numbers), revert to raw
                final_descriptions = [c if len(c) > 2 else r for c, r in zip(cleaned_descriptions, raw_descriptions)]

                if len(final_descriptions) < 2:
                    continue

                try:
                    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                    tfidf_matrix = vectorizer.fit_transform(final_descriptions)

                    # DBSCAN for clustering
                    # eps=0.5 for looser matching
                    clustering = DBSCAN(eps=self.eps, min_samples=self.min_transactions, metric='cosine').fit(
                        tfidf_matrix)
                    labels = clustering.labels_
                except Exception as e:
                    # Fallback to exact match if clustering fails (e.g. empty vocab)
                    labels = pd.factorize(window_df['bankRawDescription'])[0]

                window_df['desc_cluster'] = labels

                # Process each cluster
                cluster_groups = window_df.groupby('desc_cluster')

                for cluster_id, desc_df in cluster_groups:
                    if cluster_id == -1:  # Noise in DBSCAN
                        continue

                    if len(desc_df) < self.min_transactions:
                        continue

                    # Check for periodicity
                    dates = desc_df['date'].sort_values()
                    intervals = dates.diff().dt.days.dropna()

                    if len(intervals) == 0:
                        continue

                    median_interval = intervals.median()
                    std_interval = intervals.std()

                    # Check for Day of Month consistency
                    days_of_month = dates.dt.day
                    std_dom = days_of_month.std()

                    # Check for Day of Week consistency (for weekly patterns)
                    days_of_week = dates.dt.dayofweek
                    std_dow = days_of_week.std()

                    # Amount stability check
                    amount_std = desc_df['amount'].std()
                    amount_mean = desc_df['amount'].abs().mean()
                    amount_cv = amount_std / amount_mean if amount_mean > 0 else 0

                    # Feature Extraction
                    if return_candidates:
                        # Determine label (if ground truth exists)
                        is_recurring_label = False
                        if 'isRecurring' in desc_df.columns:
                            # If > 50% are recurring, label as True
                            if desc_df['isRecurring'].mean() > 0.5:
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
                            'description_length': np.mean([len(d) for d in final_descriptions]),  # Approx
                            'unique_descriptions': desc_df['bankRawDescription'].nunique(),
                            'label': is_recurring_label,
                            'group_id': group_id_counter,  # Temp ID
                            'indices': desc_df.index.tolist()
                        })
                        group_id_counter += 1
                        continue

                    # Heuristic Logic (Original)
                    is_periodic_interval = std_interval < self.interval_tolerance

                    is_periodic_dom = False
                    if std_dom < self.dom_std_threshold:
                        is_periodic_dom = True

                    is_periodic_dow = False
                    if std_dow < 1.0:
                        is_periodic_dow = True

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

    def analyze_groups(self, df):
        """
        Analyzes the detected groups.
        """
        if 'recurring_group_id' not in df.columns:
            return pd.DataFrame()

        groups = df[df['recurring_group_id'] != -1].groupby('recurring_group_id')

        stats = []
        for g_id, group in groups:
            dates = group['date'].sort_values()
            intervals = dates.diff().dt.days.dropna()

            stats.append({
                'group_id': g_id,
                'accountId': group['accountId'].iloc[0],
                'bankRawDescription': group['bankRawDescription'].iloc[0],  # Representative description
                'avg_amount': group['amount'].mean(),
                'cycle_length': intervals.median(),
                'count': len(group),
                'first_date': dates.min(),
                'last_date': dates.max()
            })

        return pd.DataFrame(stats)


if __name__ == "__main__":
    # Test with dummy data
    data = {
        'accountId': [1] * 6,
        'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                                '2023-01-05', '2023-01-10', '2023-01-15']),
        'amount': [10, 10, 10, 5, 5, 5],
        'bankRawDescription': ['Netflix', 'Netflix', 'Netflix', 'Coffee', 'Coffee', 'Coffee']
    }
    df = pd.DataFrame(data)

    detector = RecurringDetector()
    df_detected = detector.detect(df)
    print(df_detected)

    stats = detector.analyze_groups(df_detected)
    print(stats)
