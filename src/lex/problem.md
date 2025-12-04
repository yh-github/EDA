## Account-Level Embeddings

**Given**: A dataset of bank accounts (see all_data.csv), where each account consists of:
- **Transaction history**: Sequence of transactions over time (counter_party, amount, date, category/cpk) of up to 150 days.
- **Account metadata**: TBD (we could probably get the balance)
- **Aggregate features**: Total deposits/withdrawals, transaction frequency, etc.

**Goal**: Create embedding functions:
1. `f_txn: Transaction → ℝᵈ¹` (transaction embeddings)
2. `f_account: Account → ℝᵈ²` (account embeddings)

Where account embeddings capture behavioral patterns that enable label propagation for tasks like:
- Risk/fraud detection
- Customer segmentation
- Credit worthiness
- Account health scoring
- Churn prediction
- Prediction of taking loans, skipping payment, start using a credit card, etc.

**Key Questions to Resolve:**

1. **Should we train both jointly or hierarchically?**
   - **Hierarchical**: Train transaction embedder first, then aggregate to account level
   - **Joint**: End-to-end training where account-level objectives help shape transaction embeddings

2. **How do we aggregate transactions into account embeddings?**

3. **What temporal aspects matter?**
   - Recent vs. historical transactions (recency weighting?)
   - Spending patterns over time (trends, recurring/seasonality)
   - Transaction sequences and ordering

## Potential Approaches

**Option A: Hierarchical (Transaction → Account)**
```
Transaction → f_txn() → transaction_embedding
Account transactions → [Aggregate/Sequence Model] → account_embedding
```

**Option B: Joint Multi-Task Learning**
```
Train simultaneously with:
- Transaction-level contrastive loss
- Account-level contrastive loss
- Cross-level consistency constraints
```

**Option C: Pure Account-Level**
Skip per-transaction embeddings entirely and treat each account as a:
- Sequence of (description and/or counter_party, amount, date, category) tuples
- Feed into the embedder directly

Or other options...