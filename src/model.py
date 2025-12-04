import numpy as np
import pandas as pd

class IVVModel:
    def __init__(self, num_dist_buckets=40, num_rank_buckets=40):
        self.num_dist_buckets = num_dist_buckets
        self.num_rank_buckets = num_rank_buckets
        # Initialize parameters with 1.0 (neutral score)
        # Using random initialization can also work but 1.0 is safer for multiplicative models initially
        self.phi_dist = np.ones(num_dist_buckets)
        self.phi_rank = np.ones(num_rank_buckets)

    def score(self, dist_bucket, rank_bucket):
        """
        Calculate score s(v, m) = Phi(D) * Phi(R)
        """
        return self.phi_dist[dist_bucket] * self.phi_rank[rank_bucket]

    def predict_proba(self, df):
        """
        Calculate probabilities for each candidate in the dataframe.
        df must contain 'stay_id', 'distance_bucket', 'rank_bucket'.
        """
        scores = self.score(df['distance_bucket'].values, df['rank_bucket'].values)
        df = df.copy()
        df['score'] = scores
        
        # Calculate sum of scores per stay_id
        score_sums = df.groupby('stay_id')['score'].transform('sum')
        df['proba'] = df['score'] / score_sums
        return df

    def log_likelihood(self, df):
        """
        Calculate Log Likelihood of the data.
        LL = sum(log(P(v_true|m)))
        """
        df_proba = self.predict_proba(df)
        # Filter for the ground truth (label == 1)
        # Assuming label column exists and is 1 for visited, 0 for others
        # If multiple labels are 1 for a stay, we sum their logs.
        # Typically one true visit per stay.
        
        true_visits = df_proba[df_proba['label'] == 1]
        ll = np.sum(np.log(true_visits['proba'] + 1e-10)) # Add epsilon for stability
        return ll

    def train(self, df, learning_rate=0.01, iterations=100, verbose=True):
        """
        Train the model using Gradient Ascent.
        """
        for it in range(iterations):
            df_proba = self.predict_proba(df)
            
            # Calculate Observed counts (where label=1)
            # We want to know how many times each bucket was associated with a TRUE visit
            true_visits = df_proba[df_proba['label'] == 1]
            
            obs_dist = np.zeros(self.num_dist_buckets)
            obs_rank = np.zeros(self.num_rank_buckets)
            
            # Count occurrences in true visits
            # Using bincount for speed
            obs_dist += np.bincount(true_visits['distance_bucket'], minlength=self.num_dist_buckets)
            obs_rank += np.bincount(true_visits['rank_bucket'], minlength=self.num_rank_buckets)
            
            # Calculate Expected counts
            # For each stay, we have probabilities for all candidates.
            # Expected count for bucket k = sum_{all candidates} P(v|m) * I(bucket(v)=k)
            
            # We can aggregate probabilities by bucket
            exp_dist = df_proba.groupby('distance_bucket')['proba'].sum().reindex(range(self.num_dist_buckets), fill_value=0).values
            exp_rank = df_proba.groupby('rank_bucket')['proba'].sum().reindex(range(self.num_rank_buckets), fill_value=0).values
            
            # Gradient (w.r.t log parameters) = Observed - Expected
            grad_dist = obs_dist - exp_dist
            grad_rank = obs_rank - exp_rank
            
            # Update parameters
            # Phi <- Phi * exp(lr * grad)
            self.phi_dist *= np.exp(learning_rate * grad_dist)
            self.phi_rank *= np.exp(learning_rate * grad_rank)
            
            # Normalize parameters to prevent explosion (optional but good for stability)
            # Since probabilities are ratios, scaling Phi doesn't change P.
            # We can fix mean or max to 1.
            self.phi_dist /= self.phi_dist.mean()
            self.phi_rank /= self.phi_rank.mean()
            
            if verbose and (it % 10 == 0 or it == iterations - 1):
                ll = self.log_likelihood(df)
                print(f"Iteration {it}: LL = {ll:.4f}")
                
        return self
