import numpy as np
import pandas as pd

class IVVModel:
    def __init__(self, bucket_feature):
        """
        bucket_feature: list of dict
          [
            {"feature_name": "distance_bucket", "num_bucket": 40},
            {"feature_name": "rank_bucket", "num_bucket": 40},
            ...
          ]
        """
        self.features = {}
        for f in bucket_feature:
            fname = f["feature_name"]
            nb = f["num_bucket"]
            self.features[fname] = {
                "num_bucket": nb,
                "phi": np.ones(nb)
            }

    def score(self, df):
        """
        Calculate score s(v, m) = product over features of Phi_f(bucket_f)
        """
        score = np.ones(len(df))
        for fname, finfo in self.features.items():
            buckets = df[fname].values
            score *= finfo["phi"][buckets]
        return score

    def predict_proba(self, df):
        """
        df must contain:
          - stay_id
          - all discretized feature columns (e.g. distance_bucket, rank_bucket)
        """
        df = df.copy()
        df["score"] = self.score(df)

        # normalize within stay_id
        score_sum = df.groupby("stay_id")["score"].transform("sum")
        df["proba"] = df["score"] / score_sum
        return df

    def log_likelihood(self, df):
        """
        LL = sum_m log P(v_true | m)
        Assumes exactly one label=1 per stay_id
        """
        df_proba = self.predict_proba(df)
        true_visits = df_proba[df_proba["label"] == 1]
        ll = np.sum(np.log(true_visits["proba"] + 1e-10))
        return ll

    def train(self, df, learning_rate=0.01, iterations=100, verbose=True):
        """
        Train using Gradient Ascent.
        """
        for it in range(iterations):
            df_proba = self.predict_proba(df)

            # observed counts (label=1 only)
            true_visits = df_proba[df_proba["label"] == 1]

            for fname, finfo in self.features.items():
                nb = finfo["num_bucket"]

                # observed
                obs = np.bincount(
                    true_visits[fname],
                    minlength=nb
                )

                # expected
                exp = (
                    df_proba
                    .groupby(fname)["proba"]
                    .sum()
                    .reindex(range(nb), fill_value=0)
                    .values
                )

                # gradient
                grad = obs - exp

                # update (multiplicative, log-linear)
                finfo["phi"] *= np.exp(learning_rate * grad)

                # normalize (scale invariance)
                finfo["phi"] /= finfo["phi"].mean()

            if verbose and (it % 10 == 0 or it == iterations - 1):
                ll = self.log_likelihood(df)
                print(f"Iteration {it}: LL = {ll:.4f}")

        # return learned parameters (copy)
        return {
            fname: finfo["phi"].copy()
            for fname, finfo in self.features.items()
        }
