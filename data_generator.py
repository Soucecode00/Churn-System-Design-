import pandas as pd
import numpy as np
from scipy.special import expit


class ChurnDataGenerator:
    def __init__(self, n_samples=5000, random_seed=42):
        self.n_samples = n_samples
        np.random.seed(random_seed)

    def generate_raw_data(self):

        start_date = pd.Timestamp("2023-10-01")

        # Random timestamps within 30 days
        random_offsets = np.random.randint(
            0, 30 * 24 * 60 * 60, size=self.n_samples
        )
        timestamps = [
            start_date + pd.Timedelta(seconds=int(offset))
            for offset in random_offsets
        ]

        # Session duration
        session_durations = np.random.gamma(
            shape=2, scale=150, size=self.n_samples
        ).astype(int)

        # Pages viewed
        pages_viewed = (
            1
            + (session_durations // 60)
            + np.random.poisson(1, self.n_samples)
        ).astype(int)

        # Purchase simulation
        is_purchase = np.random.choice(
            [0, 1], size=self.n_samples, p=[0.9, 0.1]
        )
        purchase_amounts = (
            is_purchase
            * np.random.uniform(10, 200, size=self.n_samples).round(2)
        )

        data = pd.DataFrame(
            {
                "user_id": np.random.randint(1000, 1500, size=self.n_samples),
                "timestamp": timestamps,
                "session_duration_sec": session_durations,
                "pages_viewed": pages_viewed,
                "purchase_amount": purchase_amounts,
                "device_type": np.random.choice(
                    ["mobile", "desktop", "tablet"],
                    size=self.n_samples,
                    p=[0.6, 0.3, 0.1],
                ),
            }
        )

        return data


# Execute
generator = ChurnDataGenerator(n_samples=5000)
raw_data = generator.generate_raw_data()

print(raw_data.head())