import sys
from pathlib import Path

# Add parent directory to path (works regardless of where script is run from)
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from data_generator import raw_data
import pandas as pd


def build_churn_snapshot_dataset(raw_data, snapshot_freq="7D", churn_window_days=14):
    
    raw_data = raw_data.copy()
    raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
    raw_data = raw_data.sort_values(["user_id", "timestamp"])

    max_timestamp = raw_data["timestamp"].max()
    snapshot_rows = []

    for user_id, user_df in raw_data.groupby("user_id"):

        user_df = user_df.sort_values("timestamp")

        start_date = user_df["timestamp"].min().normalize()
        end_date = user_df["timestamp"].max().normalize()

        snapshot_dates = pd.date_range(start=start_date,
                                       end=end_date,
                                       freq=snapshot_freq)

        for snapshot_date in snapshot_dates:

            # --- Skip censored snapshots ---
            if snapshot_date + pd.Timedelta(days=churn_window_days) > max_timestamp:
                continue

            # --- Past data for features ---
            past_data = user_df[user_df["timestamp"] < snapshot_date]
            if len(past_data) == 0:
                continue

            days_since_last_session = (
                snapshot_date - past_data["timestamp"].max()
            ).days

            sessions_last_7d = len(
                past_data[
                    past_data["timestamp"] >= snapshot_date - pd.Timedelta(days=7)
                ]
            )

            # --- Future data for label ---
            future_data = user_df[
                (user_df["timestamp"] >= snapshot_date)
                &
                (user_df["timestamp"] <
                 snapshot_date + pd.Timedelta(days=churn_window_days))
            ]

            churn_label = 1 if len(future_data) == 0 else 0

            snapshot_rows.append({
                "user_id": user_id,
                "snapshot_date": snapshot_date,
                "days_since_last_session": days_since_last_session,
                "sessions_last_7d": sessions_last_7d,
                "churn_label": churn_label
            })

    snapshot_df = pd.DataFrame(snapshot_rows)

    return snapshot_df