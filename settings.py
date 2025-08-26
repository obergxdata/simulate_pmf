from dataclasses import dataclass


@dataclass
class Settings:
    preferences: int = 5
    customers: int = 1000
    clusters: int = 5
    cluster_std: int = 2
    starting_products: int = 1
    max_products: int = 6
    steps: int = 100
    distance_threshold: float = 0.10
    steps_until_death: int = 3
    snapshot_cluster_every_x_steps: int = 1
    customer_drift_amount: float = 0.5
    feature_improvement_rate: float = 0.01
    copy_feature_jitter: float = 3
