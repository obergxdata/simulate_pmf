from dataclasses import dataclass


@dataclass
class Settings:
    preferences: int = 5  # number of preferences for each customer
    customers: int = 1000  # number of customers in the market
    clusters: int = 5  # number of clusters for each customer
    cluster_std: int = 2  # standard deviation for each cluster
    starting_products: int = 1  # number of products in the market
    max_products: int = 6  # maximum number of products in the market
    steps: int = 100  # number of steps in the evolution
    distance_threshold: float = 0.10  # accept matches within 10% of the max L1 distance
    steps_until_death: int = 3
    snapshot_cluster_every_x_steps: int = 1
    customer_drift_amount: float = 0.5
    feature_improvement_rate: float = 0.01
    copy_feature_jitter: float = 3
