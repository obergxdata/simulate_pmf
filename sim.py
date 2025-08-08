from __future__ import annotations
import random
import uuid
from settings import Settings
from stats import ProductStats
from sklearn.datasets import make_blobs
import logging
from chart import plot_clusters, plot_timeseries
import numpy as np

SETTINGS = Settings(
    steps=500,
    preferences=2,
    cluster_std=1.3,
    clusters=8,
    starting_products=1,
    max_products=10,
    snapshot_cluster_every_x_steps=100,
    distance_threshold=0.05,
    steps_until_death=3,
    customer_drift_amount=0.15,
    feature_improvement_rate=0.05,
)

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("umap").setLevel(logging.ERROR)


class Customer:
    def __init__(self, preferences: list[int], cluster: int):
        self.name = self.generate_name()
        self.preferences = preferences
        self.cluster: int = cluster
        self.product = None
        self.previous_mutation = (0, 0)

    def generate_name(self):
        return str(uuid.uuid4())

    def __str__(self):
        return f"Customer(name={self.name}, preferences={self.preferences})"


class Product:
    def __init__(self, features: list[int], nr: int, step: int):
        self.features = features
        self.name = f"Product {nr}"
        self.stats = ProductStats(settings=SETTINGS, step=step)
        self.alive = True

    def copy_feature(self, mate: Product, nr_of_features: int = 1):
        # Select x random features and copy them from mate
        features_to_copy = random.sample(range(len(self.features)), nr_of_features)
        for feature_idx in features_to_copy:
            exact_copy = mate.features[feature_idx]
            jitter = random.uniform(
                -SETTINGS.copy_feature_jitter, SETTINGS.copy_feature_jitter
            )
            self.features[feature_idx] = exact_copy + jitter

    def improve_features(self, customers: list[Customer]):
        # Get the mean of the customer sample
        mean_prefs = np.mean([customer.preferences for customer in customers], axis=0)
        # dirft features towards the mean
        for i, feature in enumerate(self.features):
            self.features[i] = (
                feature + (mean_prefs[i] - feature) * SETTINGS.feature_improvement_rate
            )

    def create_feature(self, min_prefs: float, max_prefs: float):
        # Random step
        for i, feature in enumerate(self.features):
            self.features[i] = random.uniform(min_prefs, max_prefs)

    def __str__(self):
        return self.name


class Market:
    def __init__(self):
        self._step = 0
        self.prefs, self.cluster_map = self.gen_prefrences()
        self.max_prefs = max(max(prefs) for prefs in self.prefs)
        self.min_prefs = min(min(prefs) for prefs in self.prefs)
        self.customers = self.gen_customers()
        self.max_l1_distance = SETTINGS.preferences * (self.max_prefs - self.min_prefs)
        self.product_nr = 0
        self.products = self.gen_products()
        self.unserved = len(self.customers)

    def take_step(self):
        self._step += 1
        self.unserved = len(self.customers)
        self.sync_products_step()

    def sync_products_step(self):
        for product in self.products:
            product.stats.step = self._step
            if self._step not in product.stats.sales:
                product.stats.sales[self._step] = 0

    @property
    def step(self):
        return self._step

    def run(self):

        self.take_step()
        self.drift_customers()

        # Pick a product for each customer
        for customer in self.customers:
            self.pick_product(customer)

        self.kill_products()
        if random.random() < 0.05:
            self.spawn_new_product()

        if (
            self.step % SETTINGS.snapshot_cluster_every_x_steps == 0
            or self.step == 1
            or self.step == SETTINGS.steps
        ):
            plot_clusters(
                X=self.prefs,
                y=self.cluster_map,
                filename=f"market/clusters_{self.step}.png",
                products=self.products,
            )

        # Improve features of all products
        for product in self.get_active_products():
            sales_change = product.stats.get_last_x_sales_pct(2)
            if product.stats.sales[self.step] == 0 or (
                sales_change and sales_change < -0.5
            ):
                # 50%
                product.create_feature(
                    min_prefs=self.min_prefs, max_prefs=self.max_prefs
                )
                continue
            else:
                closest_customers = self.find_closest_customers(product)
                if closest_customers:
                    sample_customers = self.get_pct_of(closest_customers, 0.1)
                    product.improve_features(sample_customers)

    def find_closest_customers(self, product: Product):

        closest_customers = [
            (customer, self.l1_distance(customer.preferences, product.features))
            for customer in self.customers
        ]

        closest_customers.sort(key=lambda x: x[1])

        return [customer for customer, _ in closest_customers]

    def pick_product(self, customer: Customer):

        possible_choices = []
        for product in self.get_active_products():
            distance, distance_value = self.within_l1_distance(
                customer.preferences, product.features
            )
            if distance:
                possible_choices.append((product, distance_value))

        if not possible_choices:
            return None

        self.unserved -= 1
        # Get the product with the lowest distance
        choice = min(possible_choices, key=lambda x: x[1])[0]

        choice.stats.sales[self.step] += 1
        choice.stats.total_sales += 1
        choice.stats.clusters.append(customer.cluster)
        return choice

    def l1_distance(self, item_1: list, item_2: list) -> float:
        x = sum(abs(p - f) for p, f in zip(item_1, item_2))

        return x

    def within_l1_distance(self, item_1: list, item_2: list) -> bool:
        distance = self.l1_distance(item_1, item_2)
        return distance < self.max_l1_distance * SETTINGS.distance_threshold, distance

    def drift_customers(self) -> None:
        # Will drift one feature of one cluster

        # Select random cluster from numpy array of clusters
        cluster = random.choice(list(set(self.cluster_map)))

        # Select those customers in the cluster
        customers_in_cluster = [
            customer for customer in self.customers if customer.cluster == cluster
        ]

        # Decide what prefrences to drift
        prefrence_to_drift = random.sample(range(SETTINGS.preferences), 1)[0]
        # Decide how much to drift
        drift_amount = random.uniform(
            -SETTINGS.customer_drift_amount, SETTINGS.customer_drift_amount
        )

        # Change customer's preferences
        for customer in customers_in_cluster:
            customer.preferences[prefrence_to_drift] = (
                customer.preferences[prefrence_to_drift] + drift_amount
            )

    def get_neighborhoods(
        self, products: list[Product]
    ) -> dict[Product, list[Product]]:
        neighborhoods = {}
        for product in products:
            neighborhoods[product] = []
            for other_product in products:
                if product.alive and other_product.alive:
                    if product == other_product:
                        continue
                    distance, distance_value = self.within_l1_distance(
                        product.features, other_product.features
                    )
                    if distance:
                        neighborhoods[product].append(other_product)

        return neighborhoods

    def spawn_new_product(self):

        active_products = self.get_active_products()
        if len(active_products) >= SETTINGS.max_products:
            return

        if len(active_products) == 0:
            new_product = self.gen_product()
            self.products.append(new_product)
            self.sync_products_step()
        elif active_products:
            top_sellers = self.get_top_sellers()
            neighborhoods = self.get_neighborhoods(top_sellers)
            least_neighbors = min(neighborhoods, key=lambda x: len(neighborhoods[x]))
            new_product = self.gen_product(copy_from=least_neighbors)
            self.products.append(new_product)
            self.sync_products_step()

        return new_product

    def kill_products(self):
        # Kill products
        for product in self.get_active_products():
            last_x_sales = product.stats.get_last_x_sales(
                steps=SETTINGS.steps_until_death
            )
            sales_change = product.stats.get_last_x_sales_pct(
                SETTINGS.steps_until_death
            )
            # Kill product if it has no sales for a while
            if last_x_sales and sum(last_x_sales) == 0 and product.alive:
                product.alive = False
                logger.info(
                    f"Product {product.name} is dead, step {self.step}, sales {product.stats.sales}"
                )
            elif sales_change and sales_change < -0.5 and product.alive:
                product.alive = False
                logger.info(
                    f"Product {product.name} is dead, step {self.step}, sales {product.stats.sales}, pct_change {sales_change}"
                )

    def gen_customers(self):
        return [
            Customer(self.prefs[i], self.cluster_map[i])
            for i in range(SETTINGS.customers)
        ]

    def gen_products(self):
        products = []
        for _ in range(SETTINGS.starting_products):
            products.append(self.gen_product())
        return products

    def gen_product(self, copy_from: Product = None):
        self.product_nr += 1
        product = Product(
            features=[
                random.uniform(self.min_prefs, self.max_prefs)
                for _ in range(SETTINGS.preferences)
            ],
            nr=self.product_nr,
            step=self.step,
        )
        if copy_from:
            nr_of_features = random.randint(1, len(copy_from.features))
            product.copy_feature(copy_from, nr_of_features=nr_of_features)
        return product

    def gen_prefrences(self):
        X, y = make_blobs(
            n_samples=SETTINGS.customers,
            n_features=SETTINGS.preferences,
            centers=SETTINGS.clusters,
            cluster_std=SETTINGS.cluster_std,
        )
        return X, y

    def get_top_sellers(self):
        # return sorted list by product sales
        return sorted(
            self.get_active_products(), key=lambda x: x.stats.total_sales, reverse=True
        )

    def get_pct_of(self, _list: list, pct: float):
        return _list[: int(len(_list) * pct)]

    def get_active_products(self):
        return [product for product in self.products if product.alive]

    def __str__(self):
        return f"Market(customers={self.customers}, products={self.products})"


class Simulation:
    def __init__(self):
        self.market = Market()

    def __str__(self):
        return f"Simulation(market={self.market})"

    def run(self):
        for _ in range(SETTINGS.steps):
            self.market.run()


def main():
    sim = Simulation()
    sim.run()

    # Generate timeseries chart at the end
    plot_timeseries(
        products=sim.market.products,
        filename=f"sales/sales_step_{sim.market.step}",
        title=f"Product Sales Over Time (Steps 1-{sim.market.step})",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
