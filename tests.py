import pytest
from unittest.mock import patch
from sim import Customer, Product, Market, Simulation
from stats import ProductStats
import numpy as np
import random


@patch("sim.SETTINGS")
def test_create_customer(mock_settings):
    mock_settings.customers = 1000
    mock_settings.preferences = 10
    mock_settings.min_budget = 100.0
    mock_settings.max_budget = 1000.0
    mock_settings.clusters = 5
    mock_settings.cluster_std = 2
    customer = Customer(preferences=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cluster=1)
    assert len(customer.preferences) == 10
    assert customer.budget is not None
    assert 100.0 <= customer.budget <= 1000.0


@patch("sim.SETTINGS")
def test_create_product(mock_settings):
    mock_settings.preferences = 5
    mock_settings.min_budget = 100.0
    mock_settings.max_budget = 1000.0
    product = Product(features=[1, 1, 1, 1, 1], nr=1)
    assert all(isinstance(p, int) for p in product.features)
    assert all(1 <= p <= 10 for p in product.features)
    assert len(product.features) == 5
    assert 100.0 <= product.price <= 1000.0


@patch("sim.SETTINGS")
def test_create_market(mock_settings):
    mock_settings.customers = 1000
    mock_settings.starting_products = 10
    mock_settings.preferences = 10
    mock_settings.clusters = 5
    mock_settings.cluster_std = 2
    market = Market()
    assert len(market.customers) == 1000
    assert len(market.products) == 10


def test_l1_distance():
    market = Market()
    product = Product(features=[10, 1, 10], nr=1)
    customer = Customer(preferences=[10, 10, 1], cluster=1)
    distance = market.l1_distance(customer.preferences, product.features)
    assert distance == 18


def test_pick_product():
    market = Market()
    market.take_step()
    # Create two products with different features
    product_1 = Product(features=[10, 10, 10, 10, 10], nr=1)
    product_2 = Product(features=[0, 0, 0, 0, 0], nr=2)
    product_1.stats.sales[1] = 0
    product_1.stats.sales[1] = 10
    # Create a customer with a preference for the first product
    customer = Customer(preferences=[10, 10, 10, 10, 10], cluster=1)
    # Add the products to the market
    market.products = [product_1, product_2]
    # Add the customer to the market
    market.customers = [customer]

    picks = []
    for _ in range(10):
        product = market.pick_product(customer)
        picks.append(product.name)

    # Pick the product
    assert picks.count(product_1.name) > picks.count(product_2.name)


@patch("sim.SETTINGS")
def test_drift_customers(mock_settings):
    mock_settings.customer_drift_amount = 1
    mock_settings.preferences = 4
    mock_settings.customers = 1000
    mock_settings.clusters = 4
    mock_settings.cluster_std = 1

    market = Market()
    # Copy customers
    customers_1 = [customer.preferences.copy() for customer in market.customers]
    market.drift_customers()
    customers_2 = [customer.preferences.copy() for customer in market.customers]

    diffs = 0
    for customer_1, customer_2 in zip(customers_1, customers_2):
        for prefrence_1, prefrence_2 in zip(customer_1, customer_2):
            if prefrence_1 != prefrence_2:
                diffs += 1

    assert diffs == 250


@patch("sim.SETTINGS")
def test_improve_features(mock_settings):
    mock_settings.preferences = 4
    mock_settings.customers = 1000
    mock_settings.clusters = 4
    mock_settings.cluster_std = 1
    mock_settings.feature_improvement_rate = 0.5
    market = Market()
    market.customers = [Customer(preferences=[5, 5, 5, 5], cluster=1)] * 10
    product = Product(features=[6, 6, 6, 6], nr=1)
    product.improve_features(market.customers)

    assert product.features == [5.5, 5.5, 5.5, 5.5]


def test_research():
    product = Product(features=[10, 10, 10, 10], nr=1)
    product.create_feature(min_prefs=0, max_prefs=10)
    assert product.features != [10, 10, 10, 10]


def test_get_pct_of():
    market = Market()
    market.products = [Product(features=[10, 10, 10, 10], nr=1)] * 100
    assert len(market.get_pct_of(market.products, 0.35)) == 35


def test_find_closest_customers():
    market = Market()

    customers_1 = [Customer(preferences=[10, 10, 10, 10], cluster=1)] * 50
    customers_2 = [Customer(preferences=[5, 5, 5, 5], cluster=2)] * 50
    market.customers = customers_1 + customers_2

    product_1 = Product(features=[10, 10, 10, 10], nr=1)
    product_2 = Product(features=[5, 5, 5, 5], nr=2)

    closest_product_1 = market.find_closest_customers(product_1)
    closest_product_2 = market.find_closest_customers(product_2)

    closest_product_1 = closest_product_1[:10]
    closest_product_2 = closest_product_2[:10]

    assert len(closest_product_1) == 10
    assert len(closest_product_2) == 10
    assert closest_product_1 != closest_product_2


@patch("sim.SETTINGS")
def test_get_neighborhoods(mock_settings):
    mock_settings.preferences = 4
    mock_settings.customers = 1000
    mock_settings.clusters = 4
    mock_settings.cluster_std = 1
    mock_settings.distance_threshold = 0.02
    market = Market()
    product_1 = Product(features=[5, 5, 5, 5], nr=1, step=1)
    product_2 = Product(features=[5, 5, 5, 5], nr=2, step=1)
    product_3 = Product(features=[4, 4, 4, 4], nr=3, step=1)
    product_4 = Product(features=[5, 5, 5, 5], nr=4, step=1)
    product_5 = Product(features=[9, 9, 9, 9], nr=5, step=1)
    market.products = [product_1, product_2, product_3, product_4, product_5]
    neighborhoods = market.get_neighborhoods(market.products)
    neighborhood_sizes = [len(neighborhood) for neighborhood in neighborhoods.values()]
    assert neighborhood_sizes == [2, 2, 0, 2, 0]


@patch("sim.SETTINGS")
def test_spawn_new_product(mock_settings):
    mock_settings.preferences = 4
    mock_settings.customers = 1000
    mock_settings.clusters = 4
    mock_settings.cluster_std = 1
    mock_settings.distance_threshold = 0.02
    mock_settings.max_products = 10
    mock_settings.copy_feature_jitter = 0

    market = Market()
    product_1 = Product(features=[5, 5, 5, 5], nr=1, step=1)
    product_1.stats.total_sales = 10
    product_2 = Product(features=[5, 5, 5, 5], nr=2, step=1)
    product_2.stats.total_sales = 20
    product_3 = Product(features=[5, 5, 5, 5], nr=3, step=1)
    product_3.stats.total_sales = 30
    product_4 = Product(features=[5, 5, 5, 5], nr=4, step=1)
    product_4.stats.total_sales = 40
    product_5 = Product(features=[9.666, 9.666, 9.666, 9.666], nr=5, step=1)
    product_5.stats.total_sales = 35
    market.products = [product_1, product_2, product_3, product_4, product_5]
    new_product = market.spawn_new_product()
    assert new_product is not None
    assert len(market.products) == 6
    assert 9.666 in new_product.features


@patch("sim.SETTINGS")
def test_copy_feature(mock_settings):
    mock_settings.preferences = 4
    mock_settings.customers = 1000
    mock_settings.clusters = 4
    mock_settings.cluster_std = 1
    mock_settings.copy_feature_jitter = 5
    product_1 = Product(features=[10, 10, 10, 10], nr=1)
    product_2 = Product(features=[0, 0, 0, 0], nr=2)
    product_1.copy_feature(product_2, 2)

    changed = [abs(feature) for feature in product_1.features if feature != 10]
    assert len(changed) == 2
    assert all(cf > 0 and cf < mock_settings.copy_feature_jitter for cf in changed)


@patch("sim.SETTINGS")
def test_product_stats(mock_settings):
    mock_settings.customers = 100
    stats = ProductStats(settings=mock_settings, step=1)
    stats.step = 1
    stats.sales[1] = 10
    stats.step = 2
    stats.sales[2] = 20
    stats.step = 3
    stats.sales[3] = 25

    assert stats.get_last_x_sales(1) == [25]
    assert stats.get_last_x_sales(2) == [20, 25]
    assert stats.get_last_x_sales(3) == [10, 20, 25]
    assert stats.get_last_x_sales(5) is None

    stats.step = 4
    stats.sales[4] = 30

    assert stats.get_last_x_sales_pct(1) == 0.2
    assert round(stats.get_last_x_sales_pct(2), 2) == 0.83
    assert stats.get_last_x_sales_pct(3) is None
    assert stats.get_last_x_sales_pct(10) is None
