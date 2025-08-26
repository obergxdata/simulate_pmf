# Evolution Simulation

A market evolution simulation that models customer behavior, product evolution, and competitive dynamics in a multi-dimensional preference space.

## Overview

This simulation creates a market environment where:
- Customers have preferences clustered in multi-dimensional space
- Products compete by evolving their features to match customer preferences
- Products can spawn, improve, copy features from successful competitors, and die based on sales performance
- Customer preferences drift over time, creating dynamic market conditions

## Features

- **Customer Clustering**: Customers are generated with preferences clustered in multi-dimensional space using sklearn's `make_blobs`
- **Product Evolution**: Products improve their features based on customer feedback and successful competitors
- **Market Dynamics**: Products spawn and die based on performance metrics
- **Visualization**: Generate cluster plots and sales timeseries charts
- **Configurable Parameters**: Extensive settings for customizing simulation behavior

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd evo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Simulation

```bash
python sim.py
```

This will:
- Initialize a market with customers and products
- Run the simulation for the configured number of steps
- Generate visualization charts in the `charts/` directory
- Output detailed logs of product lifecycle events

### Run Tests

```bash
pytest tests.py
```

Or run with verbose output:
```bash
pytest tests.py -v
```

## Configuration

The simulation behavior can be customized by modifying the `SETTINGS` object in `sim.py` or the default values in `settings.py`:

### Key Parameters

- `steps`: Number of simulation steps to run (default: 500)
- `customers`: Number of customers in the market (default: 1000)
- `preferences`: Number of preference dimensions (default: 2)
- `clusters`: Number of customer clusters (default: 8)
- `starting_products`: Initial number of products (default: 1)
- `max_products`: Maximum number of products allowed (default: 10)
- `distance_threshold`: Purchase decision threshold (default: 0.05)
- `customer_drift_amount`: How much customer preferences drift (default: 0.15)
- `feature_improvement_rate`: Rate at which products improve (default: 0.05)

## Output

### Visualization Charts

The simulation generates two types of charts in the `charts/` directory:

1. **Cluster Plots** (`market/clusters_<step>.png`): Show customer clusters and product positions in preference space
2. **Sales Timeseries** (`sales/sales_step_<step>.png`): Show product sales performance over time

### Logging

The simulation outputs detailed logs including:
- Product lifecycle events (birth, death)
- Sales performance metrics
- Market dynamics

## Project Structure

```
evo/
├── sim.py          # Main simulation logic
├── settings.py     # Configuration parameters
├── stats.py        # Product statistics tracking
├── chart.py        # Visualization functions
├── tests.py        # Test suite
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Key Classes

- **Customer**: Represents market participants with preferences and cluster membership
- **Product**: Represents competing products with features and performance statistics
- **Market**: Manages the simulation environment and market dynamics
- **Simulation**: Main controller for running the simulation
- **ProductStats**: Tracks sales and performance metrics for products

## Algorithm Details

### Customer Behavior
- Customers purchase products within a configurable L1 distance threshold
- Customer preferences drift randomly over time to simulate changing market conditions

### Product Evolution
- Products improve features by moving toward successful customer segments
- Unsuccessful products randomize features or copy from successful competitors
- Products die if they have no sales for multiple consecutive steps

### Market Dynamics
- New products spawn randomly with a 5% probability per step
- Product spawning is limited by `max_products` parameter
- New products preferentially copy features from successful products with fewer competitors

## License

This project is open source. Please check the repository for license details.

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting pull requests:

```bash
pytest tests.py
```
