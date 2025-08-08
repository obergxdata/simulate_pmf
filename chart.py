from matplotlib import pyplot as plt
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


def plot_clusters(
    X,
    y,
    filename: str,
    products: list = None,
):
    # Clear any existing plots to prevent accumulation
    plt.clf()

    # Extract dots and labels from products if provided
    dots = None
    dots_labels = None
    if products:
        dots = [product.features for product in products]
        dots_labels = [product.stats.total_sales for product in products]

    # Create the main scatter plot
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.7, zorder=1)

    # Add dots on top with unique colors and create legend
    legend_elements = []
    if dots:
        # Use a colormap that can handle many products
        # Tab20 provides 20 distinct colors, and we can cycle through them
        if len(dots) <= 20:
            colors = plt.cm.tab20(range(len(dots)))
        else:
            # For even more products, use a continuous colormap
            colors = plt.cm.hsv([i / len(dots) for i in range(len(dots))])

        for i, point in enumerate(dots):
            # Get the color for this product
            dot_color = colors[i]

            # Check if this product is dead
            is_dead = False
            if products and i < len(products):
                is_dead = not getattr(products[i], "alive", True)

            alpha = 0.6 if is_dead else 1.0

            # Create scatter plot for each dot
            scatter = plt.scatter(
                point[0],
                point[1],
                c=[dot_color],
                marker="o",
                s=200,
                linewidths=2,
                edgecolors="black",
                alpha=alpha,
                zorder=2,
            )

            # Add a cross marker for dead products
            if is_dead:
                plt.scatter(
                    point[0],
                    point[1],
                    marker="x",
                    s=100,
                    c="red",
                    linewidths=3,
                    zorder=3,
                )

            # Add product name as text label
            if products and i < len(products):
                plt.annotate(
                    products[i].name,
                    (point[0], point[1]),
                    xytext=(5, 5),  # Offset text slightly from the dot
                    textcoords="offset points",
                    fontsize=6,
                    alpha=0.8 if is_dead else 1.0,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                    zorder=4,
                )

            # Add to legend if we have labels
            if dots_labels and i < len(dots_labels):
                label = f"{dots_labels[i]:.1f}"
                if is_dead:
                    label += " (dead)"

                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=dot_color,
                        markersize=10,
                        markeredgecolor="black",
                        markeredgewidth=2,
                        alpha=alpha,
                        label=label,
                    )
                )

            # Create legend on the right side
            if legend_elements:
                plt.legend(
                    handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5)
                )

    # Ensure the charts directory exists
    os.makedirs("charts", exist_ok=True)

    # Save the figure
    plt.savefig(f"charts/{filename}", dpi=150, bbox_inches="tight")

    # Clear the figure to free memory
    plt.clf()


def plot_timeseries(
    products: list,
    filename: str,
    title: str = "Product Sales Over Time",
    plot_type: str = "line",
):
    """
    Plot time series chart showing sales per step for each product.
    Uses the same color scheme as plot_clusters for consistency.

    Args:
        products: List of product objects with stats.sales attribute
        filename: Name of the output file (without extension)
        title: Title for the chart
        plot_type: Type of plot - "line", "scatter", or "bar"
    """
    # Clear any existing plots
    plt.clf()

    # Set up the figure
    plt.figure(figsize=(12, 8))

    if not products:
        logger.warning("No products provided for timeseries plot")
        return

    # Use the same colormap as plot_clusters for consistency
    if len(products) <= 20:
        colors = plt.cm.tab20(range(len(products)))
    else:
        # For even more products, use a continuous colormap
        colors = plt.cm.hsv([i / len(products) for i in range(len(products))])

    # No markers/dots for cleaner timeseries lines
    marker = None
    markersize = 0

    for i, product in enumerate(products):
        # Extract sales data
        sales_data = product.stats.sales

        # Convert to sorted lists for plotting
        x_values = sorted(sales_data.keys())
        y_values = [sales_data[x] for x in x_values]

        # Get the base color for this product (same as plot_clusters)
        base_color = colors[i]

        # Determine color and style based on product status
        if hasattr(product, "alive") and not product.alive:
            # Dead products use a grayed-out version of their original color
            if isinstance(base_color, tuple) and len(base_color) >= 3:
                # Average the RGB values with gray to create a muted version
                gray_factor = (
                    0.7  # How much to gray out (0.0 = original, 1.0 = pure gray)
                )
                r, g, b = base_color[:3]
                gray_value = (r + g + b) / 3
                color = (
                    r * (1 - gray_factor) + gray_value * gray_factor,
                    g * (1 - gray_factor) + gray_value * gray_factor,
                    b * (1 - gray_factor) + gray_value * gray_factor,
                    base_color[3] if len(base_color) > 3 else 1.0,
                )
            else:
                color = "gray"
            alpha = 0.6  # Consistent transparency for dead products
            linestyle = "--"  # Dashed line for dead products
            label = f"{product.name} (dead)"
        else:
            # Alive products use the original colormap
            color = base_color
            alpha = 1.0  # Consistent full opacity for alive products
            linestyle = "-"  # Solid line for alive products
            label = product.name

        # Create the plot based on type
        if plot_type == "line":
            plt.plot(
                x_values,
                y_values,
                marker=marker,
                linewidth=2,
                markersize=markersize,
                color=color,
                alpha=alpha,
                linestyle=linestyle,
                label=label,
            )
        elif plot_type == "scatter":
            plt.scatter(
                x_values,
                y_values,
                c=[color],
                alpha=alpha,
                s=50,
                label=label,
            )
        elif plot_type == "bar":
            # For bar plots, offset each product slightly
            x_offset = [x + i * 0.8 / len(products) for x in x_values]
            plt.bar(
                x_offset,
                y_values,
                width=0.8 / len(products),
                color=color,
                alpha=alpha,
                label=label,
            )

    # Customize the plot
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Add legend if multiple products
    if len(products) > 1:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Ensure the charts directory exists
    os.makedirs("charts", exist_ok=True)

    # Save the figure
    plt.savefig(f"charts/{filename}.png", dpi=150, bbox_inches="tight")
    logger.info(f"Timeseries chart saved as charts/{filename}.png")

    # Clear the figure to free memory
    plt.clf()
