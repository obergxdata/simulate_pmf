from settings import Settings
import logging

logger = logging.getLogger(__name__)


class ProductStats:
    def __init__(self, settings: Settings, step: int):
        self.settings = settings
        self.sales = {step: 0}
        self.total_sales = 0
        self.total_distance = 0
        self.step = step
        self.clusters = []

    def get_last_x_sales(self, steps: int) -> list[int]:
        # Get the total sales for the last x steps

        periods = [i for i in range(self.step - steps + 1, self.step + 1)]
        # Check if all steps are in sales
        for period in periods:
            if period not in self.sales:
                return None

        return [self.sales[i] for i in periods]

    def get_last_x_sales_pct(self, steps: int) -> float | None:
        last_x_sales = self.get_last_x_sales(steps * 2)
        if not last_x_sales:
            return None

        # Older period = first half, Newer period = second half
        older = last_x_sales[:steps]
        newer = last_x_sales[steps:]

        old_sum = sum(older)
        new_sum = sum(newer)

        if old_sum == 0 and new_sum == 0:
            return 0.0
        if old_sum == 0:
            return float("inf") if new_sum > 0 else None

        return (new_sum - old_sum) / old_sum
