from dataclasses import dataclass


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    average_entry_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0
