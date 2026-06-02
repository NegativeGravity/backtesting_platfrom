
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.market_view import MarketDataView, frame_to_market_arrays


@dataclass(slots=True)
class SharedMarketDataOwner:
    """Owns shared-memory OHLCV arrays for live replay worker processes."""

    descriptor_payload: dict[str, Any]
    _blocks: list[shared_memory.SharedMemory]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "SharedMarketDataOwner":
        arrays = frame_to_market_arrays(frame)
        blocks: list[shared_memory.SharedMemory] = []
        descriptor: dict[str, Any] = {"columns": {}}

        for column, values in arrays.items():
            contiguous = np.ascontiguousarray(values)
            block = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
            target = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=block.buf)
            target[:] = contiguous
            blocks.append(block)
            descriptor["columns"][column] = {
                "name": block.name,
                "shape": contiguous.shape,
                "dtype": str(contiguous.dtype),
            }

        return cls(descriptor_payload=descriptor, _blocks=blocks)

    def descriptor(self) -> dict[str, Any]:
        return self.descriptor_payload

    def close(self) -> None:
        for block in self._blocks:
            try:
                block.close()
            except FileNotFoundError:
                pass
            try:
                block.unlink()
            except FileNotFoundError:
                pass
        self._blocks.clear()


@dataclass(slots=True)
class SharedMarketDataClient:
    arrays: dict[str, np.ndarray]
    _blocks: list[shared_memory.SharedMemory]

    @classmethod
    def attach(cls, descriptor: dict[str, Any]) -> "SharedMarketDataClient":
        blocks: list[shared_memory.SharedMemory] = []
        arrays: dict[str, np.ndarray] = {}
        for column, meta in descriptor["columns"].items():
            block = shared_memory.SharedMemory(name=meta["name"])
            blocks.append(block)
            shape = tuple(meta["shape"])
            dtype = np.dtype(meta["dtype"])
            arrays[column] = np.ndarray(shape, dtype=dtype, buffer=block.buf)
        return cls(arrays=arrays, _blocks=blocks)

    def market_view(self) -> MarketDataView:
        base = {"ts", "open", "high", "low", "close", "volume"}
        return MarketDataView.from_arrays(
            ts=self.arrays["ts"],
            open_=self.arrays["open"],
            high=self.arrays["high"],
            low=self.arrays["low"],
            close=self.arrays["close"],
            volume=self.arrays["volume"],
            extra={key: value for key, value in self.arrays.items() if key not in base},
        )

    def close(self) -> None:
        for block in self._blocks:
            try:
                block.close()
            except FileNotFoundError:
                pass
        self._blocks.clear()

