"""
Delta Exchange Connector -- Wraps delta-rest-client for the AMD trading bot.

Handles:
  - API client initialization from .env
  - OHLCV candle fetching (5m)
  - Product lookup by symbol
  - Wallet balance queries
  - Position queries
  - Market and bracket order placement
  - Order cancellation
"""

import os
import time
import logging
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from delta_rest_client import DeltaRestClient, OrderType

logger = logging.getLogger("delta_connector")


class DeltaConnector:
    """
    High-level wrapper around DeltaRestClient for the AMD trading bot.

    Usage:
        connector = DeltaConnector()  # loads from .env
        df = connector.fetch_candles("BTCUSD", lookback_hours=2)
        connector.place_bracket_order(product_id, 10, "sell", sl=95000, tp=90000)
    """

    def __init__(self, env_path: str = ".env"):
        """
        Initialize the Delta Exchange connector.

        Loads API credentials from the .env file and creates
        the underlying DeltaRestClient.

        Parameters
        ----------
        env_path : str
            Path to the .env file (default: project root).
        """
        load_dotenv(env_path)

        self.api_key = os.getenv("DELTA_API_KEY")
        self.api_secret = os.getenv("DELTA_API_SECRET")
        self.base_url = os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "DELTA_API_KEY and DELTA_API_SECRET must be set in .env"
            )

        self.client = DeltaRestClient(
            base_url=self.base_url,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

        # Cache for product lookups
        self._product_cache: Dict[str, Dict] = {}
        logger.info("Delta Exchange connector initialized (base_url=%s)", self.base_url)

    # ------------------------------------------------------------------
    # Data Fetching
    # ------------------------------------------------------------------

    def fetch_candles(
        self,
        symbol: str = "BTCUSD",
        resolution: str = "5m",
        lookback_hours: int = 4,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Delta Exchange.

        Uses GET /v2/history/candles endpoint.

        Parameters
        ----------
        symbol : str
            Product symbol (e.g. "BTCUSD").
        resolution : str
            Candle resolution ("1m", "5m", "15m", "1h", "1d").
        lookback_hours : int
            How many hours of history to fetch.

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.
        """
        end_ts = int(time.time())
        start_ts = end_ts - (lookback_hours * 3600)

        response = self.client.request(
            "GET",
            "/v2/history/candles",
            query={
                "resolution": resolution,
                "symbol": symbol,
                "start": str(start_ts),
                "end": str(end_ts),
            },
            auth=False,
        )

        data = response.json()

        if not data.get("success"):
            error = data.get("error", "Unknown error")
            raise RuntimeError(f"Failed to fetch candles: {error}")

        candles = data.get("result", [])
        if not candles:
            raise RuntimeError(
                f"No candle data returned for {symbol} "
                f"(resolution={resolution}, lookback={lookback_hours}h)"
            )

        df = pd.DataFrame(candles)
        df["Datetime"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("Datetime", inplace=True)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df.sort_index(inplace=True)

        logger.info("Fetched %d candles for %s (%s)", len(df), symbol, resolution)
        return df

    # ------------------------------------------------------------------
    # Product Lookup
    # ------------------------------------------------------------------

    def get_product_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get product details by symbol.

        Parameters
        ----------
        symbol : str
            Product symbol (e.g. "BTCUSD").

        Returns
        -------
        dict
            Product info including 'id', 'symbol', 'tick_size', etc.
        """
        if symbol in self._product_cache:
            return self._product_cache[symbol]

        response = self.client.request(
            "GET",
            f"/v2/products/{symbol}",
            auth=False,
        )
        data = response.json()

        if not data.get("success"):
            raise RuntimeError(f"Failed to get product info for {symbol}")

        product = data["result"]
        self._product_cache[symbol] = product
        logger.info("Product %s -> id=%s, tick_size=%s",
                     symbol, product.get("id"), product.get("tick_size"))
        return product

    def get_product_id(self, symbol: str) -> int:
        """Get the integer product_id for a symbol."""
        return self.get_product_info(symbol)["id"]

    def get_tick_size(self, symbol: str) -> float:
        """Get the tick size for price rounding."""
        return float(self.get_product_info(symbol).get("tick_size", "0.5"))

    # ------------------------------------------------------------------
    # Wallet & Positions
    # ------------------------------------------------------------------

    def get_balance(self, asset_id: int = 5) -> Optional[Dict]:
        """
        Get wallet balance for a specific asset.

        Parameters
        ----------
        asset_id : int
            Asset ID (5 = USDT on Delta Exchange India).

        Returns
        -------
        dict or None
            Wallet info with 'balance', 'available_balance', etc.
        """
        try:
            wallet = self.client.get_balances(asset_id)
            if wallet:
                logger.info("Balance: %s (available: %s)",
                           wallet.get("balance"), wallet.get("available_balance"))
            return wallet
        except Exception as e:
            logger.error("Failed to get balance: %s", e)
            return None

    def get_open_position(self, product_id: int) -> Optional[Dict]:
        """
        Get the current open position for a product.

        Returns
        -------
        dict or None
            Position info with 'size', 'entry_price', 'side', etc.
            Returns None if no position is open.
        """
        try:
            position = self.client.get_position(product_id)
            if position and int(position.get("size", 0)) != 0:
                logger.info("Open position: size=%s, side=%s, entry=%s",
                           position.get("size"), position.get("side"),
                           position.get("entry_price"))
                return position
            return None
        except Exception as e:
            logger.error("Failed to get position: %s", e)
            return None

    # ------------------------------------------------------------------
    # Order Placement
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        product_id: int,
        size: int,
        side: str,
        reduce_only: bool = False,
    ) -> Dict:
        """
        Place a market order.

        Parameters
        ----------
        product_id : int
            Delta Exchange product ID.
        size : int
            Number of contracts.
        side : str
            "buy" or "sell".
        reduce_only : bool
            If True, only reduces an existing position.

        Returns
        -------
        dict
            Order response from Delta Exchange.
        """
        logger.info("MARKET ORDER: %s %d contracts (product_id=%d, reduce_only=%s)",
                    side.upper(), size, product_id, reduce_only)

        result = self.client.place_order(
            product_id=product_id,
            size=size,
            side=side,
            order_type=OrderType.MARKET,
            reduce_only="true" if reduce_only else "false",
        )
        logger.info("Order placed: %s", result)
        return result

    def place_limit_order(
        self,
        product_id: int,
        size: int,
        side: str,
        limit_price: float,
        reduce_only: bool = False,
    ) -> Dict:
        """
        Place a limit order.

        Parameters
        ----------
        product_id : int
            Delta Exchange product ID.
        size : int
            Number of contracts.
        side : str
            "buy" or "sell".
        limit_price : float
            Limit price.
        reduce_only : bool
            If True, only reduces an existing position.

        Returns
        -------
        dict
            Order response.
        """
        logger.info("LIMIT ORDER: %s %d @ %.2f (product_id=%d)",
                    side.upper(), size, limit_price, product_id)

        result = self.client.place_order(
            product_id=product_id,
            size=size,
            side=side,
            limit_price=str(limit_price),
            order_type=OrderType.LIMIT,
            reduce_only="true" if reduce_only else "false",
        )
        logger.info("Order placed: %s", result)
        return result

    def place_stop_loss(
        self,
        product_id: int,
        size: int,
        side: str,
        stop_price: float,
    ) -> Dict:
        """
        Place a stop-loss order.

        Parameters
        ----------
        product_id : int
            Delta Exchange product ID.
        size : int
            Number of contracts.
        side : str
            "buy" (for short SL) or "sell" (for long SL).
        stop_price : float
            Trigger price.

        Returns
        -------
        dict
            Order response.
        """
        logger.info("STOP LOSS: %s %d @ stop=%.2f (product_id=%d)",
                    side.upper(), size, stop_price, product_id)

        result = self.client.place_stop_order(
            product_id=product_id,
            size=size,
            side=side,
            stop_price=str(stop_price),
            order_type=OrderType.MARKET,
        )
        logger.info("Stop order placed: %s", result)
        return result

    def place_bracket_order(
        self,
        product_id: int,
        size: int,
        side: str,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        tick_size: float = 0.5,
    ) -> Dict:
        """
        Place a bracket order (entry + SL + TP).

        Uses Delta Exchange's bracket order API for atomic SL/TP attachment.

        Parameters
        ----------
        product_id : int
            Product ID.
        size : int
            Number of contracts.
        side : str
            "buy" or "sell".
        entry_price : float, optional
            Limit price for entry. If None, uses market order.
        stop_loss_price : float, optional
            Stop-loss trigger price.
        take_profit_price : float, optional
            Take-profit trigger price.
        tick_size : float
            Tick size for price rounding.

        Returns
        -------
        dict
            Order response.
        """
        from delta_rest_client import round_by_tick_size

        order = {
            "product_id": product_id,
            "size": int(size),
            "side": side,
        }

        if entry_price is not None:
            order["order_type"] = "limit_order"
            order["limit_price"] = str(round_by_tick_size(entry_price, tick_size))
        else:
            order["order_type"] = "market_order"

        if stop_loss_price is not None:
            order["bracket_stop_loss_price"] = str(
                round_by_tick_size(stop_loss_price, tick_size)
            )
            order["bracket_stop_trigger_method"] = "last_traded_price"

        if take_profit_price is not None:
            order["bracket_take_profit_price"] = str(
                round_by_tick_size(take_profit_price, tick_size)
            )

        logger.info(
            "BRACKET ORDER: %s %d contracts | entry=%s | SL=%s | TP=%s",
            side.upper(), size,
            order.get("limit_price", "MARKET"),
            order.get("bracket_stop_loss_price", "N/A"),
            order.get("bracket_take_profit_price", "N/A"),
        )

        result = self.client.create_order(order)
        logger.info("Bracket order placed: %s", result)
        return result

    # ------------------------------------------------------------------
    # Order Management
    # ------------------------------------------------------------------

    def cancel_all_orders(self, product_id: int) -> Dict:
        """Cancel all active orders for a product."""
        logger.info("Cancelling all orders for product_id=%d", product_id)
        response = self.client.request(
            "DELETE",
            "/v2/orders/all",
            payload={"product_id": product_id},
            auth=True,
        )
        result = response.json()
        logger.info("Cancel all result: %s", result)
        return result

    def get_active_orders(self, product_id: int) -> list:
        """Get all active (open) orders for a product."""
        try:
            result = self.client.get_live_orders(
                query={"product_id": product_id}
            )
            return result if result else []
        except Exception as e:
            logger.error("Failed to get active orders: %s", e)
            return []
