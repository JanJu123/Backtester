

class TradingCost:
    def __init__(self, trading_cost_param):
        self.fees = trading_cost_param.fees
        self.slippage = trading_cost_param.slippage
        self.trading_cost_params=trading_cost_param

    def calculate_fees(self, trade_value):
        """Calculate fees based on trade value."""
        return trade_value * self.fees

    def calculate_slippage(self, trade_value):
        """Calculate slippage based on trade value."""
        return trade_value * self.slippage
    
    def calculate_fees_and_slippage(self, trade_value):
        """Calculate total cost combining fees and slippage."""
        fees = self.calculate_fees(trade_value)
        slippage = self.calculate_slippage(trade_value)
        total_cost = fees + slippage
        return total_cost
