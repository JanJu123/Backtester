import pandas as pd
import numpy as np
import optuna
import copy
from tqdm import tqdm


# Import the core function from your main script
from core.backtest_runner import run_single_backtest
from core.utils import set_nested_value, set_nested_attr

class OptunaOptimizer:
    def __init__(self, df, params, strategy_name, initial_capital, search_space):
        self.df = df
        self.base_params = params
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.search_space = search_space

    def _objective(self, trial: optuna.trial.Trial, data_slice):
        """The objective function that Optuna will optimize on a given slice of data."""
        trial_params = copy.deepcopy(self.base_params)

        # --- Define the parameters you want Optuna to tune ---
        # You can add more parameters here as needed

        # ma_type = trial.suggest_categorical('ma_type', ['ema', 'sma', 'vwma', 'vwema'])

        #!Look here
        # Loop through the configuration you passed in
        for param_config in self.search_space:
            # STEP 1: Get the right "suggest_*" method from the trial object
            suggest_method = getattr(trial, param_config['method'])
            
            # Get the arguments for the method
            args = param_config.get('args', [])
            kwargs = param_config.get('kwargs', {})
            
            # STEP 2: Call the method to create the 'suggested_value'
            suggested_value = suggest_method(param_config['name'], *args, **kwargs)
            
            # STEP 3: Use the helper function to set the new value in your params dictionary
            set_nested_value(trial_params, param_config['path'], suggested_value)
        

        # macd_cfg.ma_func_fast = ma_type  # Use the chosen MA type
        # macd_cfg.ma_func_slow = ma_type  # Use the same for both for simplicity



        # Run the backtest on the training data with the trial parameters
        _ , _ , summary, _ , _ , _ = run_single_backtest(
            df=data_slice,
            params=trial_params,
            strategy_name=self.strategy_name,
            initial_capital=self.initial_capital,
            print_summary=False
        )

        num_trades = summary.get("num_trades", 0)
        # expectancy = summary.get("expectancy", -2)
        sharpe = summary.get("sharpe", -5)

        # GATE 1:
        if num_trades < 100:
            return -5.0

        # # GATE 2:
        # if expectancy <= 0:
        #     return -2.0

        return sharpe # Return Score


    def run_simple_optimization(self, n_trials=100,  data_slice=None, storage_location: str = None):
        """Runs a single, simple optimization on a given block of data."""
        if data_slice is None:
            data_slice = self.df

        print(f"--- Starting Simple Optimization on data from {data_slice.iloc[0]['Datetime']} to {data_slice.iloc[-1]['Datetime']} ---")


        if storage_location:
            study = optuna.create_study(direction='maximize', storage=storage_location)
        else:
            study = optuna.create_study(direction='maximize')

        objective_func = lambda trial: self._objective(trial, data_slice=data_slice)
        study.optimize(objective_func, n_trials=n_trials)
        
        print("\n--- Optimization Complete ---")
        print(f"Best Sharpe Ratio: {study.best_value}")
        print("Best Parameters:")
        print(study.best_params)
        return study.best_params, study


    def run_walk_forward(self, start_date, end_date, train_period_months, test_period_months, n_trials_per_window=50, strategy_name: str = None):
        """Executes the entire walk-forward analysis with a progress bar."""
        
        # --- Input Validation ---
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        user_start_date = pd.to_datetime(start_date)
        user_end_date = pd.to_datetime(end_date)
        
        data_start_date = self.df['Datetime'].min()
        data_end_date = self.df['Datetime'].max()
        
        if user_start_date < data_start_date:
            raise ValueError(f"Provided start_date ({user_start_date.date()}) is before the earliest date in the data ({data_start_date.date()}).")
            
        if user_end_date > data_end_date:
            raise ValueError(f"Provided end_date ({user_end_date.date()}) is after the latest date in the data ({data_end_date.date()}).")


        # --- Pre-calculate all the windows to use with tqdm ---
        windows = []
        window_start = user_start_date
        train_period = pd.DateOffset(months=train_period_months)
        test_period = pd.DateOffset(months=test_period_months)

        while window_start + train_period + test_period <= user_end_date:
            train_end = window_start + train_period
            test_end = train_end + test_period
            windows.append((window_start, train_end, test_end))
            window_start += test_period
        
        # --- Run the main loop with tqdm for a progress bar ---
        all_oos_trades = []
        all_oos_signals = []
        last_study = None
        best_params_per_window = {}

        storage_location = "sqlite:///optimization_results/walk_forward_studies.db" #* Kje se shranjuje study
        
        print(f"\n--- Starting Walk-Forward Analysis for {len(windows)} windows ---")
        
        for start, train_end, test_end in tqdm(windows, desc="Walk-Forward Progress"):
            
            # 1. Slice the data for the current window
            train_df = self.df[(self.df['Datetime'] >= start) & (self.df['Datetime'] < train_end)]
            test_df = self.df[(self.df['Datetime'] >= train_end) & (self.df['Datetime'] < test_end)]
            
            # if train_df.empty or test_df.empty:
            #     continue
            
            # 2. Run simple optimization on the training data slice
            best_params_dict, last_study = self.run_simple_optimization(train_df, n_trials=n_trials_per_window, storage_location=storage_location)
            
                       # 3. Create a final params object with the best parameters (DYNAMICALLY)
            final_params = copy.deepcopy(self.base_params)
            
            
            # Loop through the search space to know which parameters to update
            for param_config in self.search_space:
                param_name = param_config['name']
                param_path = param_config['path']
                
                # Check if the parameter was found in the optimization results
                if param_name in best_params_dict:
                    best_value = best_params_dict[param_name]
                    # Use the helper function to set the new value in the params object
                    set_nested_value(final_params, param_path, best_value)


            # 4. Run one final backtest on the out-of-sample test data
            df_signals , oos_trades, _ , _ , _, _ = run_single_backtest(
                df=test_df, params=final_params, strategy_name=self.strategy_name, initial_capital=self.initial_capital, print_summary=True,
                test = True
            )

            all_oos_trades.append(oos_trades)
            all_oos_signals.append(df_signals)
            
            best_params_per_window[len(best_params_per_window)] = best_params_dict
            
        

        # 5. Stitch all the out-of-sample trade lists together
        final_walk_forward_trades = pd.concat(all_oos_trades, ignore_index=True)
        final_walk_forward_signals = pd.concat(all_oos_signals, ignore_index=True)
        return  final_walk_forward_signals, final_walk_forward_trades, last_study, best_params_per_window


    def get_strategy_details(self):
            """
            Runs a backtest with a specific set of parameters on a data slice
            and returns the strategy_info and backtest_summary.
            """
            
            # Your run_single_backtest function returns 6 items

            print(self.base_params.strategy_configs[self.strategy_name])
            print(self.base_params.strategy_configs[self.strategy_name].core_signal_params)
            print(self.base_params.strategy_configs[self.strategy_name].risk_profile)
            print(str(self.base_params.strategy_configs[self.strategy_name].filters).split("("))

            
            # return strategy_info