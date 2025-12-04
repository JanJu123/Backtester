import json
from dacite import from_dict, Config
import os
import glob



from config.param_types_v1 import Params



def load_params_from_json(file_path: str) -> Params:
    """
    Loads main config and automatically merges all strategy JSON files 
    from 'config/strategies/' into the 'strategy_configs' dictionary.
    """
    
    # 1. Load the Main JSON (System settings like trading_costs)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Main config not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    # -----------------------------------------------------------
    # CRITICAL: Ensure the structure matches the Params dataclass
    # The dataclass expects: 
    # { 
    #    "strategy_configs": { ... }, 
    #    "trading_cost_params": { ... } 
    # }
    # -----------------------------------------------------------
    if "strategy_configs" not in data:
        data["strategy_configs"] = {}

    # 2. Locate the 'strategies' subfolder inside 'config'
    base_dir = os.path.dirname(os.path.abspath(file_path))
    strategies_dir = os.path.join(base_dir, "strategies")
    
    # 3. Dynamic Merge: Scan for all .json files in config/strategies/
    if os.path.exists(strategies_dir):
        strategy_files = glob.glob(os.path.join(strategies_dir, "*.json"))
        
        print(f"📂 Loading {len(strategy_files)} strategies from {strategies_dir}...")

        for strat_file in strategy_files:
            try:
                with open(strat_file, 'r') as f:
                    strat_data = json.load(f)
                    # Merge this file's content into the main 'strategy_configs' dictionary
                    data["strategy_configs"].update(strat_data)
            except Exception as e:
                print(f"⚠️ Warning: Could not load {os.path.basename(strat_file)}: {e}")
    else:
        print(f"⚠️ No 'config/strategies/' folder found. Using only main params.")

    # 4. Convert to Python Object
    # strict=True ensures that if your JSON has a typo (a field not in dataclass), it errors out early.
    config = Config(strict=True) 
    
    try:
        # Now 'data' has the correct structure: { strategy_configs: {...}, trading_cost_params: {...} }
        params = from_dict(data_class=Params, data=data, config=config)
    except Exception as e:
        print(f"❌ DATA STRUCTURE ERROR: Your merged JSON does not match the Params class.\nError: {e}")
        # Debugging tip: Uncomment the next line to see what 'data' looks like if it fails
        # import pprint; pprint.pprint(data)
        raise e

    return params

# Example usage:
if __name__ == "__main__":
    params = load_params_from_json(r"config\params_v2.json")
    # print(params)
    print(params.strategy_configs["Bollinger_Bands_mean_reversion_V1"].core_signal_params)
