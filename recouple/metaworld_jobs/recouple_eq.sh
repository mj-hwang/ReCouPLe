# reward learning from preferences + reasons
python scripts/train.py --config metaworld_configs/recouple_eq/recouple_eq_1.yaml --path ./logs/metaworld-debug
python scripts/train.py --config metaworld_configs/recouple_eq/recouple_eq_2.yaml --path ./logs/metaworld-debug
python scripts/train.py --config metaworld_configs/recouple_eq/recouple_eq_3.yaml --path ./logs/metaworld-debug

# offline RL for poliy learning
python scripts/train.py --config metaworld_configs/iql_recouple_eq/iql_pick_place_1.yaml --path ./logs/metaworld-debug
python scripts/train.py --config metaworld_configs/iql_recouple_eq/iql_pick_place_2.yaml --path ./logs/metaworld-debug
python scripts/train.py --config metaworld_configs/iql_recouple_eq/iql_pick_place_3.yaml --path ./logs/metaworld-debug
