call conda activate fs_env
call python utils/dafsl_datapreprocess.py configs/dafsl_exp_0.json
call python utils/find_stats_dataset.py configs/dafsl_exp_0.json