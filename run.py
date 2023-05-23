import argparse
import subprocess
import numpy as np

parser = argparse.ArgumentParser(description="run")
parser.add_argument("--device", type=str)
parser.add_argument("--run", type=int)

args = parser.parse_args()

# Citation datasets
if args.run == 0: 
    data = 'Cora' # 'Cora', 'Citeseer', 'Pubmed'
    gnn = 'GAT'
    
    # over_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    # lcc_list = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    over_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 1]
    under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    lcc_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for lcc in lcc_list:
        for under in under_list:
            for over in over_list:
                if over in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] and under in [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4]  and lcc in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
                    continue
                subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_l_{lcc}_o{over}u{under} --model {gnn} --calibration 'LCCT' --wdecay 5e-4 --b_over {over} --b_under {under} --default_l {lcc} --verbose --wandb", shell=True)
    
elif args.run == 1: 
    data = 'Citeseer' # 'Cora', 'Citeseer', 'Pubmed'
    gnn = 'GAT'
    
    # over_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    # lcc_list = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    over_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 1]
    under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    lcc_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for lcc in lcc_list:
        for under in under_list:
            for over in over_list:
                if over in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] and under in [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4]  and lcc in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
                    continue
                subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_l_{lcc}_o{over}u{under} --model {gnn} --calibration 'LCCT' --wdecay 5e-4 --b_over {over} --b_under {under} --default_l {lcc} --verbose --wandb", shell=True)
    
elif args.run == 2: 
    data = 'Pubmed' # 'Cora', 'Citeseer', 'Pubmed'
    gnn = 'GAT'
    
    # over_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    # lcc_list = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    over_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 1]
    under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    lcc_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for lcc in lcc_list:
        for under in under_list:
            for over in over_list:
                if over in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] and under in [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4]  and lcc in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
                    continue
                subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_l_{lcc}_o{over}u{under} --model {gnn} --calibration 'LCCT' --wdecay 5e-4 --b_over {over} --b_under {under} --default_l {lcc} --verbose --wandb", shell=True)
    
# Imbalance datasets
elif args.run == 3: 
    data = 'Computers' # 'Computers', 'Photo', 'CS', 'Physics'
    gnn = 'GAT'
    
    # over_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    # lcc_list = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

    over_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 1]
    under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    lcc_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for lcc in lcc_list:
        for under in under_list:
            for over in over_list:
                if over in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] and under in [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4]  and lcc in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
                    continue
                subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_l_{lcc}_o{over}u{under} --model {gnn} --calibration 'LCCT' --wdecay 0 --b_over {over} --b_under {under} --default_l {lcc} --verbose --wandb", shell=True)

elif args.run == 4: 
    data = 'Photo' # 'Computers', 'Photo', 'CS', 'Physics'
    gnn = 'GAT'
    
    # # over_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # # under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    # lcc_list = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

    over_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 1]
    under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    lcc_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for lcc in lcc_list:
        for under in under_list:
            for over in over_list:
                if over in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] and under in [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4]  and lcc in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
                    continue
                subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_l_{lcc}_o{over}u{under} --model {gnn} --calibration 'LCCT' --wdecay 0 --b_over {over} --b_under {under} --default_l {lcc} --verbose --wandb", shell=True)
    
elif args.run == 5: 
    data = 'CS' # 'Computers', 'Photo', 'CS', 'Physics'
    gnn = 'GAT'
    
    # over_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    # under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    # lcc_list = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    over_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 1]
    under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    lcc_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for lcc in lcc_list:
        for under in under_list:
            for over in over_list:
                if over in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] and under in [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4]  and lcc in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
                    continue
                subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_l_{lcc}_o{over}u{under} --model {gnn} --calibration 'LCCT' --wdecay 0 --b_over {over} --b_under {under} --default_l {lcc} --verbose --wandb", shell=True)
 
elif args.run == 6: 
    data = 'Physics' # 'Computers', 'Photo', 'CS', 'Physics'
    gnn = 'GCN'
    
    over_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    lcc_list = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    # over_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 1]
    # under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    # lcc_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for lcc in lcc_list:
        for under in under_list:
            for over in over_list:
                # if over in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] and under in [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4]  and lcc in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
                #     continue
                subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_l_{lcc}_o{over}u{under} --model {gnn} --calibration 'LCCT' --wdecay 0 --b_over {over} --b_under {under} --default_l {lcc} --verbose --wandb", shell=True)

# Large-scale dataset
elif args.run == 7: 
    data = 'CoraFull'
    gnn = 'GCN'
    
    over_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    lcc_list = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    # over_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 1]
    # under_list = [0, 0.02, -0.02, 0.2, -0.2, 0.4, -0.4] 
    # lcc_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
    for lcc in lcc_list:
        for under in under_list:
            for over in over_list:
                subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_l_{lcc}_o{over}u{under} --model {gnn} --calibration 'LCCT' --wdecay 0 --b_over {over} --b_under {under} --default_l {lcc} --verbose --wandb", shell=True)

elif args.run == 8:
    gnn = 'GCN'
    over = {'Cora': 0.1, 'Citeseer': 0.1, 'Pubmed': 0.08, 'Computers': 0.1, 'Photo': 0.1, 'Physics': 0.12, 'CS': 0.08, 'CoraFull': 0.35}
    under = {'Cora': -0.02, 'Citeseer': 0.2, 'Pubmed': 0, 'Computers': 0.02, 'Photo': 0.02, 'Physics': 0, 'CS': 0.02, 'CoraFull': 0.4}
    lcc = {'Cora': 0.45, 'Citeseer': 0.4, 'Pubmed': 0.4, 'Computers': 0.85, 'Photo': 0.4, 'Physics': 0.85, 'CS': 0.35, 'CoraFull': 0.5}
    for data in ['Citeseer']: #['Cora', 'Computers', 'CoraFull']:
        if data in ['Cora', 'Citeseer', 'Pubmed']:
            wd = 5e-4
        else:
            wd = 0
        subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_store_logits --model {gnn} --calibration 'LCCT' --wdecay {wd} --b_over {over[data]} --b_under {under[data]} --default_l {lcc[data]} --verbose --store_logits", shell=True)

    gnn = 'GAT'
    over = {'Cora': 0.1, 'Citeseer': 0.1, 'Pubmed': 0.08, 'Computers': 0.1, 'Photo': 0.08, 'Physics': 0.12, 'CS': 0.1, 'CoraFull': 0.4}
    under = {'Cora': 0, 'Citeseer': 0.2, 'Pubmed': -0.02, 'Computers': 0.2, 'Photo': -0.02, 'Physics': 0.02, 'CS': 0.02, 'CoraFull': 0.4}
    lcc = {'Cora': 0.35, 'Citeseer': 0.45, 'Pubmed': 0.35, 'Computers': 0.35, 'Photo': 0.4, 'Physics': 0.85, 'CS': 0.7, 'CoraFull': 0.35}
    for data in ['Citeseer']: #['Cora', 'Computers', 'CoraFull']:
        if data in ['Cora', 'Citeseer', 'Pubmed']:
            wd = 5e-4
        else:
            wd = 0
        subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_store_logits --model {gnn} --calibration 'LCCT' --wdecay {wd} --b_over {over[data]} --b_under {under[data]} --default_l {lcc[data]} --verbose --store_logits", shell=True) #  --store_logits
    
elif args.run == 9:
    gnn = 'GCN'
    over = {'Cora': 0.1, 'Citeseer': 0.1, 'Pubmed': 0.1, 'Computers': 0.1, 'Photo': 0.1, 'Physics': 0.2, 'CS': 0.1, 'CoraFull': 0.35}
    under = {'Cora': -0.02, 'Citeseer': 0.2, 'Pubmed': 0, 'Computers': 0.02, 'Photo': 0.02, 'Physics': 0, 'CS': 0.02, 'CoraFull': 0.4}
    lcc = {'Cora': 0.45, 'Citeseer': 0.4, 'Pubmed': 0.4, 'Computers': 0.85, 'Photo': 0.4, 'Physics': 0.8, 'CS': 0.35, 'CoraFull': 0.5}
    for data in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CS', 'Physics', 'CoraFull']:
        if data in ['Cora', 'Citeseer', 'Pubmed']:
            wd = 5e-4
        else:
            wd = 0
        subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_nll --model {gnn} --calibration 'LCCT' --wdecay {wd} --b_over {over[data]} --b_under {under[data]} --default_l {lcc[data]} --wandb --verbose", shell=True)

    gnn = 'GAT'
    over = {'Cora': 0.1, 'Citeseer': 0.1, 'Pubmed': 0.1, 'Computers': 0.1, 'Photo': 0.1, 'Physics': 0.25, 'CS': 0.1, 'CoraFull': 0.4}
    under = {'Cora': 0, 'Citeseer': 0.2, 'Pubmed': -0.02, 'Computers': 0.2, 'Photo': 0, 'Physics': 0.02, 'CS': 0.02, 'CoraFull': 0.4}
    lcc = {'Cora': 0.35, 'Citeseer': 0.45, 'Pubmed': 0.35, 'Computers': 0.35, 'Photo': 0.35, 'Physics': 0.85, 'CS': 0.7, 'CoraFull': 0.35}
    for data in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CS', 'Physics', 'CoraFull']:
        if data in ['Cora', 'Citeseer', 'Pubmed']:
            wd = 5e-4
        else:
            wd = 0
        subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python src/main_calibration_exp8.py --dataset {data} --exp_name {data}_{gnn}_nll2 --model {gnn} --calibration 'LCCT' --wdecay {wd} --b_over {over[data]} --b_under {under[data]} --default_l {lcc[data]} --wandb --verbose", shell=True) #  --store_logits
    
            