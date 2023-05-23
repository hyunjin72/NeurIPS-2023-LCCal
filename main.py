import abc
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
import os
import gc
from pathlib import Path
from data_utils.data_utils import load_data, load_node_to_nearest_training
from models import create_model
from calibrator import LCCal
from metrics import \
    NodewiseECE, NodewiseBrier, NodewiseNLL, Reliability, NodewiseKDE, \
    NodewiswClassECE, ECE
from utils import \
    set_global_seeds, name_model, create_nested_defaultdict, \
    metric_mean, metric_std

from torch_sparse import SparseTensor
import numpy as np
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_START_METHOD"] = "thread" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Metrics(metaclass=abc.ABCMeta):
    """
    Code adopted from 
    Hsu et al., "What Makes Graph Neural Networks Miscalibrated?" (NeurIPS'22)
    """
    @abc.abstractmethod
    def acc(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def nll(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def brier(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def ece(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def reliability(self) -> Reliability:
        raise NotImplementedError

    @abc.abstractmethod
    def kde(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def cls_ece(self) -> float:
        raise NotImplementedError

class NodewiseMetrics(Metrics):
    """
    Code adopted from 
    Hsu et al., "What Makes Graph Neural Networks Miscalibrated?" (NeurIPS'22)
    """
    def __init__(
            self, logits: Tensor, gts: LongTensor, index: LongTensor,
            bins: int = 15, scheme: str = 'equal_width', norm=1):
        self.node_index = index
        self.logits = logits
        self.gts = gts
        self.nll_fn = NodewiseNLL(index)
        self.brier_fn = NodewiseBrier(index)
        self.ece_fn = NodewiseECE(index, bins, scheme, norm)
        self.kde_fn = NodewiseKDE(index, norm)
        self.cls_ece_fn = NodewiswClassECE(index, bins, scheme, norm)

    def acc(self) -> float:
        preds = torch.argmax(self.logits, dim=1)[self.node_index]
        return torch.mean(
            (preds == self.gts[self.node_index]).to(torch.get_default_dtype())
        ).item()

    def nll(self) -> float:
        return self.nll_fn(self.logits, self.gts).item()

    def brier(self) -> float:
        return self.brier_fn(self.logits, self.gts).item()

    def ece(self) -> float:
        return self.ece_fn(self.logits, self.gts).item()

    def reliability(self) -> Reliability:
        return self.ece_fn.get_reliability(self.logits, self.gts)
    
    def kde(self) -> float:
        return self.kde_fn(self.logits, self.gts).item()

    def cls_ece(self) -> float:
        return self.cls_ece_fn(self.logits, self.gts).item()


def eval(data, prob, mask_name):
    if mask_name == 'Train':
        mask = data.train_mask
    elif mask_name == 'Val':
        mask = data.val_mask
    elif mask_name == 'Test':
        mask = data.test_mask
    else:
        raise ValueError("Invalid mask_name")
    eval_result = {}
    eval = NodewiseMetrics(prob, data.y, mask)
    acc, nll, brier, ece, kde, cls_ece = eval.acc(), eval.nll(), \
                                eval.brier(), eval.ece(), eval.kde(), eval.cls_ece()
    eval_result.update({'acc':acc,
                        'nll':nll,
                        'bs':brier,
                        'ece':ece,
                        'kde':kde,
                        'cls_ece': cls_ece})
    reliability = eval.reliability()
    del eval
    gc.collect()
    return eval_result, reliability


def main(split, init, eval_type_list, args):
    print(f'############################################## [SPLIT {split} / INIT {init}] ##############################################')
    uncal_test_result = create_nested_defaultdict(eval_type_list)
    cal_val_result = create_nested_defaultdict(eval_type_list)
    cal_test_result = create_nested_defaultdict(eval_type_list)
    max_fold = int(args.split_type.split("_")[1].replace("f",""))

    for fold in range(max_fold):
        print(f'########## [FOLD {fold}] ##########')
        # 1. Load a dataset.
        dataset = load_data(args.dataset, args.split_type, split, fold)
        data = dataset.data.to(device)
        
        args.n_nodes = data.num_nodes
        args.n_classes = dataset.num_classes
        adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).to(device).t()
        adj_t = adj_t.to_symmetric()
        data.adj_t = adj_t

        # 2. Load trained GNN.
        model = create_model(dataset, args).to(device)
        model_name = name_model(fold, args)
        dir = Path(os.path.join('vanilla_models', args.dataset, args.split_type, 'split'+str(split), 'init'+ str(init)))
        file_name = dir / (model_name + '.pt')
        model.load_state_dict(torch.load(file_name))
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.edge_index)
            log_prob = F.log_softmax(logits, dim=1).detach()

        for eval_type in eval_type_list:
            eval_result, _ = eval(data, log_prob, 'Test')
            for metric in eval_result:
                uncal_test_result[eval_type][metric].append(eval_result[metric])
        torch.cuda.empty_cache()

        # 3. Obtain calibrated logits via LCCal.
        temp_model = LCCal(args)
        logits = temp_model.calibrate(data, logits)
        log_prob = F.log_softmax(logits, dim=1).detach()

        # 4. Evaluate calibrated logits.
        for eval_type in eval_type_list:
            eval_result, _ = eval(data, log_prob, 'Val') # Use validation sets to search hyperparameters
            for metric in eval_result:
                cal_val_result[eval_type][metric].append(eval_result[metric])
        
        for eval_type in eval_type_list:
            eval_result, _ = eval(data, log_prob, 'Test')
            for metric in eval_result:
                cal_test_result[eval_type][metric].append(eval_result[metric])
        print(f"Test Acc.: {eval_result['acc']*100:.2f}, Test NLL: {eval_result['nll']:.4f}, ",
              f"Test ECE: {eval_result['ece']*100:.2f}, Test Cls_ece: {eval_result['cls_ece']*100:.2f}")
        print('---')
        
        torch.cuda.empty_cache()
    return uncal_test_result, cal_val_result, cal_test_result


if __name__ == '__main__':
    # base arguments
    parser = argparse.ArgumentParser(description='train.py and calibration.py share the same arguments')
    parser.add_argument('--seed', type=int, default=10, help='Random Seed')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora','Citeseer', 'Pubmed', 
                        'Computers', 'Photo', 'CS', 'Physics', 'CoraFull'])
    parser.add_argument('--split_type', type=str, default='5_3f_85', help='k-fold and test split')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT'])
    parser.add_argument('--wdecay', type=float, default=5e-4, help='Weight decay for training phase')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate. 1.0 denotes drop all the weights to zero')
    parser.add_argument('--folds', type=int, default=3, help='K folds cross-validation for calibration')
    parser.add_argument('--ece-bins', type=int, default=15, help='number of bins for ece')
    parser.add_argument('--ece-scheme', type=str, default='equal_width', choices=ECE.binning_schemes, help='binning scheme for ece')
    parser.add_argument('--ece-norm', type=float, default=1.0, help='norm for ece')
     
    # LCCal 
    parser.add_argument('--eps', type=float, default=0.1, help='a small value to avoid zero temperature')
    parser.add_argument('--b_over', type=float, default=1, help='temperature controling factor for nodes expected to be overconfident')
    parser.add_argument('--b_under', type=float, default=1, help='temperature controling factor for nodes expected to be underconfident')
    parser.add_argument('--default_l', type=float, default=0.4, help='base LCC value to divide over/underconfident cases')
    parser.add_argument('--n_bins', type=int, default=15, help='#(confidence intervals)')
    parser.add_argument('--n_LCC_bins', type=int, default=5, help='#(LCC intervals)')
    
    args = parser.parse_args()
    print(args)
    
    set_global_seeds(args.seed)
    eval_type_list = ['Nodewise']
    max_splits,  max_init = 5, 5

    uncal_test_total = create_nested_defaultdict(eval_type_list)
    cal_val_total = create_nested_defaultdict(eval_type_list)
    cal_test_total = create_nested_defaultdict(eval_type_list)
    for split in range(max_splits):
        for init in range(max_init):
            print(split, init)
            (uncal_test_result,
             cal_val_result,
             cal_test_result) = main(split, init, eval_type_list, args)
            for eval_type, eval_metric in uncal_test_result.items():
                for metric in eval_metric:
                    uncal_test_total[eval_type][metric].extend(uncal_test_result[eval_type][metric])
                    cal_val_total[eval_type][metric].extend(cal_val_result[eval_type][metric])
                    cal_test_total[eval_type][metric].extend(cal_test_result[eval_type][metric])

    val_mean = metric_mean(cal_val_total['Nodewise'])
    test_mean = metric_mean(cal_test_total['Nodewise'])
    test_std = metric_std(cal_test_total['Nodewise'])

    # 5. Obtain averaged calibration performances.
    for name, result in zip(['Uncal', args.calibration], [uncal_test_total, cal_test_total]):
        print(name)
        for eval_type in eval_type_list:
            test_mean = metric_mean(result[eval_type])
            test_std = metric_std(result[eval_type])
            print(f"{eval_type:>8} Accuracy: &{test_mean['acc']:.2f}$\pm${test_std['acc']:.2f} \t" + \
                                f"NLL: &{test_mean['nll']:.4f}$\pm${test_std['nll']:.4f} \t" + \
                                f"Brier: &{test_mean['bs']:.4f}$\pm${test_std['bs']:.4f} \t" + \
                                f"ECE: &{test_mean['ece']:.2f}$\pm${test_std['ece']:.2f} \t" + \
                                f"Classwise-ECE: &{test_mean['cls_ece']:.2f}$\pm${test_std['cls_ece']:.2f} \t" + \
                                f"KDE: &{test_mean['kde']:.2f}$\pm${test_std['kde']:.2f}")