
import torch

from songdo_stgcn.script import utility
from songdo_stgcn.main import get_parameters, test, data_preparate, prepare_model

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.load_state_dict(torch.load("./outputs/STGCN_" + args.dataset + ".pt"))
    model.eval()

    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

if __name__ == "__main__":
    args, device, blocks = get_parameters()
    
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex, device)
    
    test(zscore, loss, model, test_iter, args)