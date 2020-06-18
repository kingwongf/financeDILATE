import numpy as np
import torch
import pandas as pd
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset, CustomDataset, CustomDataset2d
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU, MV_LSTM
from sklearn.preprocessing import StandardScaler
from loss.dilate_loss import dilate_loss
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
from nn_soft_dtw import SoftDTW
import warnings
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


def visualise_test(m, nets, nets_name, batch_size):
    gen_test = iter(m.testloader)
    losses_mse = []
    losses_dtw = []
    losses_tdi = []

    for ind, (test_inputs, test_targets) in enumerate(gen_test):
        print(ind)
        test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(m.device)
        test_targets = torch.tensor(test_targets, dtype=torch.float32).to(m.device)
        criterion = torch.nn.MSELoss()
        print(f"test_inputs tensor shape: {test_inputs.size()}, "
              f"test_targets shape: {test_targets.size()}, ")

        fig, axs = plt.subplots(1, 3, sharey='col', figsize=(15, 8))
        for net_i, net in enumerate(nets):
            test_preds = net(test_inputs).to(m.device)

            loss_mse = criterion(test_targets, test_preds)

            # DTW and TDI
            loss_dtw, loss_tdi = 0, 0
            for k in range(batch_size):
                target_k_cpu = test_targets[k, :, 0:1].view(-1).detach().cpu().numpy()
                output_k_cpu = test_preds[k, :, 0:1].view(-1).detach().cpu().numpy()

                loss_dtw += dtw(target_k_cpu, output_k_cpu)
                path, sim = dtw_path(target_k_cpu, output_k_cpu)

                Dist = 0
                for i, j in path:
                    Dist += (i - j) * (i - j)
                loss_tdi += Dist / (m.N_output * m.N_output)

            loss_dtw = loss_dtw / batch_size
            loss_tdi = loss_tdi / batch_size

            # print statistics
            losses_mse.append(loss_mse.item())
            losses_dtw.append(loss_dtw)
            losses_tdi.append(loss_tdi)

            input = test_inputs.detach().cpu().numpy()[0, :, :]
            target = test_targets.detach().cpu().numpy()[0, :, :]
            preds = test_preds.detach().cpu().numpy()[0, :, :]

            print(f"input np shape: {input.shape}, target np shape: {target.shape}, preds np shape: {preds.shape}")

            ## select target column in input
            input = input[:, m.idx_tgt_col]

            ## Get Scaling Mean and Vol
            input_mean, target_mean = m.mean_test_input[ind, :, m.idx_tgt_col], m.mean_test_target[ind, :]
            input_vol, target_vol = m.vol_test_input[ind, :, m.idx_tgt_col], m.vol_test_target[ind, :]

            ## Scaling back to original

            print(f"target shape: {target.shape}, target_vol shape: {target_vol.shape}, target_mean shape:{target_mean.shape}")
            input = input * input_vol + input_mean
            target = target.flatten() * target_vol + target_mean
            preds = preds.flatten() * target_vol + target_mean

            print(f"input shape: {input.shape}, target shape: {target.shape}, preds shape: {preds.shape}")

            print(f"target plot shape: {np.concatenate([input[-1:], target.ravel()]).shape}"

                  f"preds plot shape: {np.concatenate([input[-1:], preds.ravel()]).shape}")

            print(f"net_i: {net_i}")
            axs[net_i].plot(range(0, m.N_input), input, label='input', linewidth=1)

            axs[net_i].plot(range(m.N_input - 1, m.N_input + m.N_output),
                            np.concatenate([input[-1:], target.ravel()]),
                            label='target', linewidth=1)

            axs[net_i].plot(range(m.N_input - 1, m.N_input + m.N_output),
                            np.concatenate([input[-1:], preds.ravel()]),
                            label='prediction', linewidth=1)
            # axs[i].xticks(range(0,40,2))
            axs[net_i].legend()
            axs[net_i].set_title(
                f"{nets_name[net_i]} \n MSE: {round(loss_mse.item(), 3)}, DTW: {round(loss_dtw, 3)}. TDI: {round(loss_tdi, 3)}")

        # plt.show()
        plt.tight_layout()
        plt.savefig(f"results/new_test_loader/log/{ind}.png")
        plt.close()

    print(' Test mse= ', np.array(losses_mse).mean(), ' dtw= ', np.array(losses_dtw).mean(), ' tdi= ',
          np.array(losses_tdi).mean())
