import random
import wandb
import torch as t
import numpy as np
import pandas as pd
from time import time
from utils.loss import f1_loss
from torcheval.metrics.functional import binary_accuracy, binary_auroc

random.seed(42)

class Trainer:
    def __init__(self, model, crit1=None, crit2=None, crit3=None,
                 scheduler=None, optim=None, train_data=None, 
                 val_data=None, cuda=True, log_wandb=True):
        """
        A class for training and evaluating a deep learning model.

        Args:
            model (torch.nn.Module): The neural network model to train.
            crit1 (torch.nn.Module): The first loss criterion.
            crit2 (torch.nn.Module): The second loss criterion.
            crit3 (torch.nn.Module): The third loss criterion.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            optim (torch.optim.Optimizer): Optimizer for model parameters.
            train_data (torch.utils.data.Dataset): Training data.
            val_data (torch.utils.data.Dataset): Validation data.
            cuda (bool): Whether to use GPU (if available).
            log_wandb (bool): Whether to log training progress to Weights and Biases.

        """
        self._model = model
        self._crit1 = crit1
        self._crit2 = crit2
        self._crit3 = crit3
        self._optim = optim
        self._scheduler = scheduler
        self._train_data = train_data
        self._val_data = val_data
        self._cuda = cuda
        self._log_wandb = log_wandb

        if cuda:
            self._model = model.cuda()
            self._crit1 = crit1.cuda()
            if self._crit2:
                self._crit2 = crit2.cuda()
            if self._crit3:
                self._crit3 = crit3.cuda()

    def save_checkpoint(self, epoch, path):
        """
        Save a checkpoint of the model and optimizer.

        Args:
            epoch (int): Current training epoch.
            path (str): Path to save the checkpoint.
        """
        state = {'epoch': epoch + 1, 
                 'state_dict': self._model.state_dict(), 
                 'optimizer': self._optim.state_dict(), 
                 }
        t.save(state, path + '/checkpoint_'+ str(epoch) + '.ckp')

    def restore_checkpoint(self, epoch_n, path):
        """
        Restore a checkpoint of the model and optimizer.

        Args:
            epoch_n (int): Epoch to restore.
            path (str): Path to load the checkpoint from.
        """
        ckp = t.load(path + '/checkpoint_' + str(epoch_n)+ '.ckp')
        self._model.load_state_dict(ckp['state_dict'])
        if self._cuda:
            device = t.device("cuda")
            self._model.to(device)
        self._optim.load_state_dict(ckp['optimizer'])   

    def train_step(self, x, y, params, lamda):
        """
        Perform a single training step.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.
            params (torch.Tensor): Model parameters.
            lamda (float): Hyperparameter for meta-loss weight.

        Returns:
            torch.Tensor: Total loss.
            torch.Tensor: Relapse loss.
            torch.Tensor: Meta-loss.
        """
        y_hat = self._model(x, params)
        y_relapse = y[:, 0][:, None]
        y_breslow = y[:, 1].type(t.LongTensor).cuda()
        y_ulceration = y[:, 2][:, None]

        loss_relapse = self._crit1(y_hat[0], y_relapse)
        loss_breslow = self._crit2(y_hat[1], y_breslow)
        loss_ulceration = self._crit3(y_hat[2], y_ulceration)
        
        meta_loss = loss_breslow + loss_ulceration
        loss = loss_relapse + lamda * meta_loss
  
        loss.backward()
        self._optim.step()
        self._optim.zero_grad(set_to_none=True)
        return loss, loss_relapse, meta_loss

    def val_step(self, x, y, params, lamda):
        """
        Perform a single validation step.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.
            params (torch.Tensor): Model parameters.
            lamda (float): Hyperparameter for meta-loss weight.

        Returns:
            torch.Tensor: Total loss.
            torch.Tensor: Relapse loss.
            torch.Tensor: Meta-loss.
            float: Accuracy.
            float: F1 score.
            float: ROC AUC.
        """
        with t.no_grad():
            y_hat = self._model(x, params)

        y_relapse = y[:, 0][:, None]
        y_breslow = y[:, 1].type(t.LongTensor).cuda()
        y_ulceration = y[:, 2][:, None]

        loss_relapse = self._crit3(y_hat[0], y_relapse).item()
        loss_breslow = self._crit2(y_hat[1], y_breslow).item()
        loss_ulceration = self._crit3(y_hat[2], y_ulceration).item()
        
        accuracy = binary_accuracy(y_hat[0][:, 0], y_relapse[:, 0], threshold=0.5)
        auc_roc = binary_auroc(y_hat[0][:, 0], y_relapse[:, 0])

        predictions = y_hat[0][:, 0] > 0.5
        f1 = f1_loss(y_relapse[:, 0], predictions.type(t.float32))
        meta_loss = loss_breslow + loss_ulceration

        loss = loss_relapse + lamda * meta_loss
        return loss, loss_relapse, meta_loss, accuracy, f1, auc_roc

    def train(self, start_epoch=0, end_epoch=100, checkpoint_interval=20, 
              checkpoint_path='./checkpoints', history_path='./history/history', lamda=1):
        """
        Train the model.

        Args:
            start_epoch (int): The starting training epoch.
            end_epoch (int): The ending training epoch.
            checkpoint_interval (int): Interval for saving checkpoints.
            checkpoint_path (str): Path to save checkpoints.
            history_path (str): Path to save training history.
            lamda (float): Hyperparameter for meta-loss weight.

        Returns:
            np.ndarray: Training loss history.
            float: Validation relapse loss of the last epoch.
        """
        epochs = end_epoch - start_epoch

        if start_epoch != 0:
            self.restore_checkpoint(start_epoch, checkpoint_path)

        loss_array = np.zeros((epochs, 3))

        if self._val_data is not None:
            loss_array = np.zeros((epochs, 9))

        device = next(self._model.parameters()).device
        
        for epoch in range(start_epoch, end_epoch):

            total_training_loss = 0
            total_training_loss_relapse = 0
            total_training_loss_meta = 0
            tic = time()
            self._model.train()

            for img, parameters, target in self._train_data:
                
                img = img.to(device)
                target = target.to(device)
                parameters = parameters.to(device)

                loss, loss_relapse, loss_meta = self.train_step(img, target, parameters, lamda) 
                 
                total_training_loss += loss.item()
                total_training_loss_relapse += loss_relapse.item()
                total_training_loss_meta += loss_meta.item()

            avg_training_loss = total_training_loss / len(self._train_data) 
            avg_training_loss_relapse = total_training_loss_relapse / len(self._train_data)
            avg_training_loss_meta = total_training_loss_meta / len(self._train_data)
            loss_array[epoch-start_epoch, 0] = float(avg_training_loss)
            loss_array[epoch-start_epoch, 1] = float(avg_training_loss_relapse)
            loss_array[epoch-start_epoch, 2] = float(avg_training_loss_meta)

            if self._val_data is not None:
                self._model.eval()
                total_val_loss = 0
                total_val_loss_relapse = 0
                total_val_loss_meta = 0
                total_accuracy = 0
                total_f1score = 0
                total_aucroc = 0

                for img, parameters, target in self._val_data:
        
                    img = img.to(device)
                    target = target.to(device)
                    parameters = parameters.to(device)

                    loss, loss_relapse, loss_meta, accuracy, f1, auc_roc = self.val_step(img, target, parameters, lamda)

                    total_val_loss += loss
                    total_val_loss_relapse += loss_relapse
                    total_val_loss_meta += loss_meta
                    total_accuracy += accuracy
                    total_f1score += f1
                    total_aucroc += auc_roc

                avg_val_loss = total_val_loss / len(self._val_data)
                avg_val_loss_relapse = total_val_loss_relapse / len(self._val_data)
                avg_val_loss_meta = total_val_loss_meta / len(self._val_data)
                avg_accuracy = total_accuracy / len(self._val_data)
                avg_f1score = total_f1score / len(self._val_data)
                avg_aucroc = total_aucroc / len(self._val_data)

                loss_array[epoch-start_epoch, 3] = float(avg_val_loss)
                loss_array[epoch-start_epoch, 4] = float(avg_val_loss_relapse)
                loss_array[epoch-start_epoch, 5] = float(avg_val_loss_meta)
                loss_array[epoch-start_epoch, 6] = float(avg_accuracy)
                loss_array[epoch-start_epoch, 7] = float(avg_f1score)
                loss_array[epoch-start_epoch, 8] = float(avg_aucroc)

                if self._scheduler:
                    self._scheduler.step(avg_val_loss)

            toc = time()
            print(f'Epoch: {epoch} | Steps: {len(self._train_data)} | Training losses total: {round(avg_training_loss, 4)} | relapse: {round(avg_training_loss_relapse, 4)} | meta: {round(avg_training_loss_meta, 4)} | time_taken: {toc - tic} s')
            print(f'Validation loss total: {round(avg_val_loss, 4)} | relapse: {round(avg_val_loss_relapse, 4)} | meta: {round(avg_val_loss_meta, 4)} | relapse accuracy: {round(float(avg_accuracy), 4)} | f1_score: {round(float(avg_f1score), 4)} | roc_auc: {round(float(avg_aucroc), 4)}')
            
            if self._log_wandb:
                wandb.log({
                'training_loss': round(avg_training_loss, 4),
                'training_relapse': round(avg_training_loss_relapse, 4),
                'training_meta': round(avg_training_loss_meta, 4),
                'val_loss': round(avg_val_loss, 4),
                'val_relapse': round(avg_val_loss_relapse, 4),
                'val_meta': round(avg_val_loss_meta, 4),
                'val_accuracy': round(float(avg_accuracy), 4),
                'val_f1score': round(float(avg_f1score), 4)
                })

            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch, checkpoint_path)
                
            loss_df = pd.DataFrame(loss_array, columns=['TrainLoss', 'LossRelapse', 'LossMeta', 'ValLoss', 'ValLossRelapse', 'ValLossMeta', 'ValAcc', 'ValF1', 'ValAucRuc'])
            loss_df.to_csv(history_path + '_' + str(start_epoch) + '.csv', index=False)
        if self._log_wandb:
            wandb.save()
        return loss_array, avg_val_loss_relapse
