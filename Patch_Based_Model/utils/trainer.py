import random
import wandb
import torch as t
import numpy as np
import pandas as pd
from time import time
from utils.loss import f1_loss
from torcheval.metrics.functional import binary_accuracy, binary_auroc

# Set a random seed for reproducibility
random.seed(42)

class Trainer:

    def __init__(self, model, crit1=None, crit2=None, crit3=None,
                       scheduler=None,
                       optim=None, train_data=None, 
                       val_data=None, cuda=True, log_wandb=True):
        """
        Initialize the trainer for model training and evaluation.

        Args:
            model: The neural network model to train.
            crit1: The primary loss criterion.
            crit2: The secondary loss criterion (optional).
            crit3: The tertiary loss criterion (optional).
            scheduler: The learning rate scheduler (optional).
            optim: The optimizer for model parameters.
            train_data: The training data loader.
            val_data: The validation data loader.
            cuda: Boolean indicating whether to use CUDA for GPU acceleration.
            log_wandb: Boolean indicating whether to log metrics with Weights and Biases (wandb).
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

        # Move model and criteria to GPU if enabled
        if cuda:
            self._model = model.cuda()
            self._crit1 = crit1.cuda()
            if self._crit2:
                self._crit2 = crit2.cuda()
            if self._crit3:
                self._crit3 = crit3.cuda()

    def save_checkpoint(self, epoch, path):
        """
        Save a training checkpoint.

        Args:
            epoch: The current training epoch.
            path: The directory path to save the checkpoint.
        """
        state = {'epoch': epoch + 1, 
                 'state_dict': self._model.state_dict(), 
                 'optimizer': self._optim.state_dict(), 
                 }

        t.save(state, path + '/checkpoint_'+ str(epoch) + '.ckp')

    def restore_checkpoint(self, epoch_n, path):
        """
        Restore a training checkpoint.

        Args:
            epoch_n: The epoch number to restore.
            path: The directory path containing the checkpoint.
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
            x: Input data.
            y: Target data.
            params: Additional parameters.
            lamda: A regularization parameter.

        Returns:
            loss: The total loss.
            loss_relapse: The loss for relapse prediction.
            loss_meta: The loss for meta prediction.
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
            x: Input data.
            y: Target data.
            params: Additional parameters.
            lamda: A regularization parameter.

        Returns:
            loss: The total validation loss.
            loss_relapse: The validation loss for relapse prediction.
            loss_meta: The validation loss for meta prediction.
            accuracy: The accuracy for relapse prediction.
            f1: The F1 score for relapse prediction.
            auc_roc: The ROC AUC score for relapse prediction.
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
                    checkpoint_path='./checkpoints', 
                    history_path='./history/history',
                    lamda=1):
        """
        Train the model.

        Args:
            start_epoch: The starting training epoch.
            end_epoch: The ending training epoch.
            checkpoint_interval: The interval at which to save checkpoints.
            checkpoint_path: The directory path to save checkpoints.
            history_path: The directory path to save training history.
            lamda: A regularization parameter.
        Returns:
            loss_array: An array of training and validation losses.
            avg_val_loss_relapse: The average validation loss for relapse prediction.
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

            loss_array[epoch - start_epoch, 0] = float(avg_training_loss)
            loss_array[epoch - start_epoch, 1] = float(avg_training_loss_relapse)
            loss_array[epoch - start_epoch, 2] = float(avg_training_loss_meta)

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

                loss_array[epoch - start_epoch, 3] = float(avg_val_loss)
                loss_array[epoch - start_epoch, 4] = float(avg_val_loss_relapse)
                loss_array[epoch - start_epoch, 5] = float(avg_val_loss_meta)
                loss_array[epoch - start_epoch, 6] = float(avg_accuracy)
                loss_array[epoch - start_epoch, 7] = float(avg_f1score)
                loss_array[epoch - start_epoch, 7] = float(avg_aucroc)

                if self._scheduler:
                    self._scheduler.step(avg_val_loss)

            toc = time()
            print(f'Epoch: {epoch} | Steps: {len(self._train_data)} | Training losses total: {round(avg_training_loss, 4)} | relapse: {round(avg_training_loss_relapse, 4)} | meta: {round(avg_training_loss_meta, 4)}| time_taken: {toc - tic} s')
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
