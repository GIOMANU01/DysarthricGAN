import os, sys
from config_dys import cfg
from dataset import UA_melspectrogram
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score

torch.manual_seed(184)


class  BinaryClassifier(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.init_dataset()
        ############################################################################################################

        #mobile net
        #self.model = torchvision.models.mobilenet_v3_small().to(cfg.device)
        #self.num_features = self.model.classifier[-1].in_features
        #self.model.classifier[-1] = torch.nn.Linear(self.num_features, 1).to(cfg.device)

        #resnet152
        #self.model = torchvision.models.resnet152().to(cfg.device)
        #self.in_features = self.model.fc.in_features
        #self.model.fc = nn.Linear(self.in_features, 1).to(cfg.device)
        
        #resnet50
        self.model = torchvision.models.resnet50().to(cfg.device)
        self.num_features = self.model.fc.in_features
        #self.model.fc = nn.Linear(self.num_features, 1).to(cfg.device) # eventualmente modificare la FC
        self.model.fc = nn.Sequential(
                nn.Linear(self.num_features, 256),
                nn.Dropout(p=0.5),
                nn.Linear(256, 1)).to(cfg.device)
        ############################################################################################################

        self.criterion = torch.nn.BCEWithLogitsLoss().to(cfg.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr) 
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.experiment_id))
        self.opt_scheduler = ExponentialLR(self.optimizer, gamma=0.96**(1/5))


    def init_dataset(self):
        df = pd.read_csv(self.cfg.meta_file)

        train_df=df[df['pat_ID'].isin(self.cfg.train_pat)]
        if cfg.augm == False:  # per il train prendo solo i campioni non sintetici o entrambi secondo la configurazione
            train_df=train_df[df['synth']==False]
        print(f'Loaded {len(train_df)} items for training')

        validation_df = df[df['pat_ID'].isin(self.cfg.val_pat)]
        validation_df=validation_df[df['synth']==False]  # per il validation prendo solo i campioni non sintetici
        print(f'Loaded {len(validation_df)} items for validation')

        test_df = df[df['pat_ID'].isin(self.cfg.test_pat)]
        test_df=test_df[df['synth']==False] # per il test prendo solo i campioni non sintetici
        print(f'Loaded {len(test_df)} items for test')

        train_set = UA_melspectrogram(train_df, self.cfg)
        val_set = UA_melspectrogram(validation_df, self.cfg)
        test_set = UA_melspectrogram(test_df, self.cfg)

        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.cfg.batch_size,
                                                        shuffle=True,
                                                        num_workers=0,
                                                        drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(val_set,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        drop_last=True)
    
    def train(self):
        self.best_accuracy = 0.0

        for e in range(1, self.cfg.epochs+1):
            self.model.train()
            cum_loss = 0.0
            print(f'Epoch {e}/{self.cfg.epochs}')

            for spec, label in tqdm(self.train_loader):
                spec = spec.to(self.cfg.device)
                label = label.float().to(self.cfg.device) # float?!?
                spec = spec.repeat(1, 3, 1, 1) # la rete richiede immagini a 3 canali quindi si ripete lo spettrogramma 3 volte
                label = label.unsqueeze(1)
                out = self.model(spec)
                loss = self.criterion(out, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                cum_loss += loss.item()
            self.opt_scheduler.step()
            cum_loss /= len(self.train_loader)

            # Validation
            val_accuracy = 0.0
            cum_loss_val = 0.0
            
            if len(self.cfg.val_pat) > 0:
                val_accuracy, cum_loss_val,  val_metrics  = self.validate()
                cum_loss_val /= len(self.val_loader)   
                val_accuracy /= len(self.val_loader)

            test_accuracy, cum_loss_test,  test_metrics  = self.test() # ESEGUO IL TEST AD OGNI EPOCA
            cum_loss_test /= len(self.test_loader)   
            test_accuracy /= len(self.test_loader)
            
            
            print(f'Train loss {cum_loss:.2f} - Val accuracy: {100*(val_accuracy):.2f} % - Val loss {cum_loss_val:.2f} % - test accuracy {100*(test_accuracy):.2f} % - test loss {cum_loss_test:.2f}')
            
            # log to tensorboard
            self.writer.add_scalar('TRAIN_LOSS', cum_loss, e)
            self.writer.add_scalar('LEARNING_RATE', self.optimizer.param_groups[0]["lr"], e)

            if len(self.cfg.val_pat) > 0:
                self.writer.add_scalar('VAL_ACCURACY', val_accuracy, e)
                self.writer.add_scalar('VAL_LOSS', cum_loss_val, e)
                self.writer.add_scalar('VAL/ACCURACY', val_metrics['Accuracy'], e)
                self.writer.add_scalar('VAL/MACRO_F1', val_metrics['macro F1-score'], e)
                self.writer.add_scalar('VAL/MICRO_F1', val_metrics['micro F1-score'], e)
                self.writer.add_scalar('VAL/WEIGHTED_F1', val_metrics['weighted F1-score'], e)
            
            self.writer.add_scalar('TEST_ACCURACY', test_accuracy, e)
            self.writer.add_scalar('TEST_LOSS', cum_loss_test, e)
            self.writer.add_scalar('TEST/ACCURACY', test_metrics['Accuracy'], e)
            self.writer.add_scalar('TEST/MACRO_F1', test_metrics['macro F1-score'], e)
            self.writer.add_scalar('TEST/MICRO_F1', test_metrics['micro F1-score'], e)
            self.writer.add_scalar('TEST/WEIGHTED_F1', test_metrics['weighted F1-score'], e)

            self.writer.flush()

            # save the model checkpoint with highest accuracy
            if test_accuracy > self.best_accuracy:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.log_dir, self.cfg.experiment_id, 'best_ckeckpoint.pth'))
                print(f'>> best checkpoint saved at epoch {e}')
                self.best_accuracy = test_accuracy
                self.best_epoch = e

            print('')
        self.writer.close()


    def validate(self):
        print('validate...')
        y_true=[] # ground truth labels
        y_pred=[] # predicted labels
        cum_acc = 0.0 
        cum_loss_val = 0.0
        self.model.eval()

        with torch.no_grad():
            for image, label in tqdm(self.val_loader):
                image = image.to(self.cfg.device)
                label = label.float().to(self.cfg.device)
                image = image.repeat(1, 3, 1, 1)
                label = label.unsqueeze(1)

                out = self.model(image)
                loss_val = self.criterion(out, label)
                cum_loss_val += loss_val.item()

                prob  = torch.sigmoid(out)               
                pred  = (prob > 0.5).long()
                y_true.append(label.cpu().view(-1))     
                y_pred.append(pred.cpu().view(-1))
                correct = (pred == label).sum().item()

                cum_acc += correct

            macro_f1 = f1_score(y_true, y_pred, average='macro')
            micro_f1 = f1_score(y_true, y_pred, average='micro')
            avg_f1 = f1_score(y_true, y_pred, average='weighted')
            acc = accuracy_score(y_true, y_pred)

            val_metrics= {
                'Accuracy': acc,
                'macro F1-score' : macro_f1,
                'micro F1-score' : micro_f1,
                'weighted F1-score' : avg_f1,
            }

        return cum_acc, cum_loss_val, val_metrics
      

    def test(self):
        print('test...')
        y_true=[] # ground truth labels
        y_pred=[] # predicted labels
        cum_acc = 0.0 
        cum_loss_test = 0.0
        self.model.eval()

        with torch.no_grad():
            for image, label in tqdm(self.test_loader):
                image = image.to(self.cfg.device)
                label = label.float().to(self.cfg.device)
                image = image.repeat(1, 3, 1, 1)
                label = label.unsqueeze(1)

                out = self.model(image)
                loss_test = self.criterion(out, label)
                cum_loss_test += loss_test.item()

                prob  = torch.sigmoid(out)               
                pred  = (prob > 0.5).long()
                y_true.append(label.cpu().view(-1))     
                y_pred.append(pred.cpu().view(-1))
                correct = (pred == label).sum().item()

                cum_acc += correct

            macro_f1 = f1_score(y_true, y_pred, average='macro')
            micro_f1 = f1_score(y_true, y_pred, average='micro')
            avg_f1 = f1_score(y_true, y_pred, average='weighted')
            acc = accuracy_score(y_true, y_pred)

            test_metrics= {
                'Accuracy': acc,
                'macro F1-score' : macro_f1,
                'micro F1-score' : micro_f1,
                'weighted F1-score' : avg_f1,
            }

        return cum_acc, cum_loss_test, test_metrics


if __name__ == "__main__":

    # create log folder
    if not os.path.exists(os.path.join(cfg.log_dir, cfg.experiment_id)):
        os.makedirs(os.path.join(os.path.join(cfg.log_dir, cfg.experiment_id)))
    else:
        raise Exception(f'Log folder {os.path.join(cfg.log_dir, cfg.experiment_id)} already exists!')
    torch.save(cfg, os.path.join(cfg.log_dir, cfg.experiment_id, 'config.txt'))
    
    BC = BinaryClassifier(cfg)

    print(f'Experiment {cfg.experiment_id}, start Training...')
    start_tstamp = datetime.now()
    BC.train()
    end_tstamp = datetime.now()
    print(f'Experiment {cfg.experiment_id}, Training started at {start_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}, Training finished at {end_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}')
    print(f'Best accuracy: {100*(BC.best_accuracy):.2f} % at epoch {BC.best_epoch}')
