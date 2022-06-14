import torch
import torch.nn as nn
import numpy as np
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(
            self, 
            exp_name, 
            train,
            log_dir, 
            total_steps,
            n_inner_step, 
            n_finetuning_step, 
            n_valid_step,
            n_test_step,
            every_valid_step,
            beta,
            gamma,
            lambda1,
            lambda2,
            outer_lr,
            clip_value,
            device: str='cpu',
            print_step: int=5
        ):
        self.train=train
        self.device = device
        self.print_step = print_step
        self.total_steps = total_steps
        self.n_inner_step = n_inner_step
        self.n_finetuning_step = n_finetuning_step
        self.n_valid_step = n_valid_step
        self.n_test_step = n_test_step
        self.every_valid_step = every_valid_step
        
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1  # penalty on model(encoder, mapping_net, decoder) parameters
        self.lambda2 = lambda2  # penalty on decoder
        self.outer_lr = outer_lr
        self.clip_value = clip_value
        

        # check if exp exists
        self.exp_name = exp_name
        self.log_dir = Path(log_dir).resolve()
        exp_dirs = sorted(list(self.log_dir.glob(f'{self.exp_name}_*')))
        if self.train:
            exp_num = int(exp_dirs[-1].name[len(self.exp_name)+1:]) if exp_dirs else 0
            self.exp_dir = self.log_dir / f'{self.exp_name}_{exp_num+1}'
            self.writer = SummaryWriter(str(self.exp_dir))
            self.ckpt_path = self.exp_dir / 'checkpoints'
            if not self.ckpt_path.exists():
                self.ckpt_path.mkdir()
        else:
            exp_num = int(exp_dirs[-1].name[len(self.exp_name)+1:]) if exp_dirs else 0
            self.load_model_dir = Path(f'{exp_dirs[-1]}').resolve()
            self.load_model_dir_list = list(self.load_model_dir.glob("checkpoints/*"))
            # print(load_model_dir_list)
            self.load_model_path = self.load_model_dir_list[0]
            test_dir = self.log_dir/'test'
            if not test_dir.exists():
                test_dir.mkdir()
            test_exp_dirs = sorted(list((test_dir).glob(f'{self.exp_name}_*')))
            test_exp_num = int(test_exp_dirs[-1].name[len(self.exp_name)+1:]) if test_exp_dirs else 0
            self.test_log_dir = test_dir / f'{self.exp_name}_{test_exp_num+1}'
            self.writer = SummaryWriter(str(self.test_log_dir))
            
        
        
        # aggregate method by window sizes
        self.log_keys = {
            'Support Loss': np.sum, 
            'Support Accuracy': np.mean, 
            'Query Loss': np.sum, 
            'Query Accuracy': np.mean, 
            'Total Loss': np.sum,
            'Inner LR': np.mean,  # average Learning Rate
            'Finetuning LR': np.mean, 
            'KLD Loss': np.sum, 
            'Z Penalty': np.sum, 
            'Orthogonality Penalty': np.sum
        }
        
    def map_to_tensor(self, tasks, device: None or str=None):
        if device is None:
            device = torch.device('cpu')
        else:
            device = torch.device(device)
        tensor_tasks = {}
        for k, v in tasks.items():
            tensor_fn = torch.LongTensor if 'labels' in k else torch.FloatTensor
            tensor = tensor_fn(np.array(v))
            if ('labels' not in k) and tensor.ndim == 1:
                tensor = tensor.view(1, -1)
            tensor_tasks[k] = tensor.to(device)
        return tensor_tasks

    def step_batch(self, model, batch_data):
        total_loss, records = model.meta_run(
            data=batch_data, 
            beta=self.beta, 
            gamma=self.gamma, 
            lambda2=self.lambda2, 
            n_inner_step=self.n_inner_step, 
            n_finetuning_step=self.n_finetuning_step, 
            rt_attn=False
        )
        return total_loss, records

    def manual_model_eval(self, model, mode=False):
        # [PyTorch Issue] RuntimeError: cudnn RNN backward can only be called in training mode
        # cannot use model.eval()
        # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        for module in model.children():
            # model.training = mode
            if isinstance(module, nn.Dropout) or isinstance(module, nn.LayerNorm):
                module.train(mode)
        return model

    def meta_train(self, model, meta_trainset):
        model = model.to(self.device)
        lr_list = ['inner_lr', 'finetuning_lr']
        params = [x[1] for x in list(filter(lambda k: k[0] not in lr_list, model.named_parameters()))]
        lr_params = [x[1] for x in list(filter(lambda k: k[0] in lr_list, model.named_parameters()))]
        optim = torch.optim.Adam(params, lr=self.outer_lr, weight_decay=self.lambda1)
        optim_lr = torch.optim.Adam(lr_params, lr=self.outer_lr)
        best_eval_acc = 0.0

        for step in range(self.total_steps):
            # Meta Train
            model.train()
            optim.zero_grad()
            optim_lr.zero_grad()
            train_tasks = meta_trainset.generate_tasks()
            train_records = {k: [] for k in self.log_keys.keys()}
            
            all_total_loss = 0.
            for window_size, tasks in train_tasks.items(): # window size x (n_sample * n_stock)
                batch_data = self.map_to_tensor(tasks, device=self.device)
                total_loss, records = self.step_batch(model, batch_data)
                # version - 2
                all_total_loss += total_loss
                

                for key, v in records.items():
                    if (key in ['Latents']):
                        self.writer.add_histogram(f'Latents-WinSize={window_size}', records['Latents'], step)
                    else:
                        train_records[key].append(v)
                        self.writer.add_scalar(f'Train-WinSize={window_size}-{key}', v, step)
                # total_loss.backward()
                # optim.step()
                # optim_lr.step()
                
            all_total_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
            optim.step()
            optim_lr.step()

            # logging summary(aggregate score for all window size tasks)
            # for key, agg_func in self.log_keys.items():
            #     self.writer.add_scalar(f'Train-{key}', agg_func(train_records[key]), step)

            if (step % self.print_step == 0) or (step == self.total_steps-1):
                for key, agg_func in self.log_keys.items():
                    self.writer.add_scalar(f'Train-{key}', agg_func(train_records[key]), step)
            
                print(f'[Meta Train]({step+1}/{self.total_steps})')
                for i, (key, agg_func) in enumerate(self.log_keys.items()):
                    s1 = '  ' if (i == 0) or (i == 4) else ''
                    s2 = '\n' if i == 3 else " | "
                    print(f'{s1}{key}: {agg_func(train_records[key]):.4f}', end=s2)
                print()
                
            # Meta Valid
            if (step % self.every_valid_step == 1) or (step == self.total_steps-1):
                # [PyTorch Issue] RuntimeError: cudnn RNN backward can only be called in training mode
                # cannot use model.eval()
                model.manual_model_eval()

                valid_records = {'Accuracy': [], 'Loss': []}  # n_valid_step x window_size 
                for val_step in range(self.n_valid_step):
                    valid_step_loss = []
                    valid_step_acc = []
                    valid_tasks = meta_trainset.generate_tasks()

                    for window_size, tasks in valid_tasks.items():
                        batch_data = self.map_to_tensor(tasks, device=self.device)
                        _, records = self.step_batch(model, batch_data)
                        valid_step_loss.append(records['Query Loss'])
                        valid_step_acc.append(records['Query Accuracy'])

                    valid_records['Accuracy'].append(valid_step_acc)
                    valid_records['Loss'].append(valid_step_loss)
                
                # aggregate window loss and accruacy: mean by n_valid_step
                valid_records['Accuracy'] = np.mean(valid_records['Accuracy'], axis=0)
                valid_records['Loss'] = np.mean(valid_records['Loss'], axis=0)
              

                for key, agg_func in zip(['Accuracy', 'Loss'], [np.mean, np.sum]):
                    for i, window_size in enumerate(meta_trainset.window_sizes):
                        self.writer.add_scalar(f'Valid-WinSize={window_size}-{key}', valid_records[key][i], step)
                    self.writer.add_scalar(f'Valid-Task {key}', agg_func(valid_records[key]), step)

                print(f'[Meta Valid]({step+1}/{self.total_steps})')
                for i, (key, agg_func) in enumerate(zip(['Accuracy', 'Loss'], [np.mean, np.sum])):
                    s1 = '  ' if i == 0 else ''
                    s2 = ' | ' if i == 0 else '\n'
                    print(f'{s1}{key}: {agg_func(valid_records[key]):.4f}', end=s2)
                    
                cur_eval_loss = np.mean(valid_records['Loss'])
                cur_eval_acc = np.mean(valid_records['Accuracy'])
                if cur_eval_acc > best_eval_acc:
                    best_eval_acc = cur_eval_acc 
                    torch.save(model.state_dict(), str(self.ckpt_path / f'{step}-{cur_eval_acc:.4f}-{cur_eval_loss:.4f}.ckpt'))

    def meta_test(self, model, meta_testset):
        # load state dict
        model = model.to(self.device)
        print(self.load_model_dir)
    
        cur_model_acc = float(str(self.load_model_path).split("-")[1])
        for path in self.load_model_dir_list:
            new_model_acc = float(str(path).split("-")[1])
            if new_model_acc > cur_model_acc:
                self.load_model_path = path
                cur_model_acc = new_model_acc
        print(self.load_model_path)
        state_dict = torch.load(self.load_model_path)
        model.load_state_dict(state_dict)
        model.manual_model_eval()
        

        for test_step in range(self.n_test_step):
            test_records = {'Accuracy': [], 'Loss': []} 
            test_step_loss = []
            test_step_acc = []
            test_tasks = meta_testset.generate_tasks()

            for window_size, tasks in test_tasks.items():
                batch_data = self.map_to_tensor(tasks, device=self.device)
                _, records = self.step_batch(model, batch_data)
                test_step_loss.append(records['Query Loss'])
                test_step_acc.append(records['Query Accuracy'])

            test_records['Accuracy'].append(test_step_acc)
            test_records['Loss'].append(test_step_loss)
                
                # aggregate window loss and accruacy: mean by n_test_step
            test_records['Accuracy'] = np.mean(test_records['Accuracy'], axis=0)
            test_records['Loss'] = np.mean(test_records['Loss'], axis=0)
              

            for key, agg_func in zip(['Accuracy', 'Loss'], [np.mean, np.sum]):
                for i, window_size in enumerate(meta_testset.window_sizes):
                    self.writer.add_scalar(f'Test-WinSize={window_size}-{key}', test_records[key][i], test_step)
                self.writer.add_scalar(f'Test-Task {key}', agg_func(test_records[key]), test_step)

            print(f'[Meta Test]({test_step+1}/{self.n_test_step})')
            for i, (key, agg_func) in enumerate(zip(['Accuracy', 'Loss'], [np.mean, np.sum])):
                s1 = '  ' if i == 0 else ''
                s2 = ' | ' if i == 0 else '\n'
                print(f'{s1}{key}: {agg_func(test_records[key]):.4f}', end=s2)
                
 
