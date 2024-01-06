import datetime
import torch
from src.transformer_libs import utils
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train:
    def __init__(self, model, opt, model_params, train_params, lr_params, config_params, loss_func, dl, lr_func, train_output=None, sample_function=None):
        self.model = model
        self.opt = opt
        self.train_params = train_params
        self.model_params = model_params
        self.loss_func = loss_func
        self.dl = dl
        self.lr_func = lr_func
        self.lr_params = lr_params
        self.config_params = config_params
        if train_output:
            self.train_output = train_output
        if sample_function:
            self.sample_function = sample_function
        else:
            self.sample_function = None

    def train(self, num_batches):
        torch.autograd.set_detect_anomaly(True)

        self.model.train()

        start_batch_num = self.train_params['batch_num']
        batch_num = start_batch_num
        bs = self.model_params['bs']
        accumulate_size = self.train_params['accumulate_size']

        output_every = self.train_params['output_every']
        save_every = self.train_params['save_every']

        seq_len = self.model_params['seq_len']

        samples_done = self.train_params['samples_done']
        print(f'samples_done: {samples_done}')

        lr = self.lr_func(self.lr_params)

        update_lr_every = self.train_params['update_lr_every']

        # for debugging. This slows down training by factor of 2
        # torch.autograd.set_detect_anomaly(True)

        for g in self.opt.param_groups:
            g['lr'] = lr

        print('Starting training')
        print(f'start_batch_num: {batch_num}')
        print(f'learning rate: {self.opt.param_groups[0]["lr"]}')

        # params we track during training
        total_loss = 0
        total_tokens = batch_num * bs * accumulate_size * seq_len

        start_time = datetime.datetime.now()

        accumulation_counter = 0

        save_path = self.config_params['model_directory']
        log_path = self.config_params['log_directory']

        while batch_num < start_batch_num + num_batches:

            # update dl parameters
            # update_params = {'batch_num': batch_num, 'samples_done': samples_done, }
            # self.dl.update(update_params)

            # load sample
            start_time = datetime.datetime.now()
            src, target = next(iter(self.dl))
            #src = (src[0].to(device), src[1].to(device))

            src = src.to(device)
            target = target.to(device)

            samples_done += 1

            # run through model
            pred = self.model(src)
            loss = self.loss_func(pred, target)
            if torch.isnan(loss):
                print("nan value found", pred)
                self.opt.zero_grad()
                accumulation_counter = 0
                continue
            loss = loss / accumulate_size
            loss.backward()

            # Batch is accumulated until we have enough samples
            if (accumulation_counter + 1) % accumulate_size == 0:

                # opt step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                self.opt.zero_grad()
                batch_num += 1

            total_loss += loss.item() * accumulate_size
            total_tokens += bs * seq_len #  this is actually tokens

            # log training data
            if (accumulation_counter + 1) % output_every == 0:
                end_time = datetime.datetime.now()
                lr = self.opt.param_groups[0]["lr"]
                current_loss = total_loss / output_every
                total_loss = 0
                time_for_batch_interval = end_time - start_time

                log_data = {}
                file_id = self.model_params['id']
                log_data['batch_num'] = batch_num
                log_data['samples_done'] = samples_done
                log_data['total_tokens'] = total_tokens
                log_data['lr'] = lr
                log_data['current_loss'] = current_loss
                log_data['accumulate_size'] = accumulate_size
                log_data['output_every'] = output_every // accumulate_size
                log_data['time_for_batch_interval'] = time_for_batch_interval

                utils.log(file_id, log_data, log_path, log_screen=True)
                if hasattr(self, 'train_output') and callable(self.train_output):
                    output_dict = {'pred': pred, 'target': target, 'log_data': log_data}
                    self.train_output(**output_dict)
                if self.sample_function:
                    self.sample_function(pred, target)

            # update learning rate
            if (accumulation_counter + 1) % update_lr_every == 0:
                self.lr_params['batch_num'] = batch_num * accumulate_size
                lr = self.lr_func(self.lr_params)

                for g in self.opt.param_groups:
                    g['lr'] = lr

            # save the model
            if (accumulation_counter + 1) % save_every == 0:
                # Ensure save_path is a directory and exists
                if not os.path.isdir(save_path):
                    os.makedirs(save_path, exist_ok=True)
                    print(f"Path did not exists.  Creating a new path {save_path}")

                # Update model parameters
                model_params_copy = self.model_params.copy()
                model_params_copy['batch_num'] = batch_num
                model_params_copy['samples_done'] = samples_done

                # Save the model
                utils.save_model(self.model, self.opt, save_path, model_params_copy, self.train_params, self.lr_params)

            accumulation_counter += 1
        print("Ending training")
        utils.save_model(self.model, self.opt, self.model_params)
        return
