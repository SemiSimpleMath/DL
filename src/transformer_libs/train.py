import datetime
import torch
from transformer_libs import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train:
    def __init__(self, model, opt, model_params, train_params, lr_params, config_params, tok, ds, loss_func, dl_func, lr_func):
        self.model = model
        self.opt = opt
        self.train_params = train_params
        self.model_params = model_params
        self.tok = tok
        self.ds = ds
        self.loss_func = loss_func
        self.dl_func = dl_func
        self.lr_func = lr_func
        self.lr_params = lr_params
        self.config_params = config_params

    def train(self, num_batches):
        self.model.train()
        d_model = self.model_params['d_model']
        start_batch_num = self.model_params['batch_num']
        batch_num = start_batch_num

        bs = self.model_params['bs']
        accumulate_size = self.train_params['accumulate_size']
        output_every = self.train_params['output_every']
        save_every = self.train_params['save_every']

        seq_len = self.model_params['seq_len']

        samples_done = self.model_params['samples_done']
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
        total_sequences = batch_num * bs * accumulate_size * seq_len

        start_time = datetime.datetime.now()

        accumulation_counter = 0

        save_path = self.config_params['model_directory']
        log_path = self.config_params['log_directory']

        while batch_num < start_batch_num + num_batches:
            # load sample
            combined = self.dl_func(self.ds, self.tok, bs, samples_done, seq_len)  # combined is bs x (L + 1)
            samples_done += 1

            src = combined[:, :-1].to(device)  # bs x L
            target = combined[:, 1:].to(device)  # bs x L
            # positional encoding
            pe = utils.get_pe(src.size()[-1], d_model).to(device)  # 1 x L x d_model
            # run through model
            pred = self.model(src, pe)

            # compute loss
            pred = pred.permute(0, 2, 1)
            loss = self.loss_func(pred, target) / accumulate_size

            loss.backward()

            # Batch is accumulated until we have enough samples
            if (accumulation_counter + 1) % accumulate_size == 0:
                # opt step
                self.opt.step()
                self.opt.zero_grad()
                batch_num += 1

            total_loss += loss.item() * accumulate_size
            total_sequences += bs * seq_len

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
                log_data['total_seq'] = total_sequences
                log_data['lr'] = lr
                log_data['current_loss'] = current_loss
                log_data['accumulate_size'] = accumulate_size
                log_data['output_every'] = output_every // accumulate_size
                log_data['time_for_batch_interval'] = time_for_batch_interval

                utils.log(file_id, log_data, log_path, log_screen=True)

                # To report loss per position uncomment both lines below
                # pred = pred.permute(0, 2, 1)
                # print(f'Loss by position: {loss_by_position(pred, target, bs, seq_len, loss)}')

                start_time = datetime.datetime.now()

            # update learning rate
            if (accumulation_counter + 1) % update_lr_every == 0:
                lr = self.lr_func(self.lr_params)

                for g in self.opt.param_groups:
                    g['lr'] = lr

            # save the model
            if (accumulation_counter + 1) % save_every == 0:
                self.model_params['batch_num'] = batch_num
                self.model_params['samples_done'] = samples_done
                utils.save_model(self.model, self.opt, save_path, self.model_params, self.train_params, self.lr_params)
                print(f'Epoch progress: {samples_done * bs / len(self.ds)}')

            # Evaluate the model on a validation set
            # if (accumulation_counter + 1) % config.eval_every == 0:
            #     model_params['batch_num'] = batch_num
            #     eval_model(model, tok, loss_func, bs, 100, seq_len, model_params)
            #     model.train()
            #     print("Starting re-starting training")

            accumulation_counter += 1
        print("Ending training")
        utils.save_model(self.model, self.opt, self.model_params)
        return


    # def eval_model(self, model, tok, loss_func, bs, num_batches, seq_len, model_params):
    #     print("Starting eval")
    #     model.eval()
    #
    #     ds = data_utils.load_ds(config.wiki_ds_file)
    #
    #     batch_num = 0
    #     d_model = model_params['d_model']
    #     samples_done = 0
    #     total_loss = 0
    #     start_time = datetime.datetime.now()
    #
    #     while batch_num < num_batches:
    #         # load sample
    #         combined = data_utils.get_batch(ds, tok, bs, samples_done, seq_len + 1)  # combined is bs x (L + 1)
    #         samples_done += 1
    #
    #         src = combined[:, :-1].to(device)  # bs x L
    #         target = combined[:, 1:].to(device)  # bs x L
    #         # positional encoding
    #         pe = utils.get_pe(src.size()[-1], d_model).to(device)  # 1 x L x d_model
    #         # run through model
    #         pred = model(src, pe)
    #         # compute loss
    #         pred = pred.permute(0, 2, 1)
    #         loss = loss_func(pred, target)
    #
    #         total_loss += loss.item()
    #
    #         if (samples_done + 1) % config.accumulate_size == 0:
    #             batch_num += 1
    #
    #         # log eval data
    #         if (samples_done + 1) % config.output_every == 0:
    #             end_time = datetime.datetime.now()
    #             current_loss = total_loss / config.output_every
    #             total_loss = 0
    #             time_for_batch_interval = end_time - start_time
    #
    #             log_data = {}
    #             file_id = model_params['id']
    #             log_data['batch_num'] = batch_num
    #             log_data['current_loss'] = current_loss
    #             log_data['batch_interval'] = config.output_every // config.accumulate_size
    #             log_data['time_for_batch_interval'] = time_for_batch_interval
    #
    #             utils.log('eval' + str(file_id), log_data, log_screen=True)
    #
    #             # To report loss per position uncomment both lines below
    #             # pred = pred.permute(0, 2, 1)
    #             # print(f'Loss by position: {loss_by_position(pred, target, bs, seq_len, loss)}')
    #
    #             start_time = datetime.datetime.now()
    #
    #     print("Ending eval")
    #     return