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
#