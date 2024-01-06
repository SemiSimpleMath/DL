from sentence_transformers import SentenceTransformer
import st_utils
import vMF
import numpy as np
model_hf = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
import save_load_models
import config
import torch
from torch.utils.data import Dataset, DataLoader
import STVAE
import VAEDataLoader
import random
print(torch.version.cuda)
print(torch.cuda.is_available())
import datetime
import torch.nn.functional as F
seed_value = 42  # Choose any integer value as the seed

# Set the seed for Python's random number generator
random.seed(seed_value)

# Set the seed for NumPy's random number generator

np.random.seed(seed_value)

# Set the seed for PyTorch's random number generator
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using CUDA

LOAD_WIKIPEDIA = False
LOAD_TEXT_FILE = False
GENERATE_BOOK_SENTENCES = False
LOAD_DATA_FILE = True
SAVE_SENTENCES = False

LOAD_MODEL = False

LOAD_NON_NORMALIZED = False

if LOAD_WIKIPEDIA:
    #Load the data from data file
    loaded_data = st_utils.load_data('wikipedia_sentences.pkl')
    all_sentences = loaded_data[:10000000]
    print("sentences loaded")
elif LOAD_DATA_FILE:
    loaded_data = st_utils.load_data('wiki_data.pkl')
    all_sentences, all_semantic_sentences = loaded_data


elif LOAD_TEXT_FILE:
    # Load from specific file
    sentences = st_utils.load_and_split_sentences("C:\\Users\\semis\\IdeaProjects\\DL\\semantic transformer\\data\\wiki_sample1.txt")

elif GENERATE_BOOK_SENTENCES:
    # generate sentences
    sentences = st_utils.generate_sentences()

# if not LOAD_DATA_FILE:
#     all_semantic_sentences = model_hf.encode(all_sentences)
#     print("Semantic sentences created.")

if SAVE_SENTENCES:
    st_utils.save_data(all_sentences, all_semantic_sentences, 'wiki_data.pkl')

if LOAD_NON_NORMALIZED:
    all_sentences, all_semantic_sentences = st_utils.load_data("non_normalized_sentences.pkl")

#all_sentences = all_sentences[:1000]
all_semantic_sentences = all_semantic_sentences[:1000000]
all_semantic_sentences = [torch.Tensor(a) for a in all_semantic_sentences]



print("Semantic sentences created")


seq_len = config.tf_model_params['seq_len']
bs = config.tf_model_params['bs']
d_model = config.tf_model_params['d_model']

dataset = VAEDataLoader.SemanticSentencesDataset(all_semantic_sentences, seq_len)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=VAEDataLoader.collate_fn)

# Transformer parameters
num_blocks = config.tf_model_params['num_blocks']
d_model = config.tf_model_params['d_model']
d_middle = config.tf_model_params['d_middle']

dropout = config.tf_model_params['dropout']
h = config.tf_model_params['h']
d_Q = config.tf_model_params['d_Q']
d_K = config.tf_model_params['d_K']
d_V = config.tf_model_params['d_V']


# Hyperparameters
num_epochs = 2000
learning_rate = config.lr_params['constant_lr']
beta = 1.0

# output_dim = config.VAE_model_params['output_dim']
# latent_dim = config.VAE_model_params['latent_dim']
# kwargs = {'latent_dim':latent_dim, 'input_dim': d_model, 'n_components': n_components}

# Initialize the model, optimizer, and loss function
embedding_model = vMF.vMFNet
n_components = config.VMF_COMPONENTS
input_dim = d_model
output_dim = d_model
hidden_dim = config.VMF_HIDDEN_DIM
kwargs = {'input_dim':input_dim, 'hidden_dim': hidden_dim, 'output_dim': output_dim, 'n_components': n_components, 'dropout_p': dropout}

model = STVAE.Decoder(num_blocks, d_model, d_middle, dropout, h, d_Q, d_K, d_V, embedding_model, **kwargs)


if LOAD_MODEL:
    model = save_load_models.load_model(model, 'VAEmodel4.pkl')
    print("model loaded")
model.to('cuda')



print(f'LR: {learning_rate}')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_params = count_parameters(model)
print(f"The model has {num_params} trainable parameters.")

model.train()
running_loss = 0.0
loss_count = 0
total_batch_count = 0
file_id = 4
start_time = datetime.datetime.now()
cos_sim_total = 0
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for idx, (input_seq, target_seq) in enumerate(dataloader):
        total_batch_count += 1
        input_seq = input_seq.to('cuda')
        target_seq = target_seq.to('cuda')
        is_offset_by_one = torch.allclose(input_seq[:, 1:,:], target_seq[:,0:-1,:])
        if not is_offset_by_one:
            print("NOPE!")

        params = model(input_seq)

        # Compute the loss
        current_batch_size, seq_len, output_dim = target_seq.size()
        target_seq = target_seq.view(current_batch_size * seq_len, output_dim)
        input_seq = input_seq.view(current_batch_size * seq_len, output_dim)
        loss = vMF.loss(params, target_seq)


        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update the running loss and count
        running_loss += loss.item()
        loss_count += 1

        # log training data
        if (idx + 1) % 10 == 0:
            avg_loss = running_loss / loss_count if loss_count > 0 else 0
            end_time = datetime.datetime.now()
            current_loss = avg_loss
            total_loss = 0
            time_for_batch_interval = end_time - start_time

            log_data = {}
            log_data['batch_num'] = idx
            log_data['samples_done'] = idx * total_batch_count
            log_data['current_loss'] = current_loss
            log_data['time_for_batch_interval'] = time_for_batch_interval

            st_utils.log(file_id, log_data, config.log_directory)

            # To report loss per position uncomment both lines below
            # pred = pred.permute(0, 2, 1)
            # print(f'Loss by position: {loss_by_position(pred, target, bs, seq_len, loss)}')

            start_time = datetime.datetime.now()

        # Print the progress and average loss over the last 1k batches
        if idx % 100 == 0 and idx != 0:
            avg_loss = running_loss / loss_count if loss_count > 0 else 0
            cos_sim_ave = cos_sim_total / loss_count
            mus, weights = params
            predicted_samples = vMF.sample_from_mixture(mus, weights)
            cos_similarities = F.cosine_similarity(predicted_samples, target_seq, dim=1).mean()

            print(cos_similarities)
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, cos_sim: {:.4f} total batch count: {}".format(
                epoch+1, num_epochs, idx, len(dataloader), avg_loss, cos_sim_ave, total_batch_count))
            file = "VMFmodel2.pkl"
            save_load_models.save_model(model, file)

            # Reset the running loss and count
            running_loss = 0.
            cos_sim_total = 0.
            loss_count = 0


            # # update learning rate
            # if epoch > 15:
            #     lr = 1e-6
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr

