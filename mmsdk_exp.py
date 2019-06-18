# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Tutorial on CMU-Multimodal SDK
#
# This is a tutorial on using ***CMU-Multimodal SDK*** to load and process multimodal time-series datasets and training a simple late-fusion LSTM model on the processed data. 
#
# For this tutorial, we specify some constants in `./constans/paths.py`. Please first take a look and modify the paths to point to the correct folders.
#
# ## Downloading the data
#
# We start off by (down)loading the datasets. In the SDK each dataset has three sets of content: `highlevel`, `raw` and `labels`. `highlevel` contains the extracted features for each modality (e.g OpenFace facial landmarks, openSMILE acoustic features) while `raw` contains the raw transctripts, phonemes. `labels` are self-explanatory. Note that some datasets have more than just one set of annotations so `labels` could also give you multiple files.
#
# Currently there's a caveat that the SDK will not automatically detect if you have downloaded the data already. In event of that it will throw a `RuntimeError`. We work around that by `try/except`. This is not ideal but it will work for now.

# +
import dataset_helper
from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
import sys

if SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    sys.path.append(SDK_PATH)

import mmsdk
import os
import re
import numpy as np
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError

# create folders for storing the data
if not os.path.exists(DATA_PATH):
    check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

# download highlevel features, low-level (raw) data and labels for the dataset MOSI
# if the files are already present, instead of downloading it you just load it yourself.
# here we use CMU_MOSI dataset as example.

DATASET = md.cmu_mosi

try:
    md.mmdataset(DATASET.highlevel, DATA_PATH)
except RuntimeError:
    print("High-level features have been downloaded previously.")

try:
    md.mmdataset(DATASET.raw, DATA_PATH)
except RuntimeError:
    print("Raw data have been downloaded previously.")
    
try:
    md.mmdataset(DATASET.labels, DATA_PATH)
except RuntimeError:
    print("Labels have been downloaded previously.")
# -

# ## Inspecting the downloaded files
#
# We can print the files in the target data folder to see what files are there.
#
# We can observe a bunch of files ending with `.csd` extension. This stands for ***computational sequences***, which is the underlying data structure for all features in the SDK. We will come back to that later when we load the data. For now we just print out what computational sequences we have downloaded.

# list the directory contents... let's see what features there are
data_files = os.listdir(DATA_PATH)
print('\n'.join(data_files))

# ## Loading a multimodal dataset
#
# Loading the dataset is as simple as telling the SDK what are the features you need and where are their computational sequences. You can construct a dictionary with format `{feature_name: csd_path}` and feed it to `mmdataset` object in the SDK.

# +
# define your different modalities - refer to the filenames of the CSD files
visual_field = 'CMU_MOSI_VisualFacet_4.1'
acoustic_field = 'CMU_MOSI_COVAREP'
text_field = 'CMU_MOSI_ModifiedTimestampedWords'


features = [
    text_field, 
    visual_field, 
    acoustic_field
]

recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
dataset = md.mmdataset(recipe)
# -

# ## A peek into the dataset
#
# The multimodal dataset, after loaded, has the following hierarchy:
#
#
# ```
#             computational_sequence_1 ---...
#            /                                   ...
#           /                                    /
#          /                          first_video     features -- T X N array
#         /                          /               /
# dataset ---computational_sequence_2 -- second_video
#         \                          \               \
#          \                          third_video     intervals -- T X 2 array
#           \                                    \...
#            \
#             computational_sequence_3 ---...
# ```
#
# It looks like a nested dictionary and can be indexed as if it is a nested dictionary. A dataset contains multiple computational sequences whose key is the `text_field`, `visual_field`, `acoustic_field` as defined above. Each computational sequence, however, has multiple video IDs in it, and different computational sequences are supposed to have the same set of video IDs. Within each video, there are two arrays: `features` and `intervals`, denoting the feature values at each time step and the start and end timestamp for each step. We can take a look at its content.

# +
print(list(dataset.keys()))
print("=" * 80)

print(list(dataset[visual_field].keys())[:10])
print("=" * 80)

some_id = list(dataset[visual_field].keys())[15]
print(list(dataset[visual_field][some_id].keys()))
print("=" * 80)

print(list(dataset[visual_field][some_id]['intervals'].shape))
print("=" * 80)

print(list(dataset[visual_field][some_id]['features'].shape))
print(list(dataset[text_field][some_id]['features'].shape))
print(list(dataset[acoustic_field][some_id]['features'].shape))
print("Different modalities have different number of time steps!")


# -

# ## Alignment of multimodal time series
#
# To work with multimodal time series that contains multiple views of data with different frequencies, we have to first align them to a ***pivot*** modality. The convention is to align to ***words***. Alignment groups feature vectors from other modalities into bins denoted by the timestamps of the pivot modality, and apply a certain processing function to each bin. We call this function ***collapse function***, because usually it is a pooling function that collapses multiple feature vectors from another modality into one single vector. This will give you sequences of same lengths in each modality (as the length of the pivot modality) for all videos.
#
# Here we define our collapse funtion as simple averaging. We feed the function to the SDK when we invoke `align` method. Note that the SDK always expect collapse functions with two arguments: `intervals` and `features`. Even if you don't use intervals (as is in the case below) you still need to define your function in the following way.
#
# ***Note: Currently the SDK applies the collapse function to all modalities including the pivot, and obviously text modality cannot be "averaged", causing some errors. My solution is to define the avg function such that it averages the features when it can, and return the content as is when it cannot average.***

# +
# we define a simple averaging function that does not depend on intervals
def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features


# -

# ## Append annotations to the dataset and get the data points
#
# Now that we have a preprocessed dataset, all we need to do is to apply annotations to the data. Annotations are also computational sequences, since they are also just some values distributed on different time spans (e.g 1-3s is 'angry', 12-26s is 'neutral'). Hence, we just add the label computational sequence to the dataset and then align to the labels. Since we (may) want to preserve the whole sequences, this time we don't specify any collapse functions when aligning. 
#
# Note that after alignment, the keys in the dataset changes from `video_id` to `video_id[segment_no]`, because alignment will segment each datapoint based on the segmentation of the pivot modality (in this case, it is segmented based on labels, which is what we need, and yes, one code block ago they are segmented to word level, which I didn't show you).
#
# ***Important: DO NOT add the labels together at the beginning, the labels will be segmented during the first alignment to words. This also holds for any situation where you want to do multiple levels of alignment.***

# +
label_field = 'CMU_MOSI_Opinion_Labels'

# we add and align to lables to obtain labeled segments
# this time we don't apply collapse functions so that the temporal sequences are preserved
label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
dataset.add_computational_sequences(label_recipe, destination=None)
dataset_helper.align(dataset, label_field)
# first we align to words with averaging, collapse_function receives a list of functions
dataset_helper.align(dataset, text_field)
# -

# check out what the keys look like now
print(list(dataset[text_field].keys())[55])

# ## Splitting the dataset
#
# Now it comes to our final step: splitting the dataset into train/dev/test splits. This code block is a bit long in itself, so be patience and step through carefully with the explanatory comments.
#
# The SDK provides the splits in terms of video IDs (which video belong to which split), however, after alignment our dataset keys already changed from `video_id` to `video_id[segment_no]`. Hence, we need to extract the video ID when looping through the data to determine which split each data point belongs to.
#
# In the following data processing, I also include instance-wise Z-normalization (subtract by mean and divide by standard dev) and converted words to unique IDs.
#
# This example is based on PyTorch so I am using PyTorch related utils, but the same procedure should be easy to adapt to other frameworks.

# +
# obtain the train/dev/test splits - these splits are based on video IDs
train_split = DATASET.standard_folds.standard_train_fold
dev_split = DATASET.standard_folds.standard_valid_fold
test_split = DATASET.standard_folds.standard_test_fold

# inspect the splits: they only contain video IDs
print(test_split)

# +
# we can see they are in the format of 'video_id[segment_no]', but the splits was specified with video_id only
# we need to use regex or something to match the video IDs...
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook
from collections import defaultdict

# a sentinel epsilon for safe division, without it we will replace illegal values with a constant
EPS = 0

# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

# place holders for the final train/dev/test dataset
train = []
dev = []
test = []

# define a regular expression to extract the video ID out of the keys
pattern = re.compile('(.*)\[.*\]')
num_drop = 0 # a counter to count how many data points went into some processing issues

for segment in dataset[label_field].keys():
    
    # get the video ID and the features out of the aligned dataset
    vid = re.search(pattern, segment).group(1)
    label = dataset[label_field][segment]['features']
    _words = dataset[text_field][segment]['features']
    _visual = dataset[visual_field][segment]['features']
    _acoustic = dataset[acoustic_field][segment]['features']

    # if the sequences are not same length after alignment, there must be some problem with some modalities
    # we should drop it or inspect the data again
    if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
        print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
        num_drop += 1
        continue

    # remove nan values
    label = np.nan_to_num(label)
    _visual = np.nan_to_num(_visual)
    _acoustic = np.nan_to_num(_acoustic)

    # remove speech pause tokens - this is in general helpful
    # we should remove speech pauses and corresponding visual/acoustic features together
    # otherwise modalities would no longer be aligned
    words = []
    visual = []
    acoustic = []
    for i, word in enumerate(_words):
        if word[0] != b'sp':
            words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
            visual.append(_visual[i, :])
            acoustic.append(_acoustic[i, :])

    words = np.asarray(words)
    visual = np.asarray(visual)
    acoustic = np.asarray(acoustic)

    # z-normalization per instance and remove nan/infs
    visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
    acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

    if vid in train_split:
        train.append(((words, visual, acoustic), label, segment))
    elif vid in dev_split:
        dev.append(((words, visual, acoustic), label, segment))
    elif vid in test_split:
        test.append(((words, visual, acoustic), label, segment))
    else:
        print(f"Found video that doesn't belong to any splits: {vid}")

print(f"Total number of {num_drop} datapoints have been dropped.")

# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK
word2id.default_factory = return_unk
# -

# ## Inspect the dataset
#
# Now that we have loaded the data, we can check the sizes of each split, data point shapes, vocabulary size, etc.

# +
# let's see the size of each set and shape of data
print(len(train))
print(len(dev))
print(len(test))

print(train[0][0][1].shape)
print(train[0][1].shape)
print(train[0][1])

print(f"Total vocab size: {len(word2id)}")


# -

# ## Collate function in PyTorch
#
# Collate functions are functions used by PyTorch dataloader to gather batched data from dataset. It loads multiple data points from an iterable dataset object and put them in a certain format. Here we just use the lists we've constructed as the dataset and assume PyTorch dataloader will operate on that.

# +
def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, lengths

# construct dataloaders, dev and test could use around ~X3 times batch size since no_grad is used during eval
batch_sz = 56
train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)

# let's create a temporary dataloader just to see how the batch looks like
temp_loader = iter(DataLoader(test, shuffle=True, batch_size=8, collate_fn=multi_collate))
batch = next(temp_loader)

print(batch[0].shape) # word vectors, padded to maxlen
print(batch[1].shape) # visual features
print(batch[2].shape) # acoustic features
print(batch[3]) # labels
print(batch[4]) # lengths
# -

# Let's actually inspect the transcripts to ensure it's correct
id2word = {v:k for k, v in word2id.items()}
examine_target = train
idx = np.random.randint(0, len(examine_target))
print(' '.join(list(map(lambda x: id2word[x], examine_target[idx][0][0].tolist()))))
# print(' '.join(examine_target[idx][0]))
print(examine_target[idx][1])
print(examine_target[idx][2])


# ## Define a multimodal model
#
# Here we show a simple example of late-fusion LSTM. Late-fusion refers to combining the features from different modalities at the final prediction stage, without introducing any interactions between them before that.

# let's define a simple model that can deal with multimodal variable length sequence
class LFLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, fc1_size, output_size, dropout_rate):
        super(LFLSTM, self).__init__()
        self.input_size = input_sizes
        self.hidden_size = hidden_sizes
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.embed = nn.Embedding(len(word2id), input_sizes[0])
        self.trnn1 = nn.LSTM(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = nn.LSTM(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = nn.LSTM(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = nn.LSTM(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = nn.LSTM(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = nn.LSTM(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.fc1 = nn.Linear(sum(hidden_sizes)*4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))
        self.bn = nn.BatchNorm1d(sum(hidden_sizes)*4)

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

        
    def fusion(self, sentences, visual, acoustic, lengths):
        batch_size = lengths.size(0)
        sentences = self.embed(sentences)
        
        # extract features from text modality
        final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        
        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        
        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)

        
        # simple late fusion -- concatenation + normalization
        h = torch.cat((final_h1t, final_h2t, final_h1v, final_h2v, final_h1a, final_h2a),
                       dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def forward(self, sentences, visual, acoustic, lengths):
        batch_size = lengths.size(0)
        h = self.fusion(sentences, visual, acoustic, lengths)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o


# ## Load pretrained embeddings
#
# We define a function for loading pretrained word embeddings stored in GloVe-style file. Contextualized embeddings obviously cannot be stored and loaded this way, though.

# +
# define a function that loads data from GloVe-like embedding files
# we will add tutorials for loading contextualized embeddings later
# 2196017 is the vocab size of GloVe here.

def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    f = open(path_to_embedding, 'r')
    found = 0
    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()


# -

# ## Training a model
#
# Next we train a model. We use Adam with gradient clipping and weight decay for training, and our loss here is Mean Absolute Error (MOSI is a regression dataset). We exclude the embeddings from trainable computation graph to prevent overfitting. We also apply a early-stopping scheme with learning rate annealing based on validation loss.

# +
from tqdm import tqdm_notebook
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

CUDA = torch.cuda.is_available()
MAX_EPOCH = 1000

text_size = 300
visual_size = 47
acoustic_size = 74

# define some model settings and hyper-parameters
input_sizes = [text_size, visual_size, acoustic_size]
hidden_sizes = [int(text_size * 1.5), int(visual_size * 1.5), int(acoustic_size * 1.5)]
fc1_size = sum(hidden_sizes) // 2
dropout = 0.25
output_size = 1
curr_patience = patience = 8
num_trials = 3
grad_clip_value = 1.0
weight_decay = 0.1

if os.path.exists(CACHE_PATH):
    pretrained_emb, word2id = torch.load(CACHE_PATH)
elif WORD_EMB_PATH is not None:
    pretrained_emb = load_emb(word2id, WORD_EMB_PATH)
    torch.save((pretrained_emb, word2id), CACHE_PATH)
else:
    pretrained_emb = None

model = LFLSTM(input_sizes, hidden_sizes, fc1_size, output_size, dropout)
if pretrained_emb is not None:
    model.embed.weight.data = pretrained_emb
model.embed.requires_grad = False
optimizer = Adam([param for param in model.parameters() if param.requires_grad], weight_decay=weight_decay)

if CUDA:
    model.cuda()
criterion = nn.L1Loss(reduction='sum')
criterion_test = nn.L1Loss(reduction='sum')
best_valid_loss = float('inf')
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
lr_scheduler.step() # for some reason it seems the StepLR needs to be stepped once first
train_losses = []
valid_losses = []
for e in range(MAX_EPOCH):
    model.train()
    train_iter = tqdm_notebook(train_loader)
    train_loss = 0.0
    for batch in train_iter:
        model.zero_grad()
        t, v, a, y, l = batch
        batch_size = t.size(0)
        if CUDA:
            t = t.cuda()
            v = v.cuda()
            a = a.cuda()
            y = y.cuda()
            l = l.cuda()
        y_tilde = model(t, v, a, l)
        loss = criterion(y_tilde, y)
        loss.backward()
        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
        optimizer.step()
        train_iter.set_description(f"Epoch {e}/{MAX_EPOCH}, current batch loss: {round(loss.item()/batch_size, 4)}")
        train_loss += loss.item()
    train_loss = train_loss / len(train)
    train_losses.append(train_loss)
    print(f"Training loss: {round(train_loss, 4)}")

    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        for batch in dev_loader:
            model.zero_grad()
            t, v, a, y, l = batch
            if CUDA:
                t = t.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            y_tilde = model(t, v, a, l)
            loss = criterion(y_tilde, y)
            valid_loss += loss.item()
    
    valid_loss = valid_loss/len(dev)
    valid_losses.append(valid_loss)
    print(f"Validation loss: {round(valid_loss, 4)}")
    print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
    if valid_loss <= best_valid_loss:
        best_valid_loss = valid_loss
        print("Found new best model on dev set!")
        torch.save(model.state_dict(), 'model.std')
        torch.save(optimizer.state_dict(), 'optim.std')
        curr_patience = patience
    else:
        curr_patience -= 1
        if curr_patience <= -1:
            print("Running out of patience, loading previous best model.")
            num_trials -= 1
            curr_patience = patience
            model.load_state_dict(torch.load('model.std'))
            optimizer.load_state_dict(torch.load('optim.std'))
            lr_scheduler.step()
            print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
    
    if num_trials <= 0:
        print("Running out of patience, early stopping.")
        break

model.load_state_dict(torch.load('model.std'))
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for batch in test_loader:
        model.zero_grad()
        t, v, a, y, l = batch
        if CUDA:
            t = t.cuda()
            v = v.cuda()
            a = a.cuda()
            y = y.cuda()
            l = l.cuda()
        y_tilde = model(t, v, a, l)
        loss = criterion_test(y_tilde, y)
        y_true.append(y_tilde.detach().cpu().numpy())
        y_pred.append(y.detach().cpu().numpy())
        test_loss += loss.item()
print(f"Test set performance: {test_loss/len(test)}")
y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)
                  
y_true_bin = y_true >= 0
y_pred_bin = y_pred >= 0
bin_acc = accuracy_score(y_true_bin, y_pred_bin)
print(f"Test set accuracy is {bin_acc}")
# -


