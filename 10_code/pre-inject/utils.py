import os
import numpy as np
import pandas as pd
import re
import h5py
import json
import torch
import random
from cv2 import imread, resize
from matplotlib.pyplot import imread
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
# pycocoevalcap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import os
import time
import shutil
import config


def txt_2_json():
    df = pd.read_csv(config.caption_txt_path)
    df.rename(columns={'caption': 'raw', 'image': 'filename'}, inplace=True)

    # make tokens
    df['tokens'] = df['raw'].apply(lambda x: x.split())
    # add imgid, sentid
    df['sentid'] = list(df.index)
    df['imgid'] = [i // 5 for i in list(df.index)]
    # total image count
    N = len(df) // 5

    # random select train, test, split
    seed(705)
    total_imgid = [i for i in range(N)]
    rest_imgid = total_imgid
    TRAIN_RATE = config.train_rate
    VAL_RATE = config.val_rate
    train_imgid = random.sample(total_imgid, int(TRAIN_RATE * len(total_imgid)))
    rest_imgid = list(set(rest_imgid).difference(set(train_imgid)))
    val_imgid = random.sample(rest_imgid, int(VAL_RATE * len(total_imgid)))
    test_imgid = list(set(rest_imgid).difference(set(val_imgid)))

    # get images and corresponding captions
    images = []
    for imgid in range(N):
        dic = {}
        dic['imgid'] = imgid
        dic['filename'] = df.loc[df['imgid'] == imgid, 'filename'].tolist()[0]
        if imgid in train_imgid:
            dic['split'] = 'train'
        elif imgid in test_imgid:
            dic['split'] = 'test'
        else:
            dic['split'] = 'val'
        dic['sentences'] = []

        sent_ids = [i for i in range(5 * imgid, 5 * imgid + 5)]
        dic['sentid'] = sent_ids
        for sentid in sent_ids:
            sent_dic = {'imgid': imgid, 'sentid': sentid}
            raw_sent = df.loc[df['sentid'] == sentid, 'raw'].tolist()[0]
            sent_dic['raw'] = raw_sent
            sent_dic['tokens'] = re.findall(r"[\w']+", raw_sent)

            dic['sentences'].append(sent_dic.copy())

        images.append(dic.copy())

        # final json & cache
        data = {'images':images, 'dataset': config.dataset_name}
        with open(config.caption_json_path, 'w') as f:
            f.write(json.dumps(data))


def create_input_files():
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param data_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # Read Karpathy JSON
    with open(config.caption_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])   # c['tokens'] are words list split from raw sentences without punctuations
            if len(c['tokens']) <= config.max_cap_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(config.image_folder, img['filepath'], img['filename']) if config.dataset_name == 'coco' else os.path.join(
            config.image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > config.min_word_freq]   # only keep words that appear more than min_word_freq
    word_map = {k: v + 1 for v, k in enumerate(words)}  # generate the word index
    # adding special tags also into word dict
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    # dataset: "flickr8k, captions_per_image:5 each image we have 5 sentences,
    base_filename = config.data_name

    # Save word map to a JSON, if existing, use already existed
    if not os.path.isfile(config.word_map_file):
        with open(config.word_map_file, 'w') as j:
            json.dump(word_map, j)      # get "WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json"
    else:
        with open(config.word_map_file, 'r') as j:
            word_map= json.load(j, strict=False)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(705)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(config.data_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = config.captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):    # add progress bar

                # Sample captions
                if len(imcaps[i]) < config.captions_per_image:     # for each image, just 5 sentences are used for training
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(config.captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=config.captions_per_image)

                # Sanity check
                assert len(captions) == config.captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]     # np.newaxis add a new dimension
                    img = np.concatenate([img, img, img], axis=2)
                img = resize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)       # 3 channels, 256 x 256
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions  transfer all caption words into words dict
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (config.max_cap_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * config.captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(config.data_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(config.data_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = config.output_folder + '/checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    torch.save(state, '{}/checkpoint_{}_epoch[{}].pth.tar'.format(config.output_folder, time.strftime('%m%d_%H', time.localtime()), epoch))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        record_old_model()
        torch.save(state, config.checkpoint)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)      # ind standing for the indexes of top k scores, corresponding to the words
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def record_old_model():
    """
    move previous best model to history folder to avoid be rewrite and loss
    :return:
    """
    old_path = config.checkpoint
    if old_path and os.path.isfile(old_path):   # only having previous checkpoints and existing in the specific directory
        mtime = os.stat(old_path).st_mtime
        file_modify_time = time.strftime('%m%d_%H', time.localtime(mtime))
        new_model_name = "BEST_{}.pth.tar".format(file_modify_time)
        new_path = config.past_model_path + '/' + new_model_name
        shutil.move(old_path, new_path)
        print('already record old model {} to {}\n'.format(old_path, new_path))


def format_for_metrics(gts, res, rev_word_map):
    """
    to generate appropriate format to use pycocoevalcap
    :param gts: groud truth list
    :param res: hypothesis list
    :param rev_word_map: reverse word map, from idxes to character
    :return:
    """
    gts_dic = {}
    for idx, sents in enumerate(gts):
       tmp = []
       for sent in sents:
           tmp.append({u'image_id': idx, u'caption':' '.join([rev_word_map[x] for x in sent])})
       gts_dic[idx] = tmp[:]

    res_dic = {}
    for idx, sent in enumerate(res):
       res_dic[idx] = [{u'image_id': idx, u'caption': ' '.join([rev_word_map[x] for x in sent])}]

    tokenizer = PTBTokenizer()
    return tokenizer.tokenize(gts_dic), tokenizer.tokenize(res_dic)