import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

import torch.nn.functional as F
from tqdm import tqdm
import config
import argparse

# # Load model
# checkpoint = torch.load(config.checkpoint)
# decoder = checkpoint['decoder']
# # decoder = decoder.to(config.device)   # no GPU
# decoder.eval()
# encoder = checkpoint['encoder']
# # encoder = encoder.to(config.device)   # no GPU
# encoder.eval()
#
# # Load word map (word2ix)
# with open(config.word_map_file, 'r') as j:
#     word_map = json.load(j)
# rev_word_map = {v: k for k, v in word_map.items()}
# vocab_size = len(word_map)
#
# # Normalization transform
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])


def evaluate(beam_size, model_path):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # Load model
    if not model_path:
        model_path = config.checkpoint
    checkpoint = torch.load(model_path)
    decoder = checkpoint['decoder']
    # decoder = decoder.to(config.device)   # no GPU
    decoder.eval()
    encoder = checkpoint['encoder']
    # encoder = encoder.to(config.device)   # no GPU
    encoder.eval()
    epoch = checkpoint['epoch']

    # Load word map (word2ix)
    with open(config.word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    # Normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(config.data_folder, config.data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size

        # Move to GPU device, if available
        # image = image.to(config.device)  # (1, 3, 256, 256)   # no GPU

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        # k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(config.device)  # (k, 1)  no GPU
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        # top_k_scores = torch.zeros(k, 1).to(config.device)  # (k, 1)   no GPU
        top_k_scores = torch.zeros(k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            # debugging https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/18
            smth_wrong = False
            if step > 50:
                smth_wrong = True
                break
            step += 1

        if smth_wrong is not True:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        else:
            seq = seqs[0][:config.max_cap_len]
            hypotheses.append([w.item() for w in seq if w.item() not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}] + [word_map['<end>']])

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # # Hypotheses
        # hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    # Calculate pycocoevalcap scores
    gts, res = format_for_metrics(references, hypotheses, rev_word_map)
    blue_scores, _ = Bleu(4).compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)
    rouge_score, _ = Rouge().compute_score(gts, res)
    # spice_score = Spice().compute_score(gts, res)
    spice_score = 'awaiting'
    return bleu4, blue_scores, cider_score, spice_score, rouge_score, epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Evaluation')
    parser.add_argument('--model', '-m', help='path to model')
    args = parser.parse_args()

    beam_size = 1
    metrics = evaluate(beam_size, args.model)
    print("\n\nEval on model {}".format(args.model if args.model else config.checkpoint))
    print('Epoch {}'.format(metrics[5]))
    print("BLEU-4 score @ beam size of %d is %.4f." % (beam_size, metrics[0]))
    print('pycocoevalcap evaluation:')
    print('\tBlue 1 is {:.4f}\tBlue 2 is {:.4f}\tBlue 3 is {:.4f}\tBlue 4 is {:.4f}'.format(*metrics[1]))
    print('\tCider is {:.4f}\t\tRouge is {:.4f}'.format(metrics[2], metrics[4]))
    log_f = open(config.eval_log_path, 'a+', encoding='utf-8')
    log_f.write("\n------------------------\nEval on model {}".format(args.model if args.model else config.checkpoint) + '\n')
    log_f.write('Epoch {}\n'.format(metrics[5]))
    log_f.write("BLEU-4 score @ beam size of %d is %.4f." % (beam_size, metrics[0]) + '\n')
    log_f.write("pycocoevalcap evaluation:" + '\n')
    log_f.write('\tblue 1 is {:.4f}\tblue 2 is {:.4f}\tblue 3 is {:.4f}\tblue 4 is {:.4f}'.format(*metrics[1]) + '\n')
    log_f.write('\tCider is {:.4f}\t\tRouge is {:.4f}'.format(metrics[2], metrics[4]) + '\n')
    log_f.close()
