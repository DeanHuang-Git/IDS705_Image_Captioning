import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *

from nltk.translate.bleu_score import corpus_bleu

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map
    best_bleu4 = config.best_bleu4
    epochs_since_improvement = config.epochs_since_improvement
    checkpoint = config.checkpoint
    start_epoch = config.start_epoch
    fine_tune_encoder = config.fine_tune_encoder
    data_name = config.data_name
    checkpoint = config.checkpoint

    log_f = open(config.train_log_path, 'a+', encoding='utf-8')

    # Read word map
    word_map_file = os.path.join(config.data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        print('no checkpoint, rebuild')
        log_f.write('\n\nno checkpoint, rebuild' + '\n')
        decoder = DecoderWithAttention(attention_dim=config.attention_dim,
                                       embed_dim=config.emb_dim,
                                       decoder_dim=config.decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=config.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=config.decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=config.encoder_lr) if fine_tune_encoder else None

    else:
        print('checkpoint exist,continue.. \n{}'.format(checkpoint))
        log_f.write('\n\ncheckpoint exist,continue.. \n{}'.format(checkpoint) + '\n')
        log_f.close()
        checkpoint = torch.load(checkpoint, map_location=config.device)  # map_location=device for cpu
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:     # check for fine tuning
            encoder.fine_tune(fine_tune_encoder)        # change requires_grad for weights
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=config.encoder_lr)

    # Move to GPU, if available
    # decoder = decoder.to(config.device)   # no GPU
    # encoder = encoder.to(config.device)   # no GPU

    # Loss function
    # criterion = nn.CrossEntropyLoss().to(config.device)   # no GPU
    criterion = nn.CrossEntropyLoss()

    # Custom batch dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 这里是原ResNet的mean和std
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(config.data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),  # CaptionDataset is in datasets.py
        batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(config.data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True)

    # Epochs
    val_writer = SummaryWriter(log_dir=config.tensorboard_path + '/val/' + time.strftime('%m-%d_%H%M', time.localtime()))   # for tensorboard
    for epoch in range(start_epoch, config.epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)        # utils.py
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                writer=val_writer,
                                epoch=epoch)

        # Check if there was an improvement, check each epoch
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        log_f = open(config.train_log_path, 'a+', encoding='utf-8')
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            log_f.write("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,) + '\n')
        else:
            epochs_since_improvement = 0
            log_f.write('\n')
        log_f.close()
        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)

    val_writer.close()


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    log_f = open(config.train_log_path, 'a+', encoding='utf-8')
    writer = SummaryWriter(log_dir=config.tensorboard_path + '/train/' + time.strftime('%m-%d_%H%M', time.localtime()))
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    with tqdm(total = len(train_loader)) as pbar:       # adding progress bar
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            # imgs = imgs.to(config.device)   # no GPU
            # caps = caps.to(config.device)   # no GPU
            # caplens = caplens.to(config.device)   # no GPU

            # Forward prop.
            imgs = encoder(imgs)        # imgs: encoded image features
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            # score shape: (batch_size, max(decode_lengths), vocab_size)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data    # score shape: (batch_size, count of words in index, vocab_size)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += config.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if config.grad_clip is not None:
                clip_gradient(decoder_optimizer, config.grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, config.grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)     # utils.py
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % config.print_freq == 0:  # print freq is based on how many batches
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              top5=top5accs))

                log_f.write('Epoch: [{0}][{1}/{2}]\t'
                            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                    batch_time=batch_time,
                                                                                    data_time=data_time, loss=losses,
                                                                                    top5=top5accs) + '\n')
                writer.add_scalar("loss/train", losses.val, i)
                writer.add_scalar("acc/train", top5accs.val, i)
            pbar.update(1)

    log_f.close()
    writer.close()


def validate(val_loader, encoder, decoder, criterion, writer, epoch):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    log_f = open(config.train_log_path, 'a+', encoding='utf-8')

    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            # imgs = imgs.to(config.device)   # no GPU
            # caps = caps.to(config.device)   # no GPU
            # caplens = caplens.to(config.device)   # no GPU

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += config.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)      # https://stackoverflow.com/questions/37668902/evaluation-calculate-top-n-accuracy-top-1-and-top-5#:~:text=Top%2D5%20accuracy%20means%20that,Tiger%3A%200.4
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % config.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

                log_f.write('Validation: [{0}/{1}]\t'
                            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                      batch_time=batch_time,
                                                                                      loss=losses,
                                                                                      top5=top5accs) + '\n')

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

        log_f.write('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4) + '\n')

        writer.add_scalar('loss/val', losses.val, epoch)
        writer.add_scalar('acc/val', top5accs.val, epoch)

    log_f.close()
    return bleu4


if __name__ == '__main__':
    main()
