import argparse
from collections import namedtuple
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, SequentialSampler

from model_bertabs import BertAbsSummarizer, build_predictor
from transformers import BertTokenizer

from utils_summarization import (
    CNNDailyMailDataset,
    encode_for_summarization,
    build_mask,
    fit_to_block_size,
    compute_token_type_ids,
)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


Batch = namedtuple("Batch", ["batch_size", "src", "segs", "mask_src", "tgt_str"])


def evaluate(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = get_pretrained_BertAbs_model("bert-ext-abs.pt", device=args.device)

    symbols = {
        "BOS": tokenizer.vocab["[unused0]"],
        "EOS": tokenizer.vocab["[unused1]"],
        "PAD": tokenizer.vocab["[PAD]"],
    }

    # these (unused) arguments are defined to keep the compatibility
    # with the legacy code and will be deleted in a next iteration.
    args.result_path = ""
    args.temp_dir = ""

    data_iterator = build_data_iterator(args, tokenizer)
    predictor = build_predictor(args, tokenizer, symbols, model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(data_iterator))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("")
    logger.info("***** Beam Search parameters *****")
    logger.info("  Beam size = %d", args.beam_size)
    logger.info("  Minimum length = %d", args.min_length)
    logger.info("  Maximum length = %d", args.max_length)
    logger.info("  Alpha (length penalty) = %.2f", args.alpha)
    logger.info("  Trigrams %s be blocked", ("will" if args.block_trigram else "will NOT"))

    for batch in data_iterator:
        batch_data = predictor.translate_batch(batch)
        translations = predictor.from_batch(batch_data)
        summaries = [format_summary(t) for t in translations]
        print(summaries)


def format_summary(translation):
    """ Transforms the output of the `from_batch` function
    into nicely formatted summaries.
    """
    raw_summary, _, _ = translation
    summary = (
        raw_summary
        .replace("[unused0]", "")
        .replace("[unused3]", "")
        .replace("[PAD]", "")
        .replace("[unused1]", "")
        .replace(r" +", " ")
        .replace(" [unused2] ", "<q> ")
        .replace("[unused2]", "")
        .strip()
    )

    return summary

#
# BUILD the model
#


def get_pretrained_BertAbs_model(path, device):
    BertAbsConfig = namedtuple(
        "BertAbsConfig",
        [
            "temp_dir",
            "large",
            "finetune_bert",
            "encoder",
            "share_emb",
            "max_pos",
            "enc_layers",
            "enc_hidden_size",
            "enc_heads",
            "enc_ff_size",
            "enc_dropout",
            "dec_layers",
            "dec_hidden_size",
            "dec_heads",
            "dec_ff_size",
            "dec_dropout",
        ],
    )

    config = BertAbsConfig(
        temp_dir=".",
        finetune_bert=False,
        large=False,
        share_emb=True,
        encoder="bert",
        max_pos=512,
        enc_layers=6,
        enc_hidden_size=512,
        enc_heads=8,
        enc_ff_size=512,
        enc_dropout=0.2,
        dec_layers=6,
        dec_hidden_size=768,
        dec_heads=8,
        dec_ff_size=2048,
        dec_dropout=0.2,
    )
    checkpoints = torch.load(path, lambda storage, loc: storage)
    bertabs = BertAbsSummarizer.from_pretrained(checkpoints, config, device)
    bertabs.eval()

    return bertabs


#
# LOAD the dataset
#


def build_data_iterator(args, tokenizer):
    dataset = load_and_cache_examples(args, tokenizer)
    sampler = SequentialSampler(dataset)
    collate_fn = lambda data: collate(data, tokenizer, block_size=512)
    iterator = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn,
    )

    return iterator


def load_and_cache_examples(args, tokenizer):
    dataset = CNNDailyMailDataset(args.documents_dir)
    return dataset


def collate(data, tokenizer, block_size):
    """ Collate formats the data passed to the data loader.

    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    # remove the files with empty an story/summary, encode and fit to block
    data = [x for x in data if not (len(x[0]) == 0 or len(x[1]) == 0)]
    data = filter(lambda x: not (len(x[0]) == 0 or len(x[1]) == 0), data)
    data = [encode_for_summarization(story, summary, tokenizer) for story, summary in data]
    data = [
        (
            fit_to_block_size(story, block_size, tokenizer.pad_token_id),
            fit_to_block_size(summary, block_size, tokenizer.pad_token_id),
        )
        for story, summary in data
    ]

    stories = torch.tensor([story for story, summary in data])
    encoder_token_type_ids = compute_token_type_ids(stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(stories, tokenizer.pad_token_id)

    batch = Batch(
        batch_size=len(stories),
        src=stories,
        segs=encoder_token_type_ids,
        mask_src=encoder_mask,
        tgt_str=[""] * len(stories),
    )

    return batch


def decode_summary(summary_tokens, tokenizer):
    """ Decode the summary and return it in a format
    suitable for evaluation.
    """
    summary_tokens = summary_tokens.to("cpu").numpy()
    summary = tokenizer.decode(summary_tokens)
    sentences = summary.split(".")
    sentences = [s + "." for s in sentences]
    return sentences


def main():
    """ The main function defines the interface with the users.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents_dir",
        default=None,
        type=str,
        required=True,
        help="The folder where the documents to summarize are located.",
    )
    parser.add_argument(
        "--summaries_output_dir",
        default=None,
        type=str,
        required=True,
        help="The folder in wich the summaries should be written.",
    )
    # EVALUATION options
    parser.add_argument(
        "--visible_gpus",
        default=-1,
        type=int,
        help="Number of GPUs with which to do the training.",
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    # BEAM SEARCH arguments
    parser.add_argument(
        "--min_length",
        default=50,
        type=int,
        help="Minimum number of tokens for the summaries.",
    )
    parser.add_argument(
        "--max_length",
        default=200,
        type=int,
        help="Maixmum number of tokens for the summaries.",
    )
    parser.add_argument(
        "--beam_size",
        default=5,
        type=int,
        help="The number of beams to start with for each example.",
    )
    parser.add_argument(
        "--alpha",
        default=0.95,
        type=float,
        help="The value of alpha for the length penalty in the beam search.",
    )
    parser.add_argument(
        "--block_trigram",
        default=True,
        type=bool,
        help="Whether to block the existence of repeating trigrams in the text generated by beam search.",
    )
    args = parser.parse_args()
    args.device = torch.device("cpu") if args.visible_gpus == -1 else torch.device("cuda")

    if not documents_dir_is_valid(args.documents_dir):
        raise FileNotFoundError(
            "We could not find the directory you specified for the documents to summarize, or it was empty. Please specify a valid path."
        )
    maybe_create_output_dir(args.summaries_output_dir)

    evaluate(args)


def documents_dir_is_valid(path):
    if not os.path.exists(path):
        return False

    file_list = os.listdir(path)
    if len(file_list) == 0:
        return False

    return True


def maybe_create_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    main()
