#!/usr/bin/env python
""" Translator Class and builder """
import codecs
import os
import math
import shutil
import time

import torch

def build_predictor(args, tokenizer, symbols, model, logger=None):
    # we should be able to refactor the global scorer a lot
    scorer = GNMTGlobalScorer(args.alpha, length_penalty="wu")
    translator = Translator(
        args, model, tokenizer, symbols, global_scorer=scorer, logger=logger
    )
    return translator


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, length_penalty):
        self.alpha = alpha
        penalty_builder = PenaltyBuilder(length_penalty)
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam, logprobs, self.alpha)
        return normalized_probs


class PenaltyBuilder(object):
    """
    Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen
    """

    def __init__(self, length_pen):
        self.length_pen = length_pen

    def length_penalty(self):
        if self.length_pen == "wu":
            return self.length_wu
        elif self.length_pen == "avg":
            return self.length_average
        else:
            return self.length_none

    """
    Below are all the different penalty terms implemented so far
    """

    def length_wu(self, beam, logprobs, alpha=0.0):
        """
        NMT length re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        modifier = ((5 + len(beam.next_ys)) ** alpha) / ((5 + 1) ** alpha)
        return logprobs / modifier

    def length_average(self, beam, logprobs, alpha=0.0):
        """
        Returns the average probability of tokens in a sequence.
        """
        return logprobs / len(beam.next_ys)

    def length_none(self, beam, logprobs, alpha=0.0, beta=0.0):
        """
        Returns unmodified scores.
        """
        return logprobs


class Translator(object):
    """
    Uses a model to translate a batch of sentences.

    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(
        self, args, model, vocab, symbols, global_scorer=None, logger=None
    ):
        self.logger = logger
        self.cuda = args.visible_gpus != "-1"

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols["BOS"]
        self.end_token = symbols["EOS"]

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

    def translate(self, data_iter, step, attn_debug=False):
        self.model.eval()

        #
        # Define output paths
        #

        # gold_path = self.args.result_path + ".%d.gold" % step
        # can_path = self.args.result_path + ".%d.candidate" % step
        # self.gold_out_file = codecs.open(gold_path, "w", "utf-8")
        # self.candidate_out_file = codecs.open(can_path, "w", "utf-8")

        # self.gold_out_file = codecs.open(gold_path, "w", "utf-8")
        # self.candidate_out_file = codecs.open(can_path, "w", "utf-8")

        # raw_src_path = self.args.result_path + ".%d.raw_src" % step
        # self.src_out_file = codecs.open(raw_src_path, "w", "utf-8")

        with torch.no_grad():
            for batch in data_iter:
                if self.args.recall_eval:
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src = trans
                    pred_str = (
                        pred.replace("[unused0]", "")
                        .replace("[unused3]", "")
                        .replace("[PAD]", "")
                        .replace("[unused1]", "")
                        .replace(r" +", " ")
                        .replace(" [unused2] ", "<q>")
                        .replace("[unused2]", "")
                        .strip()
                    )
                    gold_str = gold.strip()
                    if self.args.recall_eval:
                        _pred_str = ""
                        for sent in pred_str.split("<q>"):
                            can_pred_str = _pred_str + "<q>" + sent.strip()
                            if len(can_pred_str.split()) >= len(gold_str.split()) + 10:
                                pred_str = _pred_str
                                break
                            else:
                                _pred_str = can_pred_str

                    print(src, pred_str, gold_str)
                    # pred_str = ' '.join(pred_str.split()[:len(gold_str.split())])
                    # self.raw_candidate_out_file.write(' '.join(pred).strip() + '\n')
                    # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                    # self.candidate_out_file.write(pred_str + "\n")
                    # self.gold_out_file.write(gold_str + "\n")
                    # self.src_out_file.write(src.strip() + "\n")
                    # ct += 1
                # self.candidate_out_file.flush()
                # self.gold_out_file.flush()
                # self.src_out_file.flush()

        # self.candidate_out_file.close()
        # self.gold_out_file.close()
        # self.src_out_file.close()

        # if step != -1:
            # rouges = self._report_rouge(gold_path, can_path)
            # self.logger.info(
                # "Rouges at step %d \n%s" % (step, rouge_results_to_str(rouges))
            # )

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch, self.max_length, min_length=self.min_length
            )

    # Where the beam search lives
    # I have no idea why it is being called from the method above
    def _fast_translate_batch(self, batch, max_length, min_length=0):
        """ Beam Search using the encoder inputs contained in `batch`.
        """
        
        # The batch object is funny
        # Instead of just looking at the size of the arguments we encapsulate
        # a size argument.
        # Where is it defined?
        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src

        src_features = self.model.bert(src, segs, mask_src)
        dec_states = self.model.decoder.init_decoder_state(
            src, src_features, with_cache=True
        )
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device
        )
        alive_seq = torch.full(
            [batch_size * beam_size, 1], self.start_token, dtype=torch.long, device=device
        )

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=device
        ).repeat(batch_size)

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states = self.model.decoder(
                decoder_input, src_features, dec_states, step=step
            )

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = " ".join(words).replace(" ##", "").split()
                        if len(words) <= 3:
                            continue
                        trigrams = [
                            (words[i - 1], words[i], words[i + 1])
                            for i in range(1, len(words) - 1)
                        ]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = topk_beam_index + beam_offset[
                : topk_beam_index.size(0)
            ].unsqueeze(1)
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
            )

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(
                    -1, alive_seq.size(-1)
                )
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices)
            )

        return results

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert len(translation_batch["gold_score"]) == len(translation_batch["predictions"])
        batch_size = batch.batch_size

        preds, _, _, tgt_str, src = (
            translation_batch["predictions"],
            translation_batch["scores"],
            translation_batch["gold_score"],
            batch.tgt_str,
            batch.src,
        )

        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = " ".join(pred_sents).replace(" ##", "")
            gold_sent = " ".join(tgt_str[b].split())
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = " ".join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            translations.append(translation)

        return translations

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


#
# All things ROUGE. Uses `pyrouge` which is a hot mess.
#


def test_rouge(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )
