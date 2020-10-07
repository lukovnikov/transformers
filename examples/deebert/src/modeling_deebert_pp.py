import time
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    BertEmbeddings,
    BertLayer,
    BertPooler,
    BertPreTrainedModel,
)


def entropy(x):
    """Calculate entropy of a pre-softmax logit Tensor"""
    x_sm = torch.softmax(x, -1)
    # A = torch.sum(exp_x, dim=1)  # sum of exp(x_i)
    # B = torch.sum(x * exp_x, dim=1)  # sum of x_i * exp(x_i)
    ret = x_sm * torch.log(x_sm)
    ret = - ret.sum(-1)
    # return torch.log(A) - B / A
    return ret


class DeeBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.early_exits = nn.ModuleList([BertExit(config) for _ in range(config.num_hidden_layers)])

        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]
        self.deploymode = False

    def set_early_exit_entropy(self, x):
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x

    # def init_highway_pooler(self, pooler):
    #     loaded_model = pooler.state_dict()
    #     for early_exit in self.early_exits:
    #         for name, param in early_exit.pooler.state_dict().items():
    #             param.copy_(loaded_model[name])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        all_logits = ()
        cum_logits = ()
        all_entropies = ()
        tocs = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            current_outputs = (hidden_states,)
            if self.output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if self.output_attentions:
                current_outputs = current_outputs + (all_attentions,)

            early_exit = self.early_exits[i](current_outputs)
            exit_logits = early_exit[0]
            # logits, pooled_output
            all_logits = all_logits + (exit_logits,)
            if len(cum_logits) == 0:
                cum_logit = exit_logits
            else:
                cum_logit = cum_logits[-1] + exit_logits
            cum_logits = cum_logits + (cum_logit,)

            exit_entropy = entropy(cum_logits[-1]).mean()
            all_entropies = all_entropies + (exit_entropy,)

            tocs = tocs + (time.perf_counter(),)

            if self.deploymode:
                if exit_entropy < self.early_exit_entropy[i]:
                    break

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        outputs = (all_logits, cum_logits, all_entropies, tocs) + outputs
        return outputs  # last-layer hidden state, (all hidden states), (all attentions), all highway exits


@add_start_docstrings(
    "The Bert Model transformer with early exiting (DeeBERT). ",
    BERT_START_DOCSTRING,
)
class DeeBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = DeeBertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    # def init_highway_pooler(self):
    #     # pass
    #     self.encoder.init_highway_pooler(self.pooler)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        r"""
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.

                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
            highway_exits (:obj:`tuple(tuple(torch.Tensor))`:
                Tuple of each early exit's results (total length: number of layers)
                Each tuple is again, a tuple of length 2 - the first entry is logits and the second entry is hidden states.
        """
        tic = time.perf_counter()
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        # sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)

        all_logits, cum_logits, all_entropies, tocs = encoder_outputs[:4]
        encoder_outputs = encoder_outputs[4:]

        time_at_exit = [toc - tic for toc in tocs]

        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:-1]  # add hidden_states and attentions if they are here
        outputs = (all_logits, cum_logits, all_entropies, time_at_exit) + encoder_outputs
        return outputs
        # return outputs  # sequence_output, pooled_output, (hidden_states), (attentions), highway exits


class BertExit(nn.Module):
    """A module to provide a shortcut
    from (the output of one non-final BertLayer in BertEncoder) to (cross-entropy computation in BertForSequenceClassification)
    """

    def __init__(self, config):
        super().__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):
        # Pooler
        pooler_input = encoder_outputs[0]
        pooler_output = self.pooler(pooler_input)
        # "return" pooler_output

        # BertModel
        bmodel_output = (pooler_input, pooler_output) + encoder_outputs[1:]
        # "return" bodel_output

        # Dropout and classification
        pooled_output = bmodel_output[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output


class SmoothedCELoss(torch.nn.Module):
    """ CrossEntropyLoss with label smoothing. """
    def __init__(self, reduction="mean", ignore_index=-100, smoothing=0., mode="logits", weight=None, **kw):
        super(SmoothedCELoss, self).__init__(**kw)
        self.reduction, self.ignore_indices, self.smoothing = reduction, ignore_index, smoothing
        self.mode = mode        # "logits", "probs", "logprobs"
        self.kl = torch.nn.KLDivLoss(reduction="none")
        self.sm = torch.nn.LogSoftmax(-1) if self.mode == "logits" else None
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    @staticmethod
    def get_ignore_mask(gold, ignore_indices):
        if ignore_indices is not None and not isinstance(ignore_indices, (list, tuple, set)):
            ignore_indices = [ignore_indices]
        mask = None     # (batsize,)
        if ignore_indices is not None:
            for ignore in ignore_indices:
                mask_i = (gold != ignore)   # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask & mask_i
        if mask is None:
            mask = torch.ones_like(gold).byte()
        return mask

    def forward(self, probs, gold):
        """
        :param probs:   (batsize, ..., vocsize) logits
        :param gold:    (batsize, ..., ) int ids of correct class
        :return:
        """
        _prob_mask_crit = -np.infty if self.mode in "logits logprobs".split() else 0
        lsv = self.smoothing   # get value of label smoothing hyperparam
        assert(lsv >= 0 and lsv <= 1)
        prob_mask = (probs > _prob_mask_crit).float()     # (batsize, ..., vocsize) where probs are > 0, reverse engineering a -infty mask applied outside
        prob_mask_weights = lsv / prob_mask.sum(-1, keepdim=True)
        _gold = torch.ones_like(probs) * prob_mask_weights * prob_mask
        _gold.scatter_(-1, gold.unsqueeze(-1), (1 - lsv) + prob_mask_weights)   # (batsize, ..., vocsize) probs
        assert(torch.allclose(_gold.sum(-1), torch.ones_like(gold).float()) is True)

        logprobs = self.sm(probs) if self.mode == "logits" else (probs if self.mode == "logprobs" else torch.log(probs))
        kl_divs = self.kl(logprobs, _gold.detach())
        # kl_divs = inf2zero(kl_divs)
        kl_div = kl_divs.sum(-1)        # (batsize, ...) kl div per element

        if self.weight is not None:
            kl_div = kl_div * self.weight[gold]

        mask = self.get_ignore_mask(gold, self.ignore_indices).float()
        kl_div = kl_div * mask
        ret = kl_div.sum()
        if self.reduction in ["elementwise_mean", "mean"]:
            total = mask.sum()
            ret = ret / total
        elif self.reduction == "none":
            ret = kl_div
        return ret


@add_start_docstrings(
    """Bert Model (with early exiting - DeeBERT) with a classifier on top,
    also takes care of multi-layer training. """,
    BERT_START_DOCSTRING,
)
class DeeBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, smoothing=0.):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        self.bert = DeeBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()
        self.smoothing = smoothing

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_layer=-1,
        train_exit=0.,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification (or regression if config.num_labels==1) loss.
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
            highway_exits (:obj:`tuple(tuple(torch.Tensor))`:
                Tuple of each early exit's results (total length: number of layers)
                Each tuple is again, a tuple of length 2 - the first entry is logits and the second entry is hidden states.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # sequence_output, pooled_output, (hidden_states), (attentions), highway exits

        # pooled_output = outputs[1]

        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        all_logits, cum_logits, all_entropies, times = outputs[:4]
        outputs = outputs[4:]
        # outputs = (logits,) + outputs  # add hidden states and attention if they are here
        exit_layer = len(times)

        if labels is not None:
            exit_losses = []
            # exit_accs = []
            for exit_logit in all_logits:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    exit_loss = loss_fct(exit_logit, labels)
                else:
                    loss_fct = CrossEntropyLoss()
                    exit_loss = loss_fct(exit_logit, labels)
                exit_losses.append(exit_loss)

            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                cum_loss = loss_fct(cum_logits[-1], labels)
            else:
                loss_fct = SmoothedCELoss(smoothing=self.smoothing)
                cum_loss = loss_fct(cum_logits[-1], labels)

            loss = cum_loss
            # loss = exit_losses[-1]
            earlyloss = sum(exit_losses) * float(train_exit)
            # earlyloss = sum(exit_losses[:-1]) * float(train_exit)

            out_logits = cum_logits

            outputs = (loss + earlyloss, earlyloss, out_logits, all_entropies, times) + outputs
        else:
            # assert we're in deployment mode
            outputs = (all_logits[-1], exit_layer, all_entropies, times) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions), (highway_exits)
