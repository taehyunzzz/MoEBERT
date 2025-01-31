import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_bert import (
    BertAttention,
    BertEmbeddings,
    BertEncoder,
    BertIntermediate,
    BertLayer,
    BertModel,
    BertOutput,
    BertPooler,
    BertPreTrainedModel,
)
from ...modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput
from ...moebert.utils import (
    FeedForward,
    DiffFeedForward,
    ImportanceProcessor,
    MoEModelOutput,
    MoEModelOutputWithPooling,
    use_experts,
)
from ...moebert.moe_layer import (
    MoELayer,
)
from ...utils import logging

logger = logging.get_logger(__name__)


def symmetric_KL_loss(p, q):
    """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
    p, q = p.float(), q.float()
    loss = (p - q) * (torch.log(p) - torch.log(q))
    return 0.5 * loss.sum()


def softmax(x):
    return F.softmax(x, dim=-1, dtype=torch.float32)

class MoEBertLayer(BertLayer):
    def __init__(self, config, layer_idx=-100):
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        # construct experts
        self.use_experts = use_experts(layer_idx)
        dropout = config.moebert_expert_dropout if self.use_experts else config.hidden_dropout_prob
        if self.use_experts:
            if config.moebert == "diffmoe" :
                ffn = DiffFeedForward(
                    config              = config                            ,
                    intermediate_size   = config.intermediate_size          ,
                    shared_size         = config.moebert_share_importance   ,
                    dropout             = dropout                           ,
                    fixmask_init        = config.moebert_fixmask_init       ,
                    alpha_init          = config.moebert_alpha_init         ,
                    concrete_lower      = config.moebert_concrete_lower     ,
                    concrete_upper      = config.moebert_concrete_upper     ,
                    structured          = config.moebert_structured         ,
                    sparsity_pen        = config.moebert_sparsity_pen       ,
                )
            else :
                ffn = FeedForward(config, config.moebert_expert_dim, dropout)
            self.experts = MoELayer(
                hidden_size=config.hidden_size,
                expert=ffn,
                num_experts=config.moebert_expert_num,
                route_method=config.moebert_route_method,
                vocab_size=config.vocab_size,
                hash_list=config.moebert_route_hash_list,
            )
            self.importance_processor = ImportanceProcessor(config, layer_idx, config.moebert_expert_num, 0)
        else:
            self.experts = FeedForward(config, config.intermediate_size, dropout)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            expert_input_ids=None,
            expert_attention_mask=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = self.feed_forward(attention_output, expert_input_ids, expert_attention_mask)
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward(self, attention_output, expert_input_ids, expert_attention_mask):
        if not self.use_experts:
            layer_output = self.experts(attention_output)
            return layer_output, 0.0

        if not self.importance_processor.is_moe:
            raise RuntimeError("Need to turn the model to a MoE first.")

        layer_output, gate_loss, gate_load = self.experts(
            attention_output, expert_input_ids, expert_attention_mask
        )
        return layer_output, gate_loss


class MoEBertEncoder(BertEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

        self.layer = nn.ModuleList([MoEBertLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            expert_input_ids=None,
            expert_attention_mask=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        gate_loss = 0.0
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    expert_input_ids,
                    expert_attention_mask,
                )

            hidden_states = layer_outputs[0][0]
            gate_loss = gate_loss + layer_outputs[0][1]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return MoEModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            gate_loss=gate_loss,
        )


class MoEBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        BertModel.__init__(self, config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = MoEBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

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
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            expert_input_ids=None,
            expert_attention_mask=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            expert_input_ids=expert_input_ids,
            expert_attention_mask=expert_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return MoEModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            gate_loss=encoder_outputs.gate_loss,
        )

class MoEBertForSequenceClassification(BertPreTrainedModel):
    _keys_to_ignore_on_save = []

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.teacher = None
        self.load_balance_alpha = config.moebert_load_balance
        self.distill_alpha = config.moebert_distill
        
        ######################################################
        # Target sparsity and early exit flag
        self.moebert_target_sparsity = config.moebert_target_sparsity
        
        if self.config.moebert_fixmask_init:
            self.diff_model_state = "FIXMASK"
        else :
            self.diff_model_state = "FINETUNING"

        assert (self.moebert_target_sparsity <= 1.0) and (self.moebert_target_sparsity >= 0.0), \
            "self.moebert_target_sparsity should be in range [0.0,1.0]"
        ######################################################

        self.bert = MoEBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward_clean(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            expert_input_ids=None,
            expert_attention_mask=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            expert_input_ids=expert_input_ids,
            expert_attention_mask=expert_attention_mask,
        )
        gate_loss = outputs.gate_loss
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        loss_fct = MSELoss() if self.num_labels == 1 else CrossEntropyLoss()
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return outputs, logits, loss, gate_loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.training:
            output_hidden_states = True

        outputs, logits, loss, gate_loss = self.forward_clean(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            expert_input_ids=input_ids,
            expert_attention_mask=attention_mask,
        )

        distillation_loss = torch.tensor(0.0, device=self.device)
        if self.teacher is not None and self.training:
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            distillation_loss = self.get_distillation_loss(outputs, logits, teacher_outputs)

        ###############################################################
        # Get sparsity loss
        diff_l0_loss = self.get_sparsity_loss()
        ###############################################################

        total_loss = loss \
               + gate_loss * self.load_balance_alpha \
               + distillation_loss * self.distill_alpha \
               + diff_l0_loss * self.config.moebert_l0_loss_scale

        ###############################################################
        # Log results
        n_p, n_p_zero, n_p_one = self.count_non_zero_params()
        n_p_mid = n_p - n_p_zero - n_p_one
        n_p_nonzero = n_p - n_p_zero

        # sparsity in percent : ratio of non-zero params
        sparsity = n_p_nonzero / (n_p + 1e-8)

        self.log_msg = {
            # Loss
            "train_loss" : total_loss.item(),
            "cross_entropy_loss" : loss.item(),
            "gate_loss" : gate_loss.item() * self.load_balance_alpha if (type(gate_loss) == torch.Tensor) \
                                                                else gate_loss * self.load_balance_alpha,
            "distillation_loss" : distillation_loss.item() * self.distill_alpha,
            "diff_l0_loss" : diff_l0_loss.item() * self.config.moebert_l0_loss_scale,

            # Sparsity
            "n_p_zero" : n_p_zero,
            "n_p_mid" : n_p_mid,
            "n_p_one" : n_p_one,
            "sparsity" : sparsity,
        }
        
        # Move to fixed-mask finetuning when reaching target sparsity
        if (self.moebert_target_sparsity >= sparsity) and (self.diff_model_state == "FINETUNING"):
            self.convert_diffmodel_to_fixmask(pct = self.moebert_target_sparsity)
            self.diff_model_state = "FIXMASK"

        ###############################################################

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_distillation_loss(self, outputs, logits, teacher_outputs):
        hidden = outputs.hidden_states  # num_layers+1 elements, the first one is embedding
        teacher_hidden = teacher_outputs.hidden_states
        teacher_logits = teacher_outputs.logits

        hidden_loss_fct = MSELoss()
        loss = 0.0
        for i in range(len(hidden)):
            loss = loss + hidden_loss_fct(hidden[i], teacher_hidden[i])

        if self.num_labels == 1:
            prediction_loss_fct = MSELoss()
            prediction_loss = prediction_loss_fct(logits, teacher_logits)
        else:
            prediction_loss = symmetric_KL_loss(softmax(logits), softmax(teacher_logits)) / logits.size(0)
        loss = loss + prediction_loss

        return loss
    
    def count_non_zero_params(self):
        n_p, n_p_zero, n_p_one = 0, 0, 0
        for n, mod in list(self.named_modules()):
            if mod.__class__.__name__ == "DiffFeedForward":
                a,b,c = mod._count_non_zero_params()
                n_p += a
                n_p_zero += b
                n_p_one += c
        return (n_p, n_p_zero, n_p_one)

    def get_sparsity_loss(self):
        diff_l0_loss = torch.tensor(0.0, device=self.device)
        for n, mod in list(self.named_modules()):
            if mod.__class__.__name__ == "DiffFeedForward":
                diff_l0_loss += mod._get_sparsity_loss()
        return diff_l0_loss

    def get_diffmoe_param_groups(self,trainer_args):

        param_dense_with_decay = []
        param_dense_no_decay = []
        param_finetune = []
        param_alpha = []
        param_diff_weight = []

        for n,p in list(self.named_parameters()):

            param_suffix = n.split(".")[-1]

            if "teacher" in n :
                continue
            elif param_suffix == "original":
                continue
            elif param_suffix == "diff_weight":
                param_diff_weight.append(p)
            elif param_suffix == "finetune":
                param_finetune.append(p)
            elif (param_suffix == "alpha") or (param_suffix == "alpha_group"):
                param_alpha.append(p)
            else :
                if ("LayerNorm" in n) or (param_suffix == "bias"):
                    param_dense_no_decay.append(p)
                else :
                    param_dense_with_decay.append(p)

        param_groups = [
            {
                "name": "param_dense_with_decay",
                "params": param_dense_with_decay,
                "weight_decay": trainer_args.weight_decay,
                "lr": trainer_args.learning_rate,
            },
            {
                "name": "param_dense_no_decay",
                "params": param_dense_no_decay,
                "weight_decay": 0.0,
                "lr": trainer_args.learning_rate,
            },
        ]
        
        if param_finetune != []:
            param_groups.extend([
                {
                    "name": "param_finetune",
                    "params": param_finetune,
                    "weight_decay": trainer_args.weight_decay,
                    "lr": trainer_args.learning_rate,
                },
                {
                    "name": "param_alpha",
                    "params": param_alpha,
                    "weight_decay": trainer_args.weight_decay,
                    "lr": self.config.moebert_learning_rate_alpha,
                },
            ])

        if param_diff_weight != []:
            param_groups.extend([
                {
                    "name": "param_diff_weight",
                    "params": param_diff_weight,
                    "weight_decay": trainer_args.weight_decay,
                    "lr": trainer_args.learning_rate,
                },
            ])

        return param_groups

    def convert_diffmodel_to_fixmask(self, pct):
        print("Beginning fixed-mask finetuning")
        for n, mod in list(self.named_modules()):
            if mod.__class__.__name__ == "DiffFeedForward":
                mod._finetune_to_fixmask(
                    pct=pct,
                )

class MoEBertForQuestionAnswering(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_save = []

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.teacher = None
        self.load_balance_alpha = config.moebert_load_balance
        self.distill_alpha = config.moebert_distill

        self.bert = MoEBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward_clean(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            start_positions=None,
            end_positions=None,
            expert_input_ids=None,
            expert_attention_mask=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            expert_input_ids=expert_input_ids,
            expert_attention_mask=expert_attention_mask,
        )
        gate_loss = outputs.gate_loss

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return outputs, start_logits, end_logits, total_loss, gate_loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.training:
            output_hidden_states = True

        outputs, start_logits, end_logits, total_loss, gate_loss = self.forward_clean(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start_positions=start_positions,
            end_positions=end_positions,
            expert_input_ids=input_ids,
            expert_attention_mask=attention_mask,
        )

        distillation_loss = 0.0
        if self.teacher is not None and self.training:
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                start_positions=start_positions.clone(),
                end_positions=end_positions.clone(),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            distillation_loss = self.get_distillation_loss(outputs, start_logits, end_logits, teacher_outputs)

        if total_loss is not None:
            total_loss = total_loss \
                         + gate_loss * self.load_balance_alpha \
                         + distillation_loss * self.distill_alpha

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_distillation_loss(self, outputs, start_logits, end_logits, teacher_outputs):
        hidden = outputs.hidden_states  # num_layers+1 elements, the first one is embedding
        teacher_hidden = teacher_outputs.hidden_states
        teacher_start_logits, teacher_end_logits = teacher_outputs.start_logits, teacher_outputs.end_logits

        hidden_loss_fct = MSELoss()
        loss = 0.0
        for i in range(len(hidden)):
            loss = loss + hidden_loss_fct(hidden[i], teacher_hidden[i])

        bsz = start_logits.size(0)
        prediction_loss_start = symmetric_KL_loss(softmax(start_logits), softmax(teacher_start_logits)) / bsz
        prediction_loss_end = symmetric_KL_loss(softmax(end_logits), softmax(teacher_end_logits)) / bsz
        loss = loss + (prediction_loss_start + prediction_loss_end) * 0.5

        return loss
