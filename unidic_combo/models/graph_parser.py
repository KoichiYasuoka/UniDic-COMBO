"""Enhanced dependency parsing models."""
from typing import Tuple, Dict, Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from allennlp import data
from allennlp.nn import chu_liu_edmonds

from unidic_combo.models import base, utils


class GraphHeadPredictionModel(base.Predictor):
    """Head prediction model."""

    def __init__(self,
                 head_projection_layer: base.Linear,
                 dependency_projection_layer: base.Linear,
                 cycle_loss_n: int = 0,
                 graph_weighting: float = 0.2):
        super().__init__()
        self.head_projection_layer = head_projection_layer
        self.dependency_projection_layer = dependency_projection_layer
        self.cycle_loss_n = cycle_loss_n
        self.graph_weighting = graph_weighting

    def forward(self,
                x: Union[torch.Tensor, List[torch.Tensor]],
                labels: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                mask: Optional[torch.BoolTensor] = None,
                sample_weights: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            mask = x.new_ones(x.size()[-1])
        heads_labels = None
        if labels is not None and labels[0] is not None:
            heads_labels = labels

        head_arc_emb = self.head_projection_layer(x)
        dep_arc_emb = self.dependency_projection_layer(x)
        x = dep_arc_emb.bmm(head_arc_emb.transpose(2, 1))
        pred = x.sigmoid() > 0.5

        output = {
            "prediction": pred,
            "probability": x
        }

        if heads_labels is not None:
            if sample_weights is None:
                sample_weights = heads_labels.new_ones([mask.size(0)])
            output["loss"], output["cycle_loss"] = self._loss(x, heads_labels, mask, sample_weights)

        return output

    def _cycle_loss(self, pred: torch.Tensor):
        BATCH_SIZE, _, _ = pred.size()
        loss = pred.new_zeros(BATCH_SIZE)
        # Index from 1: as using non __ROOT__ tokens
        pred = pred.softmax(-1)[:, 1:, 1:]
        x = pred
        for i in range(self.cycle_loss_n):
            loss += self._batch_trace(x)

            # Don't multiple on last iteration
            if i < self.cycle_loss_n - 1:
                x = x.bmm(pred)

        return loss

    @staticmethod
    def _batch_trace(x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 3
        BATCH_SIZE, N, M = x.size()
        assert N == M
        identity = x.new_tensor(torch.eye(N))
        identity = identity.reshape((1, N, N))
        batch_identity = identity.repeat(BATCH_SIZE, 1, 1)
        return (x * batch_identity).sum((-1, -2))

    def _loss(self, pred: torch.Tensor, labels: torch.Tensor,  mask: torch.BoolTensor,
              sample_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        BATCH_SIZE, N, M = pred.size()
        assert N == M
        SENTENCE_LENGTH = N

        valid_positions = mask.sum()

        result = []
        true = labels
        # Ignore first pred dimension as it is ROOT token prediction
        for i in range(SENTENCE_LENGTH - 1):
            pred_i = pred[:, i + 1, 1:].reshape(-1)
            true_i = true[:, i + 1, 1:].reshape(-1)
            mask_i = mask[:, i]
            bce_loss = F.binary_cross_entropy_with_logits(pred_i, true_i, reduction="none").mean(-1) * mask_i
            result.append(bce_loss)
        cycle_loss = self._cycle_loss(pred)
        loss = torch.stack(result).transpose(1, 0) * sample_weights.unsqueeze(-1)
        return loss.sum() / valid_positions + cycle_loss.mean(), cycle_loss.mean()


@base.Predictor.register("combo_graph_dependency_parsing_from_vocab", constructor="from_vocab")
class GraphDependencyRelationModel(base.Predictor):
    """Dependency relation parsing model."""

    def __init__(self,
                 head_predictor: GraphHeadPredictionModel,
                 head_projection_layer: base.Linear,
                 dependency_projection_layer: base.Linear,
                 relation_prediction_layer: base.Linear):
        super().__init__()
        self.head_predictor = head_predictor
        self.head_projection_layer = head_projection_layer
        self.dependency_projection_layer = dependency_projection_layer
        self.relation_prediction_layer = relation_prediction_layer

    def forward(self,
                x: Union[torch.Tensor, List[torch.Tensor]],
                mask: Optional[torch.BoolTensor] = None,
                labels: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                sample_weights: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        relations_labels, head_labels, enhanced_heads_labels, enhanced_deprels_labels = None, None, None, None
        if labels is not None and labels[0] is not None:
            relations_labels, head_labels, enhanced_heads_labels = labels

        head_output = self.head_predictor(x, enhanced_heads_labels, mask, sample_weights)
        head_pred = head_output["probability"]
        BATCH_SIZE, LENGTH, _ = head_pred.size()

        head_rel_emb = self.head_projection_layer(x)

        dep_rel_emb = self.dependency_projection_layer(x)

        # All possible edges combinations for each batch
        # Repeat interleave to have [emb1, emb1 ... (length times) ... emb1, emb2 ... ]
        head_rel_pred = head_rel_emb.repeat_interleave(LENGTH, -2)
        # Regular repeat to have all combinations [deprel1, deprel2, ... deprelL, deprel1 ...]
        dep_rel_pred = dep_rel_emb.repeat(1, LENGTH, 1)

        # All possible edges combinations for each batch
        dep_rel_pred = torch.cat((head_rel_pred, dep_rel_pred), dim=-1)

        relation_prediction = self.relation_prediction_layer(dep_rel_pred).reshape(BATCH_SIZE, LENGTH, LENGTH, -1)
        output = head_output

        output["prediction"] = (relation_prediction.argmax(-1), head_output["prediction"])
        output["rel_probability"] = relation_prediction

        if labels is not None and labels[0] is not None:
            if sample_weights is None:
                sample_weights = labels.new_ones([mask.size(0)])
            loss = self._loss(relation_prediction, relations_labels, enhanced_heads_labels, mask, sample_weights)
            output["loss"] = (loss, head_output["loss"])

        return output

    @staticmethod
    def _loss(pred: torch.Tensor,
              true: torch.Tensor,
              heads_true: torch.Tensor,
              mask: torch.BoolTensor,
              sample_weights: torch.Tensor) -> torch.Tensor:
        correct_heads_mask = heads_true.long() == 1
        true = true[correct_heads_mask]
        pred = pred[correct_heads_mask]
        loss = F.cross_entropy(pred, true.long())
        return loss.sum() / pred.size(0)

    @classmethod
    def from_vocab(cls,
                   vocab: data.Vocabulary,
                   vocab_namespace: str,
                   head_predictor: GraphHeadPredictionModel,
                   head_projection_layer: base.Linear,
                   dependency_projection_layer: base.Linear
                   ):
        """Creates parser combining model configuration and vocabulary data."""
        assert vocab_namespace in vocab.get_namespaces()
        relation_prediction_layer = base.Linear(
            in_features=head_projection_layer.get_output_dim() + dependency_projection_layer.get_output_dim(),
            out_features=vocab.get_vocab_size(vocab_namespace)
        )
        return cls(
            head_predictor=head_predictor,
            head_projection_layer=head_projection_layer,
            dependency_projection_layer=dependency_projection_layer,
            relation_prediction_layer=relation_prediction_layer
        )
