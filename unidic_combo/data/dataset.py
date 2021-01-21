import copy
import logging
import pathlib
from typing import Union, List, Dict, Iterable, Optional, Any, Tuple

import conllu
import torch
from allennlp import data as allen_data
from allennlp.common import checks, util
from allennlp.data import fields as allen_fields, vocabulary
from conllu import parser
from dataclasses import dataclass
from overrides import overrides

from unidic_combo.data import fields

logger = logging.getLogger(__name__)


@allen_data.DatasetReader.register("conllu")
class UniversalDependenciesDatasetReader(allen_data.DatasetReader):

    def __init__(
            self,
            token_indexers: Dict[str, allen_data.TokenIndexer] = None,
            lemma_indexers: Dict[str, allen_data.TokenIndexer] = None,
            features: List[str] = None,
            targets: List[str] = None,
            use_sem: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if features is None:
            features = ["token", "char"]
        if targets is None:
            targets = ["head", "deprel", "upostag", "xpostag", "lemma", "feats"]

        if "token" not in features and "char" not in features:
            raise checks.ConfigurationError("There must be at least one ('char' or 'token') text-based feature!")

        if "deps" in targets and not ("head" in targets and "deprel" in targets):
            raise checks.ConfigurationError("Add 'head' and 'deprel' to targets when using 'deps'!")

        intersection = set(features).intersection(set(targets))
        if len(intersection) != 0:
            raise checks.ConfigurationError(
                "Features and targets cannot share elements! "
                "Remove {} from either features or targets.".format(intersection)
            )
        self.use_sem = use_sem

        # *.conllu readers configuration
        fields = list(parser.DEFAULT_FIELDS)
        fields[1] = "token"  # use 'token' instead of 'form'
        field_parsers = parser.DEFAULT_FIELD_PARSERS
        # Do not make it nullable
        field_parsers.pop("xpostag", None)
        # Ignore parsing misc
        field_parsers.pop("misc", None)
        if self.use_sem:
            fields = list(fields)
            fields.append("semrel")
            field_parsers["semrel"] = lambda line, i: line[i]
        self.field_parsers = field_parsers
        self.fields = tuple(fields)

        self._token_indexers = token_indexers
        self._lemma_indexers = lemma_indexers
        self._targets = targets
        self._features = features
        self.generate_labels = True
        # Filter out not required token indexers to avoid
        # Mismatched token keys ConfigurationError
        for indexer_name in list(self._token_indexers.keys()):
            if indexer_name not in self._features:
                del self._token_indexers[indexer_name]

    @overrides
    def _read(self, file_path: str) -> Iterable[allen_data.Instance]:
        file_path = [file_path] if len(file_path.split(",")) == 0 else file_path.split(",")

        for conllu_file in file_path:
            file = pathlib.Path(conllu_file)
            assert conllu_file and file.exists(), f"File with path '{conllu_file}' does not exists!"
            with file.open("r") as f:
                for annotation in conllu.parse_incr(f, fields=self.fields, field_parsers=self.field_parsers):
                    yield self.text_to_instance(annotation)

    @overrides
    def text_to_instance(self, tree: conllu.TokenList) -> allen_data.Instance:
        fields_: Dict[str, allen_data.Field] = {}
        tree_tokens = [t for t in tree if isinstance(t["id"], int)]
        tokens = [_Token(t["token"],
                         pos_=t.get("upostag"),
                         tag_=t.get("xpostag"),
                         lemma_=t.get("lemma"),
                         feats_=t.get("feats"))
                  for t in tree_tokens]

        # features
        text_field = allen_fields.TextField(tokens, self._token_indexers)
        fields_["sentence"] = text_field

        # targets
        if self.generate_labels:
            for target_name in self._targets:
                if target_name != "sent":
                    target_values = [t[target_name] for t in tree_tokens]
                    if target_name == "lemma":
                        target_values = [allen_data.Token(v) for v in target_values]
                        fields_[target_name] = allen_fields.TextField(target_values, self._lemma_indexers)
                    elif target_name == "feats":
                        target_values = self._feat_values(tree_tokens)
                        fields_[target_name] = fields.SequenceMultiLabelField(target_values,
                                                                              self._feats_indexer,
                                                                              self._feats_as_tensor_wrapper,
                                                                              text_field,
                                                                              label_namespace="feats_labels")
                    elif target_name == "head":
                        target_values = [0 if v == "_" else int(v) for v in target_values]
                        fields_[target_name] = allen_fields.SequenceLabelField(target_values, text_field,
                                                                               label_namespace=target_name + "_labels")
                    elif target_name == "deps":
                        # Graphs require adding ROOT (AdjacencyField uses sequence length from TextField).
                        text_field_deps = allen_fields.TextField([_Token("ROOT")] + copy.deepcopy(tokens),
                                                                 self._token_indexers)
                        enhanced_heads: List[Tuple[int, int]] = []
                        enhanced_deprels: List[str] = []
                        for idx, t in enumerate(tree_tokens):
                            t_deps = t["deps"]
                            if t_deps and t_deps != "_":
                                for rel, head in t_deps:
                                    # EmoryNLP skips the first edge, if there are two edges between the same
                                    # nodes. Thanks to that one is in a tree and another in a graph.
                                    # This snippet follows that approach.
                                    if enhanced_heads and enhanced_heads[-1] == (idx, head):
                                        enhanced_heads.pop()
                                        enhanced_deprels.pop()
                                    enhanced_heads.append((idx, head))
                                    enhanced_deprels.append(rel)
                        fields_["enhanced_heads"] = allen_fields.AdjacencyField(
                            indices=enhanced_heads,
                            sequence_field=text_field_deps,
                            label_namespace="enhanced_heads_labels",
                            padding_value=0,
                        )
                        fields_["enhanced_deprels"] = allen_fields.AdjacencyField(
                            indices=enhanced_heads,
                            sequence_field=text_field_deps,
                            labels=enhanced_deprels,
                            # Label namespace matches regular tree parsing.
                            label_namespace="deprel_labels",
                            padding_value=0,
                        )
                    else:
                        fields_[target_name] = allen_fields.SequenceLabelField(target_values, text_field,
                                                                               label_namespace=target_name + "_labels")

        # Restore feats fields to string representation
        # parser.serialize_field doesn't handle key without value
        for token in tree.tokens:
            if "feats" in token:
                feats = token["feats"]
                if feats:
                    feats_values = []
                    for k, v in feats.items():
                        feats_values.append('='.join((k, v)) if v else k)
                    field = "|".join(feats_values)
                else:
                    field = "_"
                token["feats"] = field

        # metadata
        fields_["metadata"] = allen_fields.MetadataField({"input": tree,
                                                          "field_names": self.fields,
                                                          "tokens": tokens})

        return allen_data.Instance(fields_)

    @staticmethod
    def _feat_values(tree: List[Dict[str, Any]]):
        features = []
        for token in tree:
            token_features = []
            if token["feats"] is not None:
                for feat, value in token["feats"].items():
                    if feat in ["_", "__ROOT__"]:
                        pass
                    else:
                        # Handle case where feature is binary (doesn't have associated value)
                        if value:
                            token_features.append(feat + "=" + value)
                        else:
                            token_features.append(feat)
            features.append(token_features)
        return features

    @staticmethod
    def _feats_as_tensor_wrapper(field: fields.SequenceMultiLabelField):
        def as_tensor(padding_lengths):
            desired_num_tokens = padding_lengths["num_tokens"]
            assert len(field._indexed_multi_labels) > 0
            classes_count = len(field._indexed_multi_labels[0])
            default_value = [0.0] * classes_count
            padded_tags = util.pad_sequence_to_length(field._indexed_multi_labels, desired_num_tokens,
                                                      lambda: default_value)
            tensor = torch.LongTensor(padded_tags)
            return tensor

        return as_tensor

    @staticmethod
    def _feats_indexer(vocab: allen_data.Vocabulary):
        label_namespace = "feats_labels"
        vocab_size = vocab.get_vocab_size(label_namespace)
        slices = get_slices_if_not_provided(vocab)

        def _m_from_n_ones_encoding(multi_label: List[str], sentence_length: int) -> List[int]:
            one_hot_encoding = [0] * vocab_size
            for cat, cat_indices in slices.items():
                if cat not in ["__PAD__", "_"]:
                    label_from_cat = [label for label in multi_label if cat == label.split("=")[0]]
                    if label_from_cat:
                        label_from_cat = label_from_cat[0]
                        index = vocab.get_token_index(label_from_cat, label_namespace)
                    else:
                        # Get Cat=None index
                        index = vocab.get_token_index(cat + "=None", label_namespace)
                    one_hot_encoding[index] = 1
            return one_hot_encoding

        return _m_from_n_ones_encoding


@allen_data.Vocabulary.register("from_instances_extended", constructor="from_instances_extended")
class Vocabulary(allen_data.Vocabulary):

    @classmethod
    def from_instances_extended(
            cls,
            instances: Iterable[allen_data.Instance],
            min_count: Dict[str, int] = None,
            max_vocab_size: Union[int, Dict[str, int]] = None,
            non_padded_namespaces: Iterable[str] = vocabulary.DEFAULT_NON_PADDED_NAMESPACES,
            pretrained_files: Optional[Dict[str, str]] = None,
            only_include_pretrained_words: bool = False,
            min_pretrained_embeddings: Dict[str, int] = None,
            padding_token: Optional[str] = vocabulary.DEFAULT_PADDING_TOKEN,
            oov_token: Optional[str] = vocabulary.DEFAULT_OOV_TOKEN,
    ) -> "Vocabulary":
        """
        Extension to manually fill gaps in missing 'feats_labels'.
        """
        # Load manually tokens from pretrained file (using different strategy
        # - only words add all embedding file, without checking if were seen
        # in any dataset.
        tokens_to_add = None
        if pretrained_files and "tokens" in pretrained_files:
            pretrained_set = set(vocabulary._read_pretrained_tokens(pretrained_files["tokens"]))
            tokens_to_add = {"tokens": list(pretrained_set)}
            pretrained_files = None

        vocab = super().from_instances(
            instances=instances,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            non_padded_namespaces=non_padded_namespaces,
            pretrained_files=pretrained_files,
            only_include_pretrained_words=only_include_pretrained_words,
            tokens_to_add=tokens_to_add,
            min_pretrained_embeddings=min_pretrained_embeddings,
            padding_token=padding_token,
            oov_token=oov_token
        )
        # Extending vocab with features that does not show up explicitly.
        # To know all features we need to read full dataset first.
        # Adding auxiliary '=None' feature for each category is needed
        # to perform classification.
        get_slices_if_not_provided(vocab)
        return vocab


def get_slices_if_not_provided(vocab: allen_data.Vocabulary):
    if hasattr(vocab, "slices"):
        return vocab.slices

    if "feats_labels" in vocab.get_namespaces():
        idx2token = vocab.get_index_to_token_vocabulary("feats_labels")
        for _, v in dict(idx2token).items():
            if v not in ["_", "__PAD__"]:
                empty_value = v.split("=")[0] + "=None"
                vocab.add_token_to_namespace(empty_value, "feats_labels")

        slices = {}
        for idx, name in vocab.get_index_to_token_vocabulary("feats_labels").items():
            # There are 2 types features: with (Case=Acc) or without assigment (None).
            # Here we group their indices by name (before assigment sign).
            name = name.split("=")[0]
            if name in slices:
                slices[name].append(idx)
            else:
                slices[name] = [idx]
        vocab.slices = slices
        return vocab.slices


@dataclass(init=False, repr=False)
class _Token(allen_data.Token):
    __slots__ = allen_data.Token.__slots__ + ['feats_']

    feats_: Optional[str]

    def __init__(self, text: str = None, idx: int = None, idx_end: int = None, lemma_: str = None, pos_: str = None,
                 tag_: str = None, dep_: str = None, ent_type_: str = None, text_id: int = None, type_id: int = None,
                 feats_: str = None) -> None:
        super().__init__(text, idx, idx_end, lemma_, pos_, tag_, dep_, ent_type_, text_id, type_id)
        self.feats_ = feats_
