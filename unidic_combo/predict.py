import logging
import os
from typing import List, Union, Dict, Any

import numpy as np
from allennlp import data as allen_data, common, models
from allennlp.common import util
from allennlp.data import tokenizers
from allennlp.predictors import predictor
from overrides import overrides

from unidic_combo import data
from unidic_combo.data import sentence2conllu, tokens2conllu, conllu2sentence
from unidic_combo.utils import download, graph

logger = logging.getLogger(__name__)


@predictor.Predictor.register("unidic_combo")
@predictor.Predictor.register("unidic_combo-spacy", constructor="with_spacy_tokenizer")
class COMBO(predictor.Predictor):

    def __init__(self,
                 model: models.Model,
                 dataset_reader: allen_data.DatasetReader,
                 tokenizer: allen_data.Tokenizer = tokenizers.WhitespaceTokenizer(),
                 batch_size: int = 1024,
                 line_to_conllu: bool = True) -> None:
        super().__init__(model, dataset_reader)
        self.batch_size = batch_size
        self.vocab = model.vocab
        self.dataset_reader = self._dataset_reader
        self.dataset_reader.generate_labels = False
        self.dataset_reader.lazy = True
        self._tokenizer = tokenizer
        self.without_sentence_embedding = False
        self.line_to_conllu = line_to_conllu

    def __call__(self, sentence: Union[str, List[str], List[List[str]], List[data.Sentence]]):
        """Depending on the input uses (or ignores) tokenizer.
        When model isn't only text-based only List[data.Sentence] is possible input.

        * str - tokenizer is used
        * List[str] - tokenizer is used for each string (treated as list of raw sentences)
        * List[List[str]] - tokenizer isn't used (treated as list of tokenized sentences)
        * List[data.Sentence] - tokenizer isn't used (treated as list of tokenized sentences)

        :param sentence: sentence(s) representation
        :return: Sentence or List[Sentence] depending on the input
        """
        return self.predict(sentence)

    def predict(self, sentence: Union[str, List[str], List[List[str]], List[data.Sentence]]):
        if isinstance(sentence, str):
            return self.predict_json({"sentence": sentence})
        elif isinstance(sentence, list):
            if len(sentence) == 0:
                return []
            example = sentence[0]
            sentences = sentence
            if isinstance(example, str) or isinstance(example, list):
                result = []
                sentences = [self._to_input_json(s) for s in sentences]
                for sentences_batch in util.lazy_groups_of(sentences, self.batch_size):
                    sentences_batch = self.predict_batch_json(sentences_batch)
                    result.extend(sentences_batch)
                return result
            elif isinstance(example, data.Sentence):
                result = []
                sentences = [self._to_input_instance(s) for s in sentences]
                for sentences_batch in util.lazy_groups_of(sentences, self.batch_size):
                    sentences_batch = self.predict_batch_instance(sentences_batch)
                    result.extend(sentences_batch)
                return result
            else:
                raise ValueError("List must have either sentences as str, List[str] or Sentence object.")
        else:
            raise ValueError("Input must be either string or list of strings.")

    @overrides
    def predict_batch_instance(self, instances: List[allen_data.Instance]) -> List[data.Sentence]:
        sentences = []
        predictions = super().predict_batch_instance(instances)
        for prediction, instance in zip(predictions, instances):
            tree, sentence_embedding = self._predictions_as_tree(prediction, instance)
            sentence = conllu2sentence(tree, sentence_embedding)
            sentences.append(sentence)
        return sentences

    @overrides
    def predict_batch_json(self, inputs: List[common.JsonDict]) -> List[data.Sentence]:
        instances = self._batch_json_to_instances(inputs)
        sentences = self.predict_batch_instance(instances)
        return sentences

    @overrides
    def predict_instance(self, instance: allen_data.Instance, serialize: bool = True) -> data.Sentence:
        predictions = super().predict_instance(instance)
        tree, sentence_embedding = self._predictions_as_tree(predictions, instance)
        return conllu2sentence(tree, sentence_embedding)

    @overrides
    def predict_json(self, inputs: common.JsonDict) -> data.Sentence:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    @overrides
    def _json_to_instance(self, json_dict: common.JsonDict) -> allen_data.Instance:
        sentence = json_dict["sentence"]
        if isinstance(sentence, str):
            tokens = [t.text for t in self._tokenizer.tokenize(json_dict["sentence"])]
        elif isinstance(sentence, list):
            tokens = sentence
        else:
            raise ValueError("Input must be either string or list of strings.")
        return self.dataset_reader.text_to_instance(tokens2conllu(tokens))

    @overrides
    def load_line(self, line: str) -> common.JsonDict:
        return self._to_input_json(line.replace("\n", "").strip())

    @overrides
    def dump_line(self, outputs: data.Sentence) -> str:
        # Check whether serialized (str) tree or token's list
        # Serialized tree has already separators between lines
        if self.without_sentence_embedding:
            outputs.sentence_embedding = []
        if self.line_to_conllu:
            return sentence2conllu(outputs, keep_semrel=self.dataset_reader.use_sem).serialize()
        else:
            return outputs.to_json()

    @staticmethod
    def _to_input_json(sentence: str):
        return {"sentence": sentence}

    def _to_input_instance(self, sentence: data.Sentence) -> allen_data.Instance:
        return self.dataset_reader.text_to_instance(sentence2conllu(sentence))

    def _predictions_as_tree(self, predictions: Dict[str, Any], instance: allen_data.Instance):
        tree = instance.fields["metadata"]["input"]
        field_names = instance.fields["metadata"]["field_names"]
        tree_tokens = [t for t in tree if isinstance(t["id"], int)]
        for field_name in field_names:
            if field_name not in predictions:
                continue
            field_predictions = predictions[field_name]
            for idx, token in enumerate(tree_tokens):
                if field_name in {"xpostag", "upostag", "semrel", "deprel"}:
                    value = self.vocab.get_token_from_index(field_predictions[idx], field_name + "_labels")
                    token[field_name] = value
                elif field_name == "head":
                    token[field_name] = int(field_predictions[idx])
                elif field_name == "deps":
                    # Handled after every other decoding
                    continue

                elif field_name == "feats":
                    slices = self._model.morphological_feat.slices
                    features = []
                    prediction = field_predictions[idx]
                    for (cat, cat_indices), pred_idx in zip(slices.items(), prediction):
                        if cat not in ["__PAD__", "_"]:
                            value = self.vocab.get_token_from_index(cat_indices[pred_idx],
                                                                    field_name + "_labels")
                            # Exclude auxiliary values
                            if "=None" not in value:
                                features.append(value)
                    if len(features) == 0:
                        field_value = "_"
                    else:
                        lowercase_features = [f.lower() for f in features]
                        arg_indices = sorted(range(len(lowercase_features)), key=lowercase_features.__getitem__)
                        field_value = "|".join(np.array(features)[arg_indices].tolist())

                    token[field_name] = field_value
                elif field_name == "lemma":
                    prediction = field_predictions[idx]
                    word_chars = []
                    for char_idx in prediction[1:-1]:
                        pred_char = self.vocab.get_token_from_index(char_idx, "lemma_characters")

                        if pred_char == "__END__":
                            break
                        elif pred_char == "__PAD__":
                            continue
                        elif "_" in pred_char:
                            pred_char = "?"

                        word_chars.append(pred_char)
                    token[field_name] = "".join(word_chars)
                else:
                    raise NotImplementedError(f"Unknown field name {field_name}!")

        if "enhanced_head" in predictions and predictions["enhanced_head"]:
            # TODO off-by-one hotfix, refactor
            h = np.array(predictions["enhanced_head"])
            h = np.concatenate((h[-1:], h[:-1]))
            r = np.array(predictions["enhanced_deprel_prob"])
            r = np.concatenate((r[-1:], r[:-1]))
            graph.sdp_to_dag_deps(arc_scores=h,
                                  rel_scores=r,
                                  tree_tokens=tree_tokens,
                                  root_idx=self.vocab.get_token_index("root", "deprel_labels"),
                                  vocab_index=self.vocab.get_index_to_token_vocabulary("deprel_labels"))
            empty_tokens = graph.restore_collapse_edges(tree_tokens)
            tree.tokens.extend(empty_tokens)

        return tree, predictions["sentence_embedding"]

    @classmethod
    def with_spacy_tokenizer(cls, model: models.Model,
                             dataset_reader: allen_data.DatasetReader):
        return cls(model, dataset_reader, tokenizers.SpacyTokenizer())

    @classmethod
    def from_pretrained(cls, path: str, tokenizer=tokenizers.SpacyTokenizer(),
                        batch_size: int = 1024,
                        cuda_device: int = -1):
        util.import_module_and_submodules("unidic_combo.commands")
        util.import_module_and_submodules("unidic_combo.models")
        util.import_module_and_submodules("unidic_combo.training")

        if os.path.exists(path):
            model_path = path
        else:
            try:
                logger.debug("Downloading model.")
                model_path = download.download_file(path)
            except Exception as e:
                logger.error(e)
                raise e

        archive = models.load_archive(model_path, cuda_device=cuda_device)
        model = archive.model
        dataset_reader = allen_data.DatasetReader.from_params(
            archive.config["dataset_reader"])
        return cls(model, dataset_reader, tokenizer, batch_size)
