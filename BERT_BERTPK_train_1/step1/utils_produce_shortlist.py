from __future__ import absolute_import, division, print_function

import csv
import logging
import warnings
import numpy as np
import pandas as pd
import random
import pickle
import os
from collections import OrderedDict
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

def pickle_dump_large_file(obj, filepath):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, gu_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.gu_id = gu_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, dict_explanations):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, dict_explanations):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class EprgProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, train_data_dir, dict_explanations):
        """See base class."""
        logger.info("LOOKING AT {}".format(train_data_dir))
        return self._create_train_examples(
            train_data_dir,  dict_explanations)

    def get_dev_examples(self, row, dict_explanations):
        """See base class."""
        return self._create_dev_examples(row, dict_explanations)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_train_examples(self, questions_file, dict_explanations):
        """Creates examples for the training set."""
        df_q = pd.read_csv(questions_file, sep='\t')
        df_q['Answer_flag'] = None
        df_q['row_flag'] = None
        df_q['explanation_lenth'] = None
        df_q['Answer_number'] = df_q['question'].map(lambda x: len(x.split('(')) - 1)
        df_q['explanation_lenth'] = df_q['explanation'].map(lambda y: len(list(OrderedDict.fromkeys(str(y).split(' ')).keys())))
        examples = []
        i_flag = 0
        count_not_in_tables=[]
        count_not_in_tables_questionid=[]
        for _, row in df_q.iterrows():
            if 'SUCCESS' not in str(row['flags']).split(' '):
                continue
            if row['AnswerKey'] == 'A' or row['AnswerKey'] == "1":
                ac = 0
            elif row['AnswerKey'] == 'B' or row['AnswerKey'] == "2":
                ac = 1
            elif row['AnswerKey'] == 'C' or row['AnswerKey'] == "3":
                ac = 2
            else:
                ac = 3
            question_ac = row['question'].split('(')[0] +'[ANSWER]'+row['question'].split('(')[ac + 1].split(')')[1]
            question_ac = question_ac.replace("''", '" ').replace("``", '" ')
            text_a = question_ac
            explanations_id_list = []
            for single_row_id in list(OrderedDict.fromkeys(str(row['explanation']).split(' ')).keys()):
                explanations_id_list.append(single_row_id.split('|')[0])
            filtered_explanations_id_list=explanations_id_list.copy()
            for filer_single in explanations_id_list:
                if filer_single not in dict_explanations.keys():
                    count_not_in_tables.append(filer_single)
                    count_not_in_tables_questionid.append(row['QuestionID'])
                    filtered_explanations_id_list.remove(filer_single)
            non_explanations_list = []
            for each_row in dict_explanations.keys():
                if each_row not in filtered_explanations_id_list:
                    non_explanations_list.append(each_row)
            non_explanations_list = random.sample(non_explanations_list, 100)
            final_rows_list = filtered_explanations_id_list+non_explanations_list
            random.shuffle(final_rows_list)
            for each_row_id in final_rows_list:
                if each_row_id in filtered_explanations_id_list:
                    i_flag += 1
                    each_row_true = dict_explanations[each_row_id]
                    each_row_true = each_row_true.replace("''", '" ').replace("``", '" ')
                    text_b = each_row_true
                    guid = i_flag
                    label = "1"
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                else:
                    i_flag += 1
                    each_row_false = dict_explanations[each_row_id]
                    each_row_false = each_row_false.replace("''", '" ').replace("``", '" ')
                    text_b = each_row_false
                    guid = i_flag
                    label = "0"
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        print('examples length: ', len(examples))
        print('count_not_in_tables_questions: ', len(count_not_in_tables_questionid))
        print('count_not_in_tables_rows: ', len(set(count_not_in_tables)))
        return examples

    def _create_dev_examples(self,row, dict_explanations):
        i_flag = 0
        examples = []
        debug_output_dict = {}
        if row['AnswerKey'] == 'A' or row['AnswerKey'] == "1":
            ac = 0
        elif row['AnswerKey'] == 'B' or row['AnswerKey'] == "2":
            ac = 1
        elif row['AnswerKey'] == 'C' or row['AnswerKey'] == "3":
            ac = 2
        else:
            ac = 3
        question_ac = row['question'].split('(')[0] + '[ANSWER]' + row['question'].split('(')[ac + 1].split(')')[1]
        question_ac = question_ac.replace("''", '" ').replace("``", '" ')
        text_a = question_ac
        explanations_id_list = []
        explanations_role_list = []
        for single_row_id in list(OrderedDict.fromkeys(str(row['explanation']).split(' ')).keys()):
            explanations_id_list.append(single_row_id.split('|')[0])
            explanations_role_list.append(single_row_id.split('|')[1])
        filtered_explanations_id_list = explanations_id_list.copy()
        filtered_explanations_role_list = explanations_role_list.copy()
        for idx, filer_single in enumerate(explanations_id_list):
            if filer_single not in dict_explanations.keys():
                filtered_explanations_id_list.remove(filer_single)
                filtered_explanations_role_list.remove(explanations_role_list[idx])
        # assert len(explanations_id_list) == int(row['explanation_lenth'])
        non_explanations_list = []
        for each_row in dict_explanations.keys():
            if each_row not in filtered_explanations_id_list:
                non_explanations_list.append(each_row)
        final_rows_list = filtered_explanations_id_list + non_explanations_list
        random.shuffle(final_rows_list)
        for each_row_id in final_rows_list:
            if each_row_id in filtered_explanations_id_list:
                i_flag += 1
                each_row_true = dict_explanations[each_row_id]
                each_row_true = each_row_true.replace("''", '" ').replace("``", '" ')
                text_b = each_row_true
                guid = i_flag
                label = "1"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                debug_output_dict[str(i_flag)] = {'text_a': text_a, 'text_b': text_b, 'label': label, 'table_row_id': each_row_id}
            else:
                i_flag += 1
                each_row_false = dict_explanations[each_row_id]
                each_row_false = each_row_false.replace("''", '" ').replace("``", '" ')
                text_b = each_row_false
                guid = i_flag
                label = "0"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                debug_output_dict[str(i_flag)] = {'text_a': text_a, 'text_b': text_b, 'label': label, 'table_row_id': each_row_id}
        return examples, debug_output_dict

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
            #print('label id: ',label_id)
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            #logger.info("guid: %s" % (example.guid))
            logger.info("guid: %d" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              gu_id=example.guid))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "eprg":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "eprg": EprgProcessor
}

output_modes = {
    "eprg": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "eprg": 2
}
