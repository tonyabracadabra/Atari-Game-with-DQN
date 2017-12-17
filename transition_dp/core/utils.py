"""Common functions you may find useful in your implementation."""

import semver
import tensorflow as tf
import re
import numpy as np
import random
from string import punctuation


def get_uninitialized_variables(variables=None):
    """Return a list of uninitialized tf variables.

    Parameters
    ----------
    variables: tf.Variable, list(tf.Variable), optional
      Filter variable list to only those that are uninitialized. If no
      variables are specified the list of all variables in the graph
      will be used.

    Returns
    -------
    list(tf.Variable)
      List of uninitialized tf variables.
    """
    sess = tf.get_default_session()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)

    if len(variables) == 0:
        return []

    if semver.match(tf.__version__, '<1.0.0'):
        init_flag = sess.run(
            tf.pack([tf.is_variable_initialized(v) for v in variables]))
    else:
        init_flag = sess.run(
            tf.stack([tf.is_variable_initialized(v) for v in variables]))

    return [v for v, f in zip(variables, init_flag) if not f]


# Tears of the debugging...
def initialize_updates_operations(target_vars):
    # placeholders for updating the online network
    update_phs = [tf.placeholder(tf.float32, shape=var.get_shape()) for var in target_vars]
    # update operations
    update_ops = [update_pair[0].assign(update_pair[1])
                  for update_pair in zip(target_vars, update_phs)]

    return update_phs, update_ops


def get_soft_target_model_updates(target, source, tau):
    """Return list of target model update ops.

    These are soft target updates. Meaning that the target values are
    slowly adjusted, rather than directly copied over from the source
    model.

    The update is of the form:

    $W' \gets (1- \tau) W' + \tau W$ where $W'$ is the target weight
    and $W$ is the source weight.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.
    tau: float
      The weight of the source weights to the target weights used
      during update.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """

    weights = map(lambda x, y: x + y, zip(map(lambda w: (1 - tau) * w, target.get_weights()),
                                          map(lambda w: tau * w, source.get_weights())))

    target.set_weights(weights)


def get_hard_target_model_updates(target, source):
    """Return list of target model update ops.

    These are hard target updates. The source weights are copied
    directly to the target network.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    # updating the parameters from the previous network
    target.set_weights(source.get_weights())


'''
IO Utils

'''


UNK = "<UNK>"
NUM = "<NUM>"
NULL = '<NULL>'
ROOT = '<ROOT>'


def read_tag_list(file_name):
    tag_list = []
    with open(file_name) as reader:
        for line in reader:
            line = line.strip()
            tag_list.append(line)
    return tag_list


def data_generator(sentences):
    while True:
        for sentence in sentences:
            yield sentence


class TokManager:
    POS_MARKER = "<POS>"
    LABEL_MARKER = "<LABEL>"
    isdigit = re.compile(r"^[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?$")

    def __init__(self, args):
        word_embedding, word_dict = create_word_embedding(args.embedding_file, args.embedding_size)

        word_list = read_tag_list(args.word_list)
        tag_list = read_tag_list(args.pos_tag)
        label_list = read_tag_list(args.dependency_label_file)
        self.unified_dict = {}
        for l in label_list:
            self.unified_dict[self.LABEL_MARKER + l] = len(self.unified_dict)
        self.unified_dict[self.LABEL_MARKER + NULL] = self.L_NULL = len(self.unified_dict)
        for pos in tag_list:
            self.unified_dict[self.POS_MARKER + pos] = len(self.unified_dict)
        self.unified_dict[self.POS_MARKER + NULL] = self.P_NULL = len(self.unified_dict)
        self.unified_dict[self.POS_MARKER + ROOT] = self.P_ROOT = len(self.unified_dict)
        for w in word_list:
            self.unified_dict[w] = len(self.unified_dict)
        for punc in punctuation:
            if punc not in self.unified_dict:
                self.unified_dict[punc] = len(self.unified_dict)
        self.unified_dict[NUM] = len(self.unified_dict)
        self.unified_dict[UNK] = len(self.unified_dict)
        self.unified_dict[NULL] = self.NULL = len(self.unified_dict)
        self.unified_dict[ROOT] = self.ROOT = len(self.unified_dict)

        self.embeddings_matrix = np.random.normal(0, 0.9, (len(self.unified_dict), args.embedding_size))
        for w in list(set(word_list + [p for p in punctuation])):
            if w in word_dict:
                self.embeddings_matrix[self.unified_dict[w], :] = word_embedding[word_dict[w], :]

        self.word_dict = word_dict

    def get_pos_id(self, pos):
        return self.unified_dict[self.POS_MARKER + pos]

    def get_label_id(self, label):
        return self.unified_dict[self.LABEL_MARKER + label]

    def get_word_id(self, word):
        word = word.lower()
        if word in self.unified_dict:
            return self.unified_dict[word]
        elif self.isdigit.match(word) is not None:
            return self.unified_dict[NUM]
        return self.unified_dict[UNK]


class ConllData:
    def __init__(self, tok_manager):
        self.tok_manager = tok_manager

    def read_conll(self, in_file):
        data = []
        with open(in_file) as f:
            word, pos, head, label = [], [], [], []
            for line in f.readlines():
                sp = line.strip().split('\t')
                if len(sp) == 10:
                    if '-' not in sp[0]:
                        word.append(self.tok_manager.get_word_id(sp[1]))
                        pos.append(self.tok_manager.get_pos_id(sp[4]))
                        head.append(int(sp[6]))
                        label.append(self.tok_manager.get_label_id(sp[7]))
                elif len(word) > 0:
                    data.append({
                        'word': [self.tok_manager.ROOT] + word,
                        'pos': [self.tok_manager.P_ROOT] + pos,
                        'head': [-1] + head, 'label': [-1] + label
                    })
                    word, pos, head, label = [], [], [], []
            if len(word) > 0:
                data.append({'word': word, 'pos': pos, 'head': head, 'label': label})
        return data


def read_word_embedding(file_name):
    word_embedding_word = []
    word_embedding_dict = {}
    with open(file_name) as word_embedding:
        for line in word_embedding:
            line = line.strip()
            current_word = line.split()[0].lower()
            word_embedding_word.append(current_word)
            word_embedding_dict[current_word] = line.split()[1:]
    return word_embedding_word, word_embedding_dict


def create_word_embedding(embedding_file, embeds_dim):
    word_list, word_dict = read_word_embedding(embedding_file)
    word_index_dict = {}
    word_embedding = np.random.normal(scale=0.01, size=(len(word_dict) + len(word_index_dict), embeds_dim))
    for w in word_list:
        word_index_dict[w] = len(word_index_dict)
        assert len(word_dict[w]) == embeds_dim
        word_embedding[word_index_dict[w], :] = np.asarray(word_dict[w], dtype=np.float64)
    return word_embedding, word_index_dict


def minibatches(data, batch_size, shuffle=True):
    if shuffle: random.shuffle(data)
    n_batch = len(data) // batch_size
    new_data = data[: n_batch * batch_size]
    for i in range(0, len(new_data), batch_size):
        batch_data = new_data[i: i + batch_size]
        batch_x = np.array([x[0] for x in batch_data])
        batch_y = np.array([x[1] for x in batch_data])
        yield batch_x, batch_y



