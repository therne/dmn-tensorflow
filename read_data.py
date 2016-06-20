""" a neat code from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/ """
import os

from utils.data_utils import DataSet
from copy import deepcopy


def load_babi(data_dir, task_id, type='train'):
    """ Load bAbi Dataset.
    :param data_dir
    :param task_id: bAbI Task ID
    :param type: "train" or "test"
    :return: dict
    """
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    file_name = [f for f in files if s in f and type in f][0]

    # Parsing
    tasks = []
    skip = False
    curr_task = None
    for i, line in enumerate(open(file_name)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            skip = False
            curr_task = {"C": [], "Q": "", "A": ""}

        # Filter tasks that are too large
        if skip: continue
        if task_id == 3 and id > 130:
            skip = True
            continue

        elif task_id != 3 and id > 70:
            skip = True
            continue

        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ') + 1:]
        if line.find('?') == -1:
            curr_task["C"].append(line)
        else:
            idx = line.find('?')
            tmp = line[idx + 1:].split('\t')
            curr_task["Q"] = line[:idx]
            curr_task["A"] = tmp[1].strip()
            tasks.append(deepcopy(curr_task))

    print("Loaded {} data from bAbI {} task {}".format(len(tasks), type, task_id))
    return tasks


def process_babi(raw, word_table):
    """ Tokenizes sentences.
    :param raw: dict returned from load_babi
    :param word_table: WordTable
    :return:
    """
    questions = []
    inputs = []
    answers = []
    fact_counts = []

    for x in raw:
        inp = []
        for fact in x["C"]:
            sent = [w for w in fact.lower().split(' ') if len(w) > 0]
            inp.append(sent)
            word_table.add_vocab(*sent)

        q = [w for w in x["Q"].lower().split(' ') if len(w) > 0]

        word_table.add_vocab(*q, x["A"])

        inputs.append(inp)
        questions.append(q)
        answers.append(x["A"])  # NOTE: here we assume the answer is one word!
        fact_counts.append(len(inp))

    return inputs, questions, answers, fact_counts


def read_babi(data_dir, task_id, type, batch_size, word_table):
    """ Reads bAbi data set.
    :param data_dir: bAbi data directory
    :param task_id: task no. (int)
    :param type: 'train' or 'test'
    :param batch_size: how many examples in a minibatch?
    :param word_table: WordTable
    :return: DataSet
    """
    data = load_babi(data_dir, task_id, type)
    x, q, y, fc = process_babi(data, word_table)
    return DataSet(batch_size, x, q, y, fc, name=type)


def get_max_sizes(*data_sets):
    max_sent_size = max_ques_size = max_fact_count = 0
    for data in data_sets:
        for x, q, fc in zip(data.xs, data.qs, data.fact_counts):
            for fact in x: max_sent_size = max(max_sent_size, len(fact))
            max_ques_size = max(max_ques_size, len(q))
            max_fact_count = max(max_fact_count, fc)

    return max_sent_size, max_ques_size, max_fact_count
