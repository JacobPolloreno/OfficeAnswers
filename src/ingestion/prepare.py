"""Prepare data for preprocessing and model training

Some code from:
https://github.com/faneshion/MatchZoo/blob/master/matchzoo/inputs/preparation.py
"""

import abc
import hashlib
import logging
import random
import typing


logger = logging.getLogger(__name__)


class BasePrepare(abc.ABC):
    """Base Prepare class to prepare data
    """

    @staticmethod
    def save_relations(save_path: str, relations: dict) -> None:
        try:
            with open(save_path, 'w') as f:
                for rel in relations:
                    l, qid, did = rel
                    f.write(f"{l} {qid} {did}")
        except Exception:
            logger.error("Error saving relations")
            raise

    @staticmethod
    def split_train_valid_test(relations: dict,
                               ratio: typing.Tuple=(0.8, 0.1, 0.1)):
        if not isinstance(ratio, tuple) and len(ratio) == 3:
            error_msg = f"{ratio} is not a tuple"
            logger.error(error_msg)
            raise ValueError(error_msg)

        qid_set = set()
        for r, q, d in relations:
            qid_set.add(q)
        qid_group = list(qid_set)

        random.shuffle(qid_group)
        total_rel = len(qid_group)
        num_train = int(total_rel * ratio[0])
        num_valid = int(total_rel * ratio[1])
        valid_end = num_train + num_valid

        qid_train = qid_group[: num_train]
        qid_valid = qid_group[num_train: valid_end]
        qid_test = qid_group[valid_end:]

        def select_rel_by_qids(qids):
            rels = []
            qids = set(qids)
            for r, q, d in relations:
                if q in qids:
                    rels.append((r, q, d))
            return rels

        rel_train = select_rel_by_qids(qid_train)
        rel_valid = select_rel_by_qids(qid_valid)
        rel_test = select_rel_by_qids(qid_test)

        return rel_train, rel_valid, rel_test


class DSSMPrepare(BasePrepare):
    """Format data for preprocessing and model training

    Corpus should follow seperated format:

        Label, Question, Sentence

    Usage:
        >>> base_dir = './data/raw/'
        >>> data_path = os.path.join(base_dir, 'data.txt')
        >>> prepare = DSSMPrepare()
    """

    def _parse_line(self, line, delimiter='\t') -> typing.Tuple:
        tokens = line.split(delimiter)
        if len(tokens) != 3:
            error_msg = "Format of data is wrong." + \
                "Should be 'label,text1,text2'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return tuple(tokens)

    def _get_text_id(self, hashid, text, idtag='T') -> str:
        hash_obj = hashlib.sha1(text.encode('utf8'))
        hex_dig = hash_obj.hexdigest()
        if hex_dig in hashid:
            return hashid[hex_dig]
        else:
            tid = idtag + str(len(hashid))
            hashid[hex_dig] = tid
            return tid

    def from_one_corpus(self, path: str) -> typing.Tuple:
        """Format dataset for preprocessing

        Args:
            path (str): Path to corpus.txt file
        """
        logger.info("Building dataframe from delimeted file")

        hashid_q: dict = {}
        hashid_d: dict = {}
        corpus_q = {}
        corpus_d = {}
        relations = []
        try:
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    label, q, d = self._parse_line(line)
                    qid = self._get_text_id(hashid_q, q, 'Q')
                    did = self._get_text_id(hashid_d, d, 'D')
                    corpus_q[qid] = q
                    corpus_d[did] = d
                    relations.append((label, qid, did))
        except IOError:
            logger.error(f"Error opening file `{path}`")
            raise
        return corpus_q, corpus_d, relations
