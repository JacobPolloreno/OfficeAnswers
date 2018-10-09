import abc
import pandas as pd
import logging
import os
import typing

logger = logging.Logger(__name__)


class BaseFormatter(abc.ABC):
    """Base format helper
    """

    @classmethod
    @abc.abstractmethod
    def from_files(cls,
                   files: typing.Dict[str, str],
                   stage: str) -> pd.DataFrame:
        """Transform inputs to expected structure

        Args:
            files (list): list of file paths with raw inputs and relations
            stage (str): stage of model influences structure of data

        Return:
            DataFrame: structured inputs
        """

    @classmethod
    @abc.abstractmethod
    def from_inputs(
            cls,
            inputs: typing.Dict[str, typing.Dict[str, str]],
            relations: typing.List[typing.Tuple[str, str, str]],
            stage: str) -> pd.DataFrame:
        """Transform inputs to expected structure

        Args:
            inputs (dict): dict of dict data
            stage (str): stage of model influences structure of data

        Return:
            DataFrame: structured inputs
        """

    @staticmethod
    def check_files(files: typing.List[str]):
        for f in files:
            if not os.path.exists(f):
                error_msg = f"`{f}` cannot be found." + \
                    "File should be generated using Prepare"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            if not os.path.isdir(f):
                error_msg = f"`{f}`should not be a directory"
                logger.error(error_msg)
                raise IOError(error_msg)
            if os.stat("file").st_size == 0:
                error_msg = f"`{f}` is empty"
                logger.error(error_msg)
                raise ValueError(error_msg)


class DSSMFormatter(BaseFormatter):
    """DSSM format helper to transform inputs to expected manner

    Example:

        label questionID(QID) documentID(DID) ->

        output = [(QID, DID, QUESTION, DOCUMENT, LABEL), ...]

    """

    @classmethod
    def from_inputs(
            cls,
            inputs: typing.Dict[str, typing.Dict[str, str]],
            relations: typing.List[typing.Tuple[str, str, str]],
            stage: str) -> pd.DataFrame:
        """Transform inputs to expected structure

        Args:
            inputs (dict): dict of dict data with questions, answers and
                relations
            stage (str): stage of model influences structure of data

        Return:
            DataFrame: structured inputs

        Usage:
            >>> corpus_q, corpus_d, relations = Prepare().from_one_corpus(...)
            >>> inputs = {'questions': corpus_q, 'documents': corpus_d}
            >>> train_inputs = DSSMFormatter.from_inputs(
            ...    inputs, relations, stage='train')
        """
        try:
            corpus_q, corpus_d = inputs['questions'], inputs['documents']
        except KeyError as e:
            error_msg = "DSSM needs question and answer pairs." + \
                "Requires mappings for questions and documents" + \
                f"KeyError: {e}"
            logger.error(error_msg)
            raise

        output = []
        for rel in relations:
            label, qid, did = rel
            try:
                row = [qid, did, corpus_q[qid], corpus_d[did]]
                if stage == 'train':
                    row.append(label)
                output.append(tuple(row))
            except KeyError as e:
                logger.error(f"KeyError with {e} while formatting data")
                raise
        columns = ['QuestionID', 'DocumentID', 'Question', 'Document']
        if stage == 'train':
            columns.append('Label')
        return pd.DataFrame(output, columns=columns)

    @classmethod
    def from_files(cls,
                   files: typing.Dict[str, str],
                   stage: str) -> pd.DataFrame:
        """Transform inputs to expected structure

        Args:
            files (list): list of file paths containing questions, answers and
                relations
            stage (str): stage of model influences structure of data

        Return:
            DataFrame: structured inputs

        Usage:
            >>> base_data_dir = os.path.abspath('data/')
            >>> files = {'questions': 'questions.txt'
            ...           'documents': 'documents.txt',
            ...            'relations': 'relations.txt'}
            >>> file_paths = {k: os.path.join(base_data_dir, v) for k, v in
            ...               files.items()}
            >>> train_inputs = DSSMFormatter.from_files(files, stage='train')
        """
        cls.check_files(list(files.values()))
        try:
            relations = []
            with open(files['relations'], 'r') as f:
                for line in f:
                    line = line.strip()
                    tokens = line.split()
                    if len(tokens) != 3:
                        error_msg = "Invalid format for relation text." + \
                            "Should be `label questionID documentID`"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    relations.append((tokens[0], tokens[1], tokens[2]))

            corpus_q = {}
            with open(files['questions'], 'r') as f:
                for line in f:
                    line = line.strip()
                    tokens = line.split()
                    if len(tokens) < 2:
                        error_msg = "Invalid format for question text." + \
                            "Should be `questionID question"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    corpus_q[tokens[0]] = "".join(tokens[1:])

            corpus_d = {}
            with open(files['documents'], 'r') as f:
                for line in f:
                    line = line.strip()
                    tokens = line.split()
                    if len(tokens) < 2:
                        error_msg = "Invalid format for document text." + \
                            "Should be `documentID document"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    corpus_d[tokens[0]] = "".join(tokens[1:])
        except FileNotFoundError as e:
            logger.error("FileNotFoundError, check paths for files.")

        inputs = {'questions': corpus_q,
                  'documents': corpus_d}
        return cls.from_inputs(inputs, relations, stage=stage)
