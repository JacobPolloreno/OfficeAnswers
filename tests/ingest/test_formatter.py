import pytest

from officeanswers.ingestion import DSSMFormatter


@pytest.fixture
def sample_data():
    corpus_q = {'Q01': 'Question 1', 'Q02': 'Question 2',
                'Q03': 'Question 3'}
    corpus_d = {'D01': 'Document 1', 'D02': 'Document 2',
                'D03': 'Document 3'}
    relations = [(0, 'Q01', 'D01'),
                 (1, 'Q01', 'D02'),
                 (0, 'Q01', 'D02'),
                 (0, 'Q02', 'D02'),
                 (0, 'Q03', 'D03'),
                 (1, 'Q02', 'D03'),
                 (1, 'Q03', 'D01')]
    inputs = {'questions': corpus_q,
              'documents': corpus_d}
    return inputs, relations


def test_from_inputs_teststage(sample_data):
    inputs, relations = sample_data
    df = DSSMFormatter().from_inputs(inputs, relations, stage='test')
    assert len(df.columns) == 4


def test_from_inputs_trainstage(sample_data):
    inputs, relations = sample_data
    df = DSSMFormatter().from_inputs(inputs, relations, stage='train')
    assert len(df.columns) == 5
