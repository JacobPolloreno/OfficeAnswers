import os
import pytest

from officeanswers.ingestion import DSSMPrepare

file_dir = os.path.abspath(os.path.join(__file__, os.pardir))
test_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(test_dir, os.pardir))


@pytest.fixture(scope='module')
def data_path():
    data_dir = os.path.join(root_dir, 'data')
    return os.path.join(data_dir, 'tests/')


class TestDSSMPrepare(object):
    @pytest.fixture
    def prepare(self):
        return DSSMPrepare()

    def test_from_one_corpus(self, prepare, data_path):
        data_path = os.path.join(data_path, 'sample_data.txt')
        corpus_q, corpus_d, rels = prepare.from_one_corpus(data_path)

        for r in rels:
            assert len(r) == 3
            label, qid, did = r
            assert label is '0' or '1'
            assert did.startswith('D')
            assert qid.startswith('Q')
            assert corpus_d[did]
            assert corpus_q[qid]

    def test_traintest_split(self, prepare, data_path):
        data_path = os.path.join(data_path, 'sample_data.txt')
        corpus_q, corpus_d, rels = prepare.from_one_corpus(data_path)

        rel_train, rel_valid, rel_test = prepare.split_train_valid_test(rels)
