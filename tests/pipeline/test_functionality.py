import os
import pytest

from officeanswers.model import train
from officeanswers.model import predict
from officeanswers.preprocess import prepare_and_preprocess
from officeanswers.util import Config

from matchzoo import engine

file_dir = os.path.abspath(os.path.join(__file__, os.pardir))


@pytest.fixture(scope='module')
def config(tmpdir_factory):
    cfg = Config()
    cfg.net_name = 'test'
    cfg.paths = {}
    cfg.paths['preprocess_dir'] = tmpdir_factory.mktemp('preprocess')
    cfg.paths['processed_dir'] = tmpdir_factory.mktemp('processed')
    cfg.inputs = {}
    cfg.inputs['share'] = {}
    cfg.inputs['train'] = {}
    cfg.inputs['share']['raw_corpus'] = os.path.join(file_dir, 'test_data.txt')
    cfg.inputs['train']['steps_per_epoch'] = None
    cfg.inputs['train']['epochs'] = 1
    cfg.inputs['train']['verbose'] = 0
    cfg.model = {}
    cfg.model['type'] = 'dssm'
    cfg.outputs = os.path.join(tmpdir_factory.getbasetemp(), 'out.txt')
    return cfg


class TestPipelineFunctionality(object):
    def test_config_file(self, config):
        assert config.net_name == 'test'
        assert os.path.exists(config.inputs['share']['raw_corpus'])
        assert config.paths['preprocess_dir']
        assert config.paths['processed_dir']

    def test_prepare(self, config):
        prepare_and_preprocess(config)
        pp_dir = config.paths['preprocess_dir']
        pp_files = ["train", "test", "valid",
                    "preprocessor", "questions",
                    "documents"]
        pp_files = [f"{config.net_name}_{name}.dill" for name in pp_files]
        pp_file_paths = [os.path.join(pp_dir, f) for f in pp_files]
        for file_path in pp_file_paths:
            print(file_path)
            assert os.path.exists(file_path)

    def test_train(self, config):
        metrics = train(config)
        p_dir = config.paths['processed_dir']
        weights_file = os.path.join(p_dir, f"{config.net_name}.h5")
        params_file = os.path.join(p_dir, f"{config.net_name}_params.dill")
        assert metrics
        assert os.path.exists(weights_file)
        assert os.path.exists(params_file)
        with open('pytest_train_res.txt', 'w') as f:
            for m in metrics:
                f.write(f"{m[0]}\t{m[1]}\n")

    def test_predict(self, config):
        p_dir = config.paths['processed_dir']
        model = engine.load_model(p_dir, config.net_name)
        results = predict(config, model, "glacier caves", 5)
        assert len(results) == 5
        with open(config.outputs, 'w') as f:
            for r in results:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
