# Office Answers
Organization-specific question answering system.

## Setup
Clone repository (and create virtualenv)

**Assumes virtualenv is installed. If not `pip install virtualenv` before continuing**

```sh
git clone https://github.com/JacobPolloreno/OfficeAnswers.git
cd OfficeAnswers
virtualenv venv
source venv/bin/activate
git clone https://github.com/JacobPolloreno/MatchZoo.git
cd MatchZoo
python setup.py install
cd ..
pip install -r build/requirements.txt
cd data/raw/
wget https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip
unzip WikiQACorpus.zip
```

## Requisites
- Python 3.6+
- virtualenv
- Keras(Tensorflow backend)
- Pandas
- Click
- PyTest


## Testing
```sh
   cd OfficeAnswers
   source venv/bin/activate
   python -m pytest
```

## TODO
* Make installation more smooth
* Add custom dataset
* Build REST API
