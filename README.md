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
pip install -r build/requirements.txt
```

## Requisites
- Python 3.6+
- virtualenv
- Keras(Tensorflow backend)
- Click
- PyTest


## Testing
```sh
   cd OfficeAnswers
   source venv/bin/activate
   python -m pytest
```
