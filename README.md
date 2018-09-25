# Office Answers
Organization-specific informational retrieval system.

#### Installation

**Assumes _virtualenv_ is installed. If not `pip install virtualenv`**

```sh
git clone https://github.com/JacobPolloreno/OfficeAnswers.git
cd OfficeAnswers
bash build/run_build.sh
```

##### Dependencies
- Python 3.6
- virtualenv
- Keras(Tensorflow backend)
- Pandas
- Click
- PyTest

See More dependencies in ``build/requirements.txt``

### How it works?
After you run the build script found in ```build/run_build.sh```, you'll find that the WikiQA dataset was downloaded into ```data/raw```. The WikiQA dataset provide the main framework for learning question-answer pairs. It'll be augmented by your own custom dataset which you want to search. See header below to find out how to format your custom dataset. 


#### Step 1: Configuration

Create a copy of config file
```sh
cd configs
cp sample.config custom.config
#edit custom.config
```
* Modify line 14 "custom_corpus" with the path to your custom dataset. **recommend placing the dataset in ```data/raw``` folder**
	- e.g. "custom_corpus": "./data/raw/custom_corpus.txt"

#### Step 2: Prepare and Preprocess
```sh
cd OfficeAnswers
python src/officeanswers/main.py configs/custom.config prepare_and_preprocess
```

#### Step 3: Train
```sh
cd OfficeAnswers
python src/officeanswers/main.py configs/custom.config train
```

#### Step 4: Predict
```sh
cd OfficeAnswers
python src/officeanswers/main.py configs/custom.config predict
```

### How should my data be formatted?
Raw data should be tab-seperated in the follow format:

_<QUESTION\>\t<ANSWER\>\n_
	
```
how are glacier caves formed ?	A partly submerged glacier cave on Perito Moreno Glacier .
how are glacier caves formed ?	The ice facade is approximately 60 m high
```
### Testing
```sh
cd OfficeAnswers
source venv/bin/activate
python -m pytest
```
