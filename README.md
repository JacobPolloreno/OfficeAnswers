# Office Answers
Organization-specific informational retrieval system.

#### Installation

```sh
git clone https://github.com/JacobPolloreno/OfficeAnswers.git
cd OfficeAnswers
```

##### Locally with virtualenv
**Assumes _virtualenv_ is installed. If not `pip install virtualenv`**

```sh
source venv/bin/activate
bash build/run_build.sh
```

##### AWS with _Conda env_ with TF + Keras Py36

```sh
source activate <CONDA_ENV_NAME>
bash build/aws_build.sh
```

#### How it works?
After you run the build script, WikiQA dataset was downloaded to ```data/raw```. 

The WikiQA dataset provides the main framework for learning question-answer pairs. It'll be augmented by your own custom dataset which you want to search. [See below to find out how to format your custom dataset.] 


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
python -m pytest
```
