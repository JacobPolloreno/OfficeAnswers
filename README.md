[SLIDES](https://drive.google.com/open?id=1GZxIdP2MOtHSkg2Qy3kq2UzXu_VVXDwCN1A1epawk7Y) 
# WorkBuddy

<p align='center'>
<img src='./assets/sys_arch.png'  alt="WorkBuddy System Architecture"/>
</p>

Workbuddy helps you get to the real work by create a search engine for company information. It leverages neural information retrieval to build embeddings, the cache and serve them through a REST API.

#### Installation

```sh
git clone https://github.com/JacobPolloreno/OfficeAnswers.git
cd OfficeAnswers
```

##### Locally with virtualenv
###### **Assumes _virtualenv_ is installed. If not `pip install virtualenv`**

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
python src/main.py configs/custom.config prepare_and_preprocess
```

#### Step 3: Train
```sh
cd OfficeAnswers
python src/cli.py configs/custom.config train
```

#### Step 4: Predict
```sh
cd OfficeAnswers
python src/cli.py configs/custom.config predict
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
