# Q-learning strategies
This project provides a clasical deep-q-learning strategy for solving diverse games.
It uses gym-retro for the emulation of games and the environment.
The files training.py and training-smb.py provides a framework for Super Mario World and Super Mario Bros games.
## Installation 
- Supported Pythons:
    - 3.5
    - 3.6

- Required packages:
    - numpy, matplotlib, tensorflow, keras, docopt, gym-retro

    These packages can be installed using pip (with ```pip install <package_name>```).
    However, tensorflow is no more available on pip.
    Use ```pip install --upgrade  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl```
    (see more [here](https://www.tensorflow.org/install/pip)); be aware of your OS version to choose the right tensorflow's version.

- Install Super Mario World and Super Mario Bros
The files in the directories SuperMarioX in this project (data and scenario json files) have
to be put in the corresponding gym-retro directory.
    - ***Where is located my gym-retro installation ?***
    
    Simply run a python interpreter and execute
    ```python
  import retro
  retro.__path__
    ```
	to know your gym-retro installation location
	- ***Where to put the scenarios and data files ?***
	
	Go to the directory found on the previous step (e.g., ```PATH_TO_PYTHON/python3.X/site-packages/retro```) and
	and then go to the data/stable directory. All games supported by gym-retro are located here.
	In our case, go to SuperMarioWorld-Snes or SuperMarioBros-Nes and put the scenario et data json files in it.

- ***Run the training ?***

Simply run ```training.py``` for SuperMarioWorld or ```training-smb.py``` for SuperMarioBros

- Change the reward function

	- The file data.json contains RAM addresses linked to some game informations.
	- The file scenario.json contains the rewards of states reaching these game informations.

