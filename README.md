# START TREK

The goal of this project is to create 3 algorithm of deep learning, and to optimize them for the "lunar-lander-v3" environnement.

## Installation

Fist thing first, clone this repo :
```bash
  git clone git@github.com:EpitechPGE2-2025/G-AIA-401-PAR-4-1-starttrek-3.git START_TREK
  cd START_TREK
```
To run START_TREK,  you must install python.

If you are on Ubuntu 16.10 or newer :
```bash
  sudo apt-get update
  sudo apt-get install python3.6
```
If you’re using another version of Ubuntu (e.g. the latest LTS release) or you want to use a more current Python, we recommend using the deadsnakes PPA to install Python 3.8
```bash
  sudo apt-get install software-properties-common
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install python3.8
```
If you are using other Linux distribution, chances are you already have Python 3 pre-installed as well. If not, use your distribution’s package manager. For example on Fedora, you would use dnf:
```bash
  sudo dnf install python3
```

Then, you will need to install pip :
```bash
  sudo apt install python3-pip
```
Finnaly, you will need to install all the dependencies for this project:
```bash
  pip install -r requirements.txt
```
You will probably need to create a virtual environnement (venv) to run this project.
If it's a case, follow this documentation :

https://docs.python.org/3/library/venv.html

## Running the algorithms

Go to the directory of the algorithm you want to try:
```bash
  cd Proximal_Policy_Optimization
```

If you want to train the model and evaluate it , run:
```bash
  python main.py
```

If you only want to train the model :
```bash
  python train.py
```

If you only want to evaluate it :
```bash
  python eval.py
```
And that's it !
You will find the video footages into the video folders (ppo_lunar_lander_videos for example).