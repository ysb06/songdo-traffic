# songdo_metr

## Get Started

This project provides a script that automates the process of downloading data from the original sources, including the Korean Standard Node-Link and Incheon City traffic data, and converting it into a usable dataset.

### Installing Requirements
#### Using PDM

```bash
pdm install
```

#### Not Using PDM

```bash
pip install -r requirements.txt
```

### Running the Code

1. Set Environment Variables

To access Incheon City traffic data, you need to sign up for data.go.kr and register your API Key as an environment variable. Since this project uses a src layout, set the environment variables as shown below:

```bash
export PYTHONPATH="./src"
export PDP_KEY = "[API Key for data.go.kr]"
```

Alternatively, you can create a .env file in the project folder with the following content:

```
PYTHONPATH="./src"
PDP_KEY = "[API Key for data.go.kr]"
```

2. Run the Code

```bash
python -m metr.dataset
```