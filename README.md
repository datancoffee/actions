## Data App Actions Framework

Simple framework of typical data app actions (data readers, transforms, inference, data writers) for Python.


## Install dependencies

Install pytoch locally
```
pip3 install torch torchvision torchaudio
```

Install some extra libs
```
pip3 install scipy numpy pandas pyarrow
```

Extra libs for common data transforms
```
pip3 install jmespath
```

Install dltHub
```
pip3 install dlt
```

Install additional readers for Action development
```
pip3 install pipe langchain
```

Install Huggingface
```
pip3 install transformers accelerate
```

## Run examples

Go to the repository directory
```
cd actions
```

Set Huggingface token if you intend to use examples that use Huggingface API

```
export HF_TOKEN=<your-hf-token>
```


### Example: Generate a meaningful technical response to a Github issue comments thread
This example can be thought of as a bot proposing an answer to the thread of issue comments on GitHub.

**Uses**: HuggingFace model API, local Pytorch, local meta-llama/Meta-Llama-3-8B-Instruct inference, Apple MPS device (can be switched to cpu)

**Input**: A particular GitHub issue (#933 of the dlthub repo) will be downloaded with all its comments

**Output**: Files will be written to the "out" directory, containing the proposed response and the full message chat 

```
python3 examples/github_bot.py
```

### Example: Classify all images in a folder
The following example runs every image file in data/flowers-102-100 folder
through Imagenet Labeler. 

**Uses**: local PyTorch, Apple "mps" GPU device (can run with cpu as well).

**Input**: image files in data/flowers-102-100 folder

**Output**: Standard Output will contain probabilities of image classes e.g. 'bee[0.564373791217804], daisy[0.07953538745641708]'

```
python3 examples/pytorch_image_labeler.py
```


## Troubleshooting

### ModuleNotFoundError error

If you get an error when executing one of the examples
```
ModuleNotFoundError: No module named 'core'
```
just add to the Python Path
```
export PYTHONPATH=.:$PYTHONPATH
```

### NotImplementedError on MPS device error

If you are trying to do local Pytorch inference on Apple silicone and get an error when executing one of the examples
```
NotImplementedError: The operator 'aten::isin.Tensor_Tensor_out' is not currently implemented for the MPS device.
```
just run this before running the example script
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

