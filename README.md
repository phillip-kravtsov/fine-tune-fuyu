## Finetune Fuyu 8B

Software for fine-tuning Adept's [Fuyu Multimodal Model](https://www.adept.ai/blog/fuyu-8b) . To get started, install the requirements from `requirements.txt` into a new virtual environment (as this uses a fork of HF transformers for now, which may cause issues with your current installation).

Adapt the code to use your own dataset or download the AI2D data from [here](https://prior.allenai.org/projects/diagram-understanding) including test ids and put their location in `ai2d.py`.

Then run `scripts/surgery.py` to remove the 70,000 unused tokens from the vocabulary, then run `run.sh`. See config options in `config.py`. The full fine tune uses FSDP and takes about 3 hours on 4x RTX 4090.
