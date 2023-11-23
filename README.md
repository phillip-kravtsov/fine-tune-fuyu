## Finetune Fuyu 8B

Software for fine-tuning Adept's [Fuyu Multimodal Model](https://www.adept.ai/blog/fuyu-8b) . To get started, install the requirements from `requirements.txt` into a new virtual environment (as this uses a fork of HF transformers for now, which may cause issues with your current installation).

Adapt the code to use your own dataset, or use one of the existing datasets (scienceqa, textvqa, ai2d).

Then run `scripts/surgery.py` to remove the 70,000 unused tokens from the vocabulary, then run `scripts/run.sh`. See config options in `config.py`. The full fine tune uses FSDP and can technically be run on 4x 24GB GPU's but is quite slow depending on GPU interconnect. It can also be run on two 40GB GPU's.

Since the Fuyu architecture is fully causal over image and text, one can presumably use it to predict patches and potentially generate / debug the model's conception of images. There is some code for this, but not the main purpose of the repo.

Thanks to the QLoRA repo + Stas Bekman for some code.
