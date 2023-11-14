DATA_DIR="/workspace/"
wget -P "$DATA_DIR" https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
wget -P "$DATA_DIR" https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
wget -P "$DATA_DIR" https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

unzip "$DATA_DIR/train_val_images.zip" -d "$DATA_DIR"
mv "$DATA_DIR/train_val_images" "$DATA_DIR/textvqa/train_val_images"
mv "$DATA_DIR/TextVQA_0.5.1_train.json" "$DATA_DIR/textvqa/train.json"
mv "$DATA_DIR/TextVQA_0.5.1_val.json" "$DATA_DIR/textvqa/val.json"
