DATA_DIR="/workspace/"

wget -P "$DATA_DIR" http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip
wget -P "$DATA_DIR" https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv
wget -P "$DATA_DIR" https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf

unzip "$DATA_DIR/ai2d-all.zip" -d "$DATA_DIR"

mv "$DATA_DIR/ai2d_test_ids.csv" "$DATA_DIR/ai2d/"

python3 ../ai2d.py

