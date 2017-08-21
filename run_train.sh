python3 train.py --train-dataset-path=./data/flickr8k_train.pkl \
				 --val-dataset-path=./data/flickr8k_val.pkl \
				 --batch-size=1024 \
				 --epochs=99999 \
				 --embedding-dim=128 \
				 --checkpoint-period=1 \
				 --image-features-dim=4096