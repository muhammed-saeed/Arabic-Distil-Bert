python3 /home/CE/musaeed/Pretrained-Language-Model/TinyBERT/pregenerate_training_data.py \
  --train_corpus /home/CE/musaeed/ar/xaa \
  --bert_model /home/CE/musaeed/Tiny_BERT/BERT_FILE \
  --do_lower_case \
  --epochs_to_generate 3 \
  --num_workers 1 --reduce_memory \
  --output_dir /home/CE/musaeed/train_data_tiny_bert