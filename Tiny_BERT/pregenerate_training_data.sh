python3 /home/CE/musaeed/Pretrained-Language-Model/TinyBERT/pregenerate_training_data.py \
  --train_corpus /home/CE/musaeed/ar/ar_part1-36.txt \
  --reduce_memory \
  --bert_model /home/CE/musaeed/Tiny_BERT/BERT_FILE \
  --do_lower_case \
  --num_workers 1 \
  --epochs_to_generate 3 \
  --output_dir /home/CE/musaeed/train_data_tiny_bert_ar