python /content/TinyBERT/general_distill.py --pregenerated_data /content/train_data \
  --teacher_model /content/bert \
  --student_model /content \
  --reduce_memory --do_lower_case \
  --train_batch_size 256 \
  --output_dir /content/tiny \
  --num_train_epochs 8