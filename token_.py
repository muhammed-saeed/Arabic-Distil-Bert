import os

for i in range(27):
    argss = 'python3 create_pretraining_data.py --input_file=gs://sudabert_bucket/sudanese_txt_cleaned_shuffled_splits/suda_txt_'+ str(i) +'.txt --output_file=gs://sudabert_bucket/tmp3/tf_examples.tfrecord_' + str(i) + ' --vocab_file=gs://sudabert_bucket/ar_bert_large/vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5'


    
    #input_file1 = 'gs://sudabert_bucket/sudanese_txt_cleaned_shuffled_splits/suda_txt_' + str(i) +'.txt '
    #output_file1 = 'output_file=gs://sudabert_bucket/tmp3/tf_examples.tfrecord_'+ str(i)
#
    os.system(argss)
