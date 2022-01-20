from tokenizers import BertWordPieceTokenizer


# prepare text files to train vocab on them
#files = ['/home/data_driven_project_enigma/ar.txt']
files = ['/home/sudanese_distilbert/theory_4.txt']
# train BERT tokenizer

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False,
)

trainer = tokenizer.train(
    files,
    vocab_size=30522,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

tokenizer.save("/home/sudanese_distilbert/vocab.txt", True)
# the first argument to save method is the path of the saving file, the second argument is boolean True
