from transformers import DistilBertModel, DistilBertConfig
# Initializing a DistilBERT configuration
configuration = DistilBertConfig()
# Initializing a model from the configuration
model = DistilBertModel(configuration)
# Accessing the model configuration
configuration = model.config
model.save_pretrained('/content/')
