import os
import stanza
from stanza.utils.training import run_ner

# 1. Define your paths (ensure your BIOES formatted data is in these directories)
DATA_DIR = "./ner_data"
TRAIN_FILE = f"{DATA_DIR}/train.bioes"
DEV_FILE = f"{DATA_DIR}/dev.bioes"
MODEL_SAVE_DIR = "./saved_models"

# 2. Configure the hyperparameters based on the research paper's baseline
# We will use RoBERTa as the transformer baseline
training_args = [
    "--data_dir", DATA_DIR,
    "--train_file", TRAIN_FILE,
    "--eval_file", DEV_FILE,
    "--shorthand", "en_worldwide", 
    "--mode", "train",
    "--save_dir", MODEL_SAVE_DIR,
    
    # Transformer configuration
    "--charlm",  # Often used alongside embeddings in Stanza
    "--bert_model", "roberta-large", 
    
    # Hyperparameters from the study's Appendix C
    "--optimizer", "sgd",
    "--learning_rate", "0.1",
    "--lr_decay", "0.5",
    "--batch_size", "32",
    "--max_grad_norm", "5.0",
    "--hidden_dim", "256",
    "--num_layers", "1",
    "--dropout", "0.5",
    "--word_dropout", "0.01",
    
    # Early termination condition (learning rate reaching 0.0001)
    "--early_stopping_tolerance", "0.0001" 
]

# 3. Run the training module
if __name__ == "__main__":
    print("Starting Stanza NER training with RoBERTa-Large...")
    # This invokes Stanza's internal NER training routine
    run_ner.main(training_args)