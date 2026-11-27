# Training configuration
INPUT_LEN = 2097152
PAD_TOKEN = 256
EXCLUDE_HEADER = True

# Training
EPOCHS = 20
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 0.0
NUM_WORKERS = 8
OPTIMIZER = "Adam"
LOSS = "CrossEntropyLoss"
DEVICE = "cuda"
OUT_CKPT = "best_malconv_v3.pt"
HISTORY_CSV = "train_history.csv"
PLOT_DIR = "training_plots"
