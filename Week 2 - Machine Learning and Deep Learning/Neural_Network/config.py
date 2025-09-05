# 📂 Data paths
train_folder = 'Train_Full'
test_folder = 'Test_Full'

# 🔡 Text preprocessing
vocab_size = 20000   # Limit vocab size (VNTC ~30k words)
max_length = 300     # Max sequence length for padding/truncating

# 🧠 Model hyperparameters
dropout_rate = 0.3
optimizer_algorithm = 'adam'

# 🎯 Training parameters
epoch_size = 10
batch_size = 32
training_validation_split = 0.2