# preprocessing
python processing.py

# SVM
python classic_classifier.py --entity-type aff
python classic_classifier.py --entity-type venue
python classic_classifier.py --entity-type author

# RNN-based matching
python train_rnn_match.py --n-try 5 --entity-type aff --n-seq 1
python train_rnn_match.py --n-try 5 --entity-type venue
python train_rnn_match.py --n-try 5 --entity-type author

# CNN-based matching
python train_cnn_match.py --n-try 5 --entity-type aff
python train_cnn_match.py --n-try 5 --entity-type venue
python train_cnn_match.py --n-try 5 --entity-type author

# HGAT-based matching
python train_hgat_match.py --n-try 5 --entity-type author
