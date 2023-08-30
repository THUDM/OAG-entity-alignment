model_name=glm-roberta
task=venue
lr=1e-5
bs=64
len=64

albert=/home/shishijie/workspace/PTMs/albert-base-v2
bert=/home/shishijie/workspace/PTMs/bert-base-uncased
roberta_base=/home/shishijie/workspace/PTMs/xlm-roberta-base
roberta_large=/home/shishijie/workspace/PTMs/xlm-roberta-large
deberta_base=/home/shishijie/workspace/PTMs/deberta-base
deberta_v3_large=/home/shishijie/workspace/PTMs/deberta-v3-large
LaBSE=/home/shishijie/workspace/PTMs/LaBSE
glm_roberta=/home/shishijie/workspace/PTMs/glm-roberta-large


CUDA_VISIBLE_DEVICES=4 nohup python -u train_ditto.py \
  --task oag/$task \
  --batch_size $bs \
  --max_len $len \
  --lr $lr \
  --n_epochs 40 \
  --lm $glm_roberta \
  --fp16 \
  --da del \
  --dk product \
  --summarize \
  >log/train-$task-$model_name-bs$bs-lr$lr-lasttoken.log 2>&1 &