#!/bin/sh
ARCH=deeper_iwslt_de_en_v4
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined

SEED=123
[ $# -gt 0 ] && SEED=$1
echo "seed=$SEED"

OUTPUT_PATH=checkpoints/IWSLT/deeper4_seed$SEED
RESULT_PATH=results/IWSLT/deeper4_seed$SEED

mkdir -p $OUTPUT_PATH
mkdir -p $RESULT_PATH

CUDA_VISIBLE_DEVICES=3 python3 train.py $DATA_PATH \
  --seed $SEED \
  -s de -t en \
  -a $ARCH --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 \
  --lr 0.0015 --min-lr '1e-09' \
  --clip-norm 0.0 --dropout 0.3 \
  --lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --warmup-updates 8000 \
  --max-tokens 4096 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-update 50000 \
  --save-dir $OUTPUT_PATH \
  --ddp-backend=no_c10d --fp16 \
  --no-progress-bar --log-interval 100 \
  | tee -a $OUTPUT_PATH/train_log.txt
  
python3 scripts/average_checkpoints.py --inputs $OUTPUT_PATH \
  --num-epoch-checkpoints 10 --output $OUTPUT_PATH/avg_10.pt
   
python3 generate.py $DATA_PATH \
  --path $OUTPUT_PATH/avg_10.pt \
  --log-format simple \
  --batch-size 128 --beam 5 --remove-bpe --lenpen 1.0 \
  > $RESULT_PATH/avg_10.txt
  
python3 generate.py $DATA_PATH \
  --path $OUTPUT_PATH/checkpoint_best.pt \
  --log-format simple \
  --batch-size 128 --beam 5 --remove-bpe --lenpen 1.0 \
  > $RESULT_PATH/checkpoint_best.txt
