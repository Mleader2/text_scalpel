# 扩充文本匹配的语料  文本复述任务
export HOST_NAME=$1
# set gpu id to use
if [[ "wzk" == "$HOST_NAME" ]]
then
  # set gpu id to use
  export CUDA_VISIBLE_DEVICES=0
else
  # not use gpu
  export CUDA_VISIBLE_DEVICES=""
fi

start_tm=`date +%s%N`;


### Optional parameters ###
# If you train multiple models on the same data, change this label.
export EXPERIMENT="scapal_model" #"wikisplit_experiment_name_BertTiny" #　

# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).
export num_train_epochs=30
export TRAIN_BATCH_SIZE=256
export PHRASE_VOCAB_SIZE=500
export MAX_INPUT_EXAMPLES=1000000
export SAVE_CHECKPOINT_STEPS=2000
export keep_checkpoint_max=8
export enable_swap_tag=false
export output_arbitrary_targets_for_infeasible_examples=false
export REPHRASE_DIR="/home/${HOST_NAME}/Mywork/corpus/rephrase_corpus"
export BERT_BASE_DIR="/home/${HOST_NAME}/Mywork/model/RoBERTa-tiny-clue"
export OUTPUT_DIR="${REPHRASE_DIR}/output"

export max_seq_length=40
# Check these numbers from the "*.num_examples" files created in step 2.
export NUM_TRAIN_EXAMPLES=310922
export NUM_EVAL_EXAMPLES=5000
export CONFIG_FILE=configs/lasertagger_config.json

### Get the most recently exported model directory.
TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/export/" | \
            grep -v "temp-" | sort -r | head -1)
SAVED_MODEL_DIR=${OUTPUT_DIR}/models/${EXPERIMENT}/export/${TIMESTAMP}
PREDICTION_FILE=${OUTPUT_DIR}/models/${EXPERIMENT}/pred.tsv


#python phrase_vocabulary_optimization.py \
#  --input_file=${REPHRASE_DIR}/train.txt \
#  --input_format=wikisplit \
#  --vocabulary_size=${PHRASE_VOCAB_SIZE} \
#  --max_input_examples=1000000 \
#  --enable_swap_tag=${enable_swap_tag} \
#  --output_file=${OUTPUT_DIR}/label_map.txt  \
#  --vocab_file=${BERT_BASE_DIR}/vocab.txt
#
#python preprocess_main.py \
#  --input_file=${REPHRASE_DIR}/tune.txt \
#  --input_format=wikisplit \
#  --output_tfrecord=${OUTPUT_DIR}/tune.tf_record \
#  --label_map_file=${OUTPUT_DIR}/label_map.txt \
#  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#  --max_seq_length=${max_seq_length} \
#  --enable_swap_tag=${enable_swap_tag} \
#  --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples}
#python preprocess_main.py \
#    --input_file=${REPHRASE_DIR}/train.txt \
#    --input_format=wikisplit \
#    --output_tfrecord=${OUTPUT_DIR}/train.tf_record \
#    --label_map_file=${OUTPUT_DIR}/label_map.txt \
#    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#    --max_seq_length=${max_seq_length} \
#    --enable_swap_tag=${enable_swap_tag} \
#    --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples}


#python run_lasertagger.py \
#  --training_file=${OUTPUT_DIR}/train.tf_record \
#  --eval_file=${OUTPUT_DIR}/tune.tf_record \
#  --label_map_file=${OUTPUT_DIR}/label_map.txt \
#  --model_config_file=${CONFIG_FILE} \
#  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
#  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
#  --do_train=true \
#  --do_eval=true \
#  --num_train_epochs=${num_train_epochs} \
#  --train_batch_size=${TRAIN_BATCH_SIZE} \
#  --save_checkpoints_steps=${SAVE_CHECKPOINT_STEPS} \
#  --keep_checkpoint_max=${keep_checkpoint_max} \
#  --max_seq_length=${max_seq_length} \
#  --num_train_examples=${NUM_TRAIN_EXAMPLES} \
#  --num_eval_examples=${NUM_EVAL_EXAMPLES}


## Export the model.
echo "Export the model."
python run_lasertagger.py \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --model_config_file=${CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
  --do_export=true \
  --export_path=${OUTPUT_DIR}/models/${EXPERIMENT}/export
#
#
#
### 4. Prediction
#echo "predict"
python predict_main.py \
  --input_file=${REPHRASE_DIR}/test.txt \
  --input_format=wikisplit \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --max_seq_length=${max_seq_length} \
  --enable_swap_tag=${enable_swap_tag} \
  --saved_model=${SAVED_MODEL_DIR}

### 5. Evaluation
python score_main.py --prediction_file=${PREDICTION_FILE}


end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}'`
echo "cost time" $use_tm "h"