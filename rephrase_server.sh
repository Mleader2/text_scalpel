# 文本复述(rephrase)的服务
#　成都
# pyenv activate python373tf115
# pip install -i https://pypi.douban.com/simple/ bert-tensorflow==1.0.1
#pip install -i https://pypi.douban.com/simple/ tensorflow==1.15.0
#python -m pip install --upgrade pip -i https://pypi.douban.com/simple


# 房山
# pyenv activate python363tf111
start_tm=`date +%s%N`;

export HOST_NAME="cloudminds"
### Optional parameters ###

# If you train multiple models on the same data, change this label.
EXPERIMENT=wikisplit_experiment # TODO  rephrase
# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).

export TRAIN_BATCH_SIZE=256  # 512 OOM   256 OK
PHRASE_VOCAB_SIZE=500
MAX_INPUT_EXAMPLES=1000000
SAVE_CHECKPOINT_STEPS=200
export enable_swap_tag=false
export output_arbitrary_targets_for_infeasible_examples=false
export WIKISPLIT_DIR="/home/${HOST_NAME}/Mywork/corpus/rephrase_corpus"
export OUTPUT_DIR="${WIKISPLIT_DIR}/output"

#python phrase_vocabulary_optimization.py \
#  --input_file=${WIKISPLIT_DIR}/train.txt \
#  --input_format=wikisplit \
#  --vocabulary_size=500 \
#  --max_input_examples=1000000 \
#  --enable_swap_tag=${enable_swap_tag} \
#  --output_file=${OUTPUT_DIR}/label_map.txt


export max_seq_length=40 # TODO
export BERT_BASE_DIR="/home/${HOST_NAME}/Mywork/model/RoBERTa-tiny-clue" # chinese_L-12_H-768_A-12"

#python preprocess_main.py \
#  --input_file=${WIKISPLIT_DIR}/tune.txt \
#  --input_format=wikisplit \
#  --output_tfrecord=${OUTPUT_DIR}/tune.tf_record \
#  --label_map_file=${OUTPUT_DIR}/label_map.txt \
#  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#  --max_seq_length=${max_seq_length} \
#  --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples}  # TODO true
#
#python preprocess_main.py \
#    --input_file=${WIKISPLIT_DIR}/train.txt \
#    --input_format=wikisplit \
#    --output_tfrecord=${OUTPUT_DIR}/train.tf_record \
#    --label_map_file=${OUTPUT_DIR}/label_map.txt \
#    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#    --max_seq_length=${max_seq_length} \
#    --output_arbitrary_targets_for_infeasible_examples=${output_arbitrary_targets_for_infeasible_examples} # TODO false



# Check these numbers from the "*.num_examples" files created in step 2.
export NUM_TRAIN_EXAMPLES=310922
export NUM_EVAL_EXAMPLES=5000
export CONFIG_FILE=configs/lasertagger_config.json
export EXPERIMENT=wikisplit_experiment_name


### 4. Prediction

# Export the model.
#python run_lasertagger.py \
#  --label_map_file=${OUTPUT_DIR}/label_map.txt \
#  --model_config_file=${CONFIG_FILE} \
#  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
#  --do_export=true \
#  --export_path=${OUTPUT_DIR}/models/${EXPERIMENT}/export

## Get the most recently exported model directory.
TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/export/" | \
            grep -v "temp-" | sort -r | head -1)
SAVED_MODEL_DIR=${OUTPUT_DIR}/models/${EXPERIMENT}/export/${TIMESTAMP}
label_map_file=${OUTPUT_DIR}/label_map.txt
export host="0.0.0.0"
export port=6000

# start server
python rephrase_server/rephrase_server_flask.py \
  --input_file=${WIKISPLIT_DIR}/test.txt \
  --input_format=wikisplit \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --max_seq_length=${max_seq_length} \
  --saved_model=${SAVED_MODEL_DIR} \
  --host=${host} \
  --port=${port}

end_tm=`date +%s%N`;
use_tm=`echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000 /3600}'`
echo "cost time" $use_tm "h"