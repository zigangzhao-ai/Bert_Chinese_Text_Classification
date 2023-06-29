# pip install pip -U
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# pip install -r requirements.txt 

##推理                                                                     
python3 inference/run_classifier_infer.py \
--load_model_path checkpoints/roberta/0629/model_epoch2_step600_acc_0.9737.bin \
--vocab_path models/google_zh_vocab.txt \
--config_path models/bert/base_config.json \
--test_path datasets/0629/dev_0629.tsv \
--prediction_path datasets/0629/dev_0629_pred.txt \
--labels_num 2 \
--batch_size 128 \
--seq_length 128

