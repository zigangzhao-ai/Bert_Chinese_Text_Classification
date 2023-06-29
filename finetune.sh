# pip install pip -U 
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# pip install -r requirements.txt 

##bert
python3 finetune/run_classifier.py \
        --pretrained_model_path pretrained_model/roberta/uer/cluecorpussmall_roberta_base_seq512_model.bin-250000 \
        --vocab_path models/google_zh_vocab.txt \
        --config_path models/roberta/base_config.json \
        --data_pth datasets/ \
        --train_path 0629/train.tsv \
        --dev_path 0629/dev_0530.tsv \
        --test_path 0629/dev_0530.tsv \
        --epochs_num 6 \
        --batch_size 64 \
        --seq_length 128 \
        --report_steps 20 \
        --eval_steps 200 \
        --learning_rate 3e-5 \
        --scheduler cosine \
        --output_model_path checkpoints/roberta/0629/seq_128/best_model \
        --log_dir checkpoints/roberta/0629/seq_128/log

