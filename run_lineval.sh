echo "CUDA_VISIBLE_... ./run_pred.sh selective_rob 1"
python lin_eval.py --learning_rate 0.1 --schedule step --ckpt_path $1 --loading_epoch $2 --fname ep$2_lr0.1_step
python lin_eval.py --learning_rate 0.1 --rLE --schedule step --ckpt_path $1 --loading_epoch $2 --fname ep$2_lr0.1_step
