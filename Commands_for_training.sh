# ps ax | grep python3 | awk '{print $1}' | xargs kill

######################################################################################
# OAA
######################################################################################
# for training on Ubuntu 18.04 
python3 Train_OAA_VGG16.py --use_gpu 0 --batch_size_per_gpu 32 --valid_iteration 5000 --max_iteration 20000 --OAA_dir ./dataset/OAA/ --OAA_threshold 0.9
