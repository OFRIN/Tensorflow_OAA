import tensorflow as tf

def flags_to_dict(flags):
    return {k : flags[k].value for k in flags}

def get_config():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    ###############################################################################
    # Default Config
    ###############################################################################
    flags.DEFINE_string('root_dir', './flower_dataset/', 'unknown')

    flags.DEFINE_string('experimenter', 'JSH', 'unknown')
    flags.DEFINE_string('use_gpu', '0', 'unknown')
    
    ###############################################################################
    # Training Schedule
    ###############################################################################
    flags.DEFINE_float('init_learning_rate', 0.016, 'unknown')
    flags.DEFINE_float('alpha_learning_rate', 0.002, 'unknown')
    
    flags.DEFINE_integer('batch_size', 5, 'unknown')
    flags.DEFINE_integer('batch_size_per_gpu', 5, 'unknown')
    
    # ex. flower dataset images : 2900, 1 epoch = 580 iteration, 15 epoch = 8700 iteration
    flags.DEFINE_integer('log_iteration', 100, 'unknown')
    flags.DEFINE_integer('warmup_iteration', 500, 'unknown')
    flags.DEFINE_integer('valid_iteration', 2000, 'unknown')
    flags.DEFINE_integer('max_iteration', 10000, 'unknown')
    
    ###############################################################################
    # Training Technology
    ###############################################################################
    flags.DEFINE_string('OAA_pretrained_model', None, 'unknown')
    flags.DEFINE_string('OAA_dir', './dataset/OAA_results/', 'unknown')
    flags.DEFINE_float('OAA_threshold', 0.1, 'unknown')
    flags.DEFINE_integer('OAA_update_iteration', -1, 'unknown, -1 : auto 1 epoch')
    
    flags.DEFINE_integer('image_size', 448, 'unknown')
    flags.DEFINE_float('weight_decay', 0.0005, 'unknown')
    
    return FLAGS

if __name__ == '__main__':
    import json
    
    flags = get_config()

    print(flags.use_gpu)
    print(flags_to_dict(flags))
    
    # print(flags.mixup)
    # print(flags.efficientnet_option)