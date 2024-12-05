import os

class config:
    root = os.getcwd()
    dataset = 'chinese ner'
    train_data_path = os.path.join(root, 'input/train.json')
    dev_data_path = os.path.join(root, 'input/dev.json')
    test_data_path = os.path.join(root, 'input/test.json')

    cache_path = os.path.join(root, 'cache/')

    save_path = os.path.join(root, 'saved_models/model.pt')
    predict_path = os.path.join(root, 'output/predict.json')

    dist_emb_size = 20
    type_emb_size = 20
    lstm_hid_size = 512
    conv_hid_size = 96
    bert_hid_size = 768
    biaffine_size = 512
    ffnn_hid_size = 288

    dilation = [1, 2, 3]

    emb_dropout = 0.5
    conv_dropout = 0.5
    out_dropout = 0.33

    epochs = 10
    batch_size = 4
    checkout_params = {'batch_size': 1, 'shuffle': False}
    train_params = {'batch_size': 1, 'shuffle': True}
    dev_params = {'batch_size': 1, 'shuffle': False}
    test_params = {'batch_size': 1, 'shuffle': False}

    learning_rate = 1e-3
    weight_decay = 0
    clip_grad_norm = 5.0
    bert_name = 'bert-base-uncased'
    bert_learning_rate = 5e-6
    warm_factor = 0.1

    use_bert_last_4_layers = True

    seed = 2022
    logger = None