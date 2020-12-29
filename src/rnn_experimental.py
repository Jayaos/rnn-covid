import tensorflow as tf
import numpy as np
import pickle
import random
import os
import time

class RNN(tf.keras.Model):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.embeddings = None
        self.optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        self.masking_layer = tf.keras.layers.Masking(mask_value=0.0, name="masking_layer")
        self.gru = tf.keras.layers.GRU(units=config["gru_units"], name="gru")
        self.concatenation = tf.keras.layers.Concatenate(axis=1, name="concatenation")
        self.mlp1 = tf.keras.layers.Dense(config["hidden_units"], activation=tf.keras.activations.tanh, name="mlp1", 
        kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))
        self.mlp2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name="mlp2", 
        kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))
        
    def initParams(self, config):
        print("use randomly initialzed value...")
        self.embeddings = tf.Variable(tf.random.normal([config["input_vocabsize"], config["embedding_dim"]], 0, 0.01))   
   
    def loadParams(self, pretrained_emb):
        print("use pre-trained embeddings...")
        self.embeddings = tf.Variable(pretrained_emb)
    
    def call(self, x, d, training):
        x = tf.matmul(x, self.embeddings)
        x = self.masking_layer(x)
        x = self.gru(x)
        x = self.mlp1(self.concatenation([x, d]))
        return self.mlp2(x)
    
    def generate_patientvec(self, x, d):
        x = tf.matmul(x, self.embeddings)
        x = self.masking_layer(x)
        x = self.gru(x)
        return self.mlp1(self.concatenation([x, d]))

def compute_loss(model, x, d, label, training):
    prediction = model(x, d, training)
    loss_sum = tf.negative(tf.add(tf.multiply(5, tf.multiply(label, tf.math.log(prediction))), 
                                  tf.multiply(tf.subtract(1., label), tf.math.log(tf.subtract(1., prediction)))))
    return tf.reduce_mean(loss_sum)

def validate_loss(model, validate_x, validate_d, validate_y, config):

    batch_size = config["batch_size"]
    num_batches = int(np.ceil(float(len(validate_x)) / float(batch_size)))
    validate_loss_record = []

    for i in range(num_batches):
        batch_x = np.array(validate_x[i * batch_size:(i+1) * batch_size])
        batch_d = np.array(validate_d[i * batch_size:(i+1) * batch_size])
        batch_y = np.array(validate_y[i * batch_size:(i+1) * batch_size])

        x, d, y = pad_matrix(batch_x, batch_d, batch_y, config)
        loss = compute_loss(model, x, d, y, training=False)
        validate_loss_record.append(loss.numpy())

    return np.mean(validate_loss_record)

def calculate_auc(model, test_x, test_d, test_y, config):
    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    AUC.reset_states()
    x, d, y = pad_matrix(test_x, test_d, test_y, config)
    pred = model(x, d, training=False)
    AUC.update_state(y, pred)

    return AUC.result().numpy()

def load_data(patient_record_path, demo_record_path, labels_path):
    patient_record = pickle.load(open(patient_record_path, 'rb'))
    demo_record = pickle.load(open(demo_record_path, 'rb'))
    labels = pickle.load(open(labels_path, 'rb'))
    
    return patient_record, demo_record, labels

def save_data(output_path, mydata):
    with open(output_path, 'wb') as f:
        pickle.dump(mydata, f)

def pad_matrix(records, demos, labels, config):
    n_patients = len(records)
    lengths = np.array([len(rec) for rec in records])
    max_len = np.max(lengths)
    input_vocabsize = config["input_vocabsize"]
    demo_vocabsize = config["demo_vocabsize"]

    x = np.zeros((n_patients, max_len, input_vocabsize)).astype(np.float32)
    d = np.zeros((n_patients, demo_vocabsize)).astype(np.float32)
    y = np.array(labels).astype(np.float32)
    
    for idx, rec in enumerate(records):
        for xvec, visit in zip(x[idx,: ,: ], rec[:]):
            xvec[visit] = 1.
        
    for idx, demo in enumerate(demos):
        d[idx, int(demo[:-1])] = 1. # the last element of demos is age 
        d[idx, -1] = demo[-1]
        
    return x, d, y

def shuffle_data(data1, data2, data3):
    data1, data2, data3 = np.array(data1), np.array(data2), np.array(data3)
    idx = np.arange(len(data1))
    random.shuffle(idx)

    return data1[idx], data2[idx], data3[idx]

def load_pkl_data(data_path):
    mydata = pickle.load(open(data_path, 'rb'))
    
    return mydata

def get_prediction_score(model_path, data_path, config_path, measure_time):
    """"get prediction score for the given data using the saved model"""
    
    # load the saved model
    print("loading the saved model...")
    trained_model = restore_model(model_path, config_path)
        
    # load the test data
    print("loading the test data...")
    test_x, test_d, test_y = load_pkl_data(data_path)
    config = load_pkl_data(config_path)
    x, d, _ = pad_matrix(test_x, test_d, test_y, config)
    
    # calculate prediction scores
    print("calculating prediction scores...")
    start_time = time.time()
    scores = trained_model(x, d, training=False)
    end_time = time.time()

    if measure_time:
        print("{t} seconds to make predictions".format(t=end_time-start_time))
        print("{t} seconds to make prediction per patient".format(t=(end_time-start_time) / len(test_d)))

    return scores.numpy()

def train_rnn_kfold(output_path, patient_record_path, demo_record_path, labels_path, max_epoch, batch_size, gru_units, hidden_units, embedding_dim,
              input_vocabsize, demo_vocabsize, l2_reg=0.001, learning_rate=0.001, k=5, pretrained_embedding=None):

    config = locals().copy()

    print("load data...")
    recs, demos, labels = load_data(patient_record_path, demo_record_path, labels_path)

    print("split the dataset into k-fold...")
    recs, demos, labels = shuffle_data(recs, demos, labels)
    chunk_size = int(np.floor(len(labels) / k))
    np.split(np.arange(len(labels)), [chunk_size*i for i in range(k)])
    folds = np.tile(np.split(np.arange(len(labels)), [chunk_size*i for i in range(int(k))])[1:], 2)

    k_fold_auc = []
    k_fold_training_loss = []
    k_fold_validation_loss = []
    time_elapsed = []

    for i in range(k):
        start_time = time.time()
        train_x, valid_x, test_x = recs[np.concatenate(folds[(i%k):(i%k)+k-2])], recs[folds[(i%k)+k-1]], recs[folds[(i%k)+k]]
        train_d, valid_d, test_d = demos[np.concatenate(folds[(i%k):(i%k)+k-2])], demos[folds[(i%k)+k-1]], demos[folds[(i%k)+k]]
        train_y, valid_y, test_y = labels[np.concatenate(folds[(i%k):(i%k)+k-2])], labels[folds[(i%k)+k-1]], labels[folds[(i%k)+k]]

        print("saving test data for {f} fold...".format(f=i))
        save_data(os.path.join(output_path, "RNN_{f}fold_testset.pkl".format(f=i+1)), [test_x, test_d, test_y])

        num_batches = int(np.ceil(float(len(train_x)) / float(batch_size)))
        training_loss = []
        validation_loss = []

        print("build and initialize model for {k}th fold...".format(k=i+1))
        rnn_model = RNN(config)
        if pretrained_embedding != None:
            loaded_embedding = np.load(pretrained_embedding)
            rnn_model.loadParams(loaded_embedding)
        else:
            rnn_model.initParams(config)
    
        best_loss = np.inf
        best_epoch = 0
        best_model = None
        print("start training...")
        for epoch in range(max_epoch):
            loss_record = []
            progbar = tf.keras.utils.Progbar(num_batches)
        
            for t in random.sample(range(num_batches), num_batches):
                batch_x = train_x[t * batch_size:(t+1) * batch_size]
                batch_d = train_d[t * batch_size:(t+1) * batch_size]
                batch_y = train_y[t * batch_size:(t+1) * batch_size]
            
                x, d, y = pad_matrix(batch_x, batch_d, batch_y, config)
            
                with tf.GradientTape() as tape:
                    batch_cost = compute_loss(rnn_model, x, d, y, training=True)
                gradients = tape.gradient(batch_cost, rnn_model.trainable_variables)
                rnn_model.optimizer.apply_gradients(zip(gradients, rnn_model.trainable_variables))
                
                loss_record.append(batch_cost.numpy())
                progbar.add(1)
        
            print('epoch:{e}, training loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
            training_loss.append(np.mean(loss_record))
            current_loss = validate_loss(rnn_model, np.array(valid_x), np.array(valid_d), np.array(valid_y), config)
            print('epoch:{e}, validation loss:{l:.6f}'.format(e=epoch+1, l=current_loss))
            validation_loss.append(current_loss)
            if current_loss < best_loss: 
                best_loss = current_loss
                best_epoch = epoch+1
                best_model = rnn_model.get_weights()

        end_time = time.time()
        time_elapsed.append(end_time - start_time)
        print("{t:.6f} seconds for training {k}th fold".format(t=end_time-start_time, k=i+1))

        print('Best model of {k}th fold: at epoch {e}, best model validation loss:{l:.6f}'.format(k=i+1, e=best_epoch, l=best_loss))
        k_fold_training_loss.append(training_loss)
        k_fold_validation_loss.append(validation_loss)

        print("calculate AUC using the best model on the test set")
        rnn_model.set_weights(best_model)
        test_auc = calculate_auc(rnn_model, test_x, test_d, test_y, config)
        print("test auc of {k}th fold: {auc:.6f}".format(k=i+1, auc=test_auc))
        k_fold_auc.append(test_auc)

        print("saving the best model in the fold...")
        np.save(os.path.join(output_path, "RNN_fold{f}_best_model.npy".format(f=i+1)), best_model)
        np.save(os.path.join(output_path, "RNN_fold{f}_training_loss.npy".format(f=i+1)), training_loss)
        np.save(os.path.join(output_path, "RNN_fold{f}_validation_loss.npy".format(f=i+1)), validation_loss)

    print("saving k-fold results...")
    if pretrained_embedding != None:
        mode_name = "preemb"
    else:
        mode_name = "emb"
    np.save(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_training_loss.npy".format(k=k, m=mode_name, l=learning_rate)), k_fold_training_loss)
    np.save(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_validation_loss.npy".format(k=k, m=mode_name, l=learning_rate)), k_fold_validation_loss)
    np.save(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_auc.npy".format(k=k, m=mode_name, l=learning_rate)), k_fold_auc)
    np.save(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_time.npy".format(k=k, m=mode_name, l=learning_rate)), time_elapsed)
    save_data(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_config.pkl".format(k=k, m=mode_name, l=learning_rate)), config)

def restore_model(model_path, config_path):

    print("load data...")
    config = pickle.load(open(config_path, 'rb'))
    model_weight = np.load(model_path, allow_pickle=True)
    recs, demos, labels = load_data(config["patient_record_path"], config["demo_record_path"], config["labels_path"])
    recs, demos, labels = shuffle_data(recs, demos, labels)
    batch_size = config["batch_size"]
    
    print("restoring model...")
    batch_x = recs[1 * batch_size:(2) * batch_size]
    batch_d = demos[1 * batch_size:(2) * batch_size]
    batch_y = labels[1 * batch_size:(2) * batch_size]

    model = RNN(config)
    model.initParams(config)
    x, d, y = pad_matrix(batch_x, batch_d, batch_y, config)
    _ = compute_loss(model, x, d, y, training=False)
    model.set_weights(model_weight)

    return model

def train_rnn_entire(output_path, patient_record_path, demo_record_path, labels_path, epochs, batch_size, gru_units, hidden_units, embedding_dim,
              input_vocabsize, demo_vocabsize, l2_reg=0.001, learning_rate=0.001, pretrained_embedding=None):
    """
    train rnn model to generate patient vectors
    """

    config = locals().copy()

    print("load data...")
    recs, demos, labels = load_data(patient_record_path, demo_record_path, labels_path)

    num_batches = int(np.ceil(float(len(recs)) / float(batch_size)))
    training_loss = []

    print("build and initialize model...")
    rnn_model = RNN(config)
    if pretrained_embedding != None:
        loaded_embedding = np.load(pretrained_embedding)
        rnn_model.loadParams(loaded_embedding)
    else:
        rnn_model.initParams(config)
    
    print("start training...")
    for epoch in range(epochs):
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batches)
        
        for t in random.sample(range(num_batches), num_batches):
            batch_x = recs[t * batch_size:(t+1) * batch_size]
            batch_d = demos[t * batch_size:(t+1) * batch_size]
            batch_y = labels[t * batch_size:(t+1) * batch_size]
            
            x, d, y = pad_matrix(np.array(batch_x), np.array(batch_d), np.array(batch_y), config)
            
            with tf.GradientTape() as tape:
                batch_cost = compute_loss(rnn_model, x, d, y, training=True)
            gradients = tape.gradient(batch_cost, rnn_model.trainable_variables)
            rnn_model.optimizer.apply_gradients(zip(gradients, rnn_model.trainable_variables))
                
            loss_record.append(batch_cost.numpy())
            progbar.add(1)
        
        print('epoch:{e}, training loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
        training_loss.append(np.mean(loss_record))
    
    print("training done")

    if pretrained_embedding != None:
        mode_name = "preemb"
    else:
        mode_name = "emb"
    np.save(os.path.join(output_path, "RNN_{m}_entire_l{l}_training_loss.npy".format(m=mode_name, l=learning_rate)), training_loss)
    np.save(os.path.join(output_path, "RNN_{m}_entire_l{l}_e{e}_model.npy".format(m=mode_name, l=learning_rate, e=epochs)), rnn_model.get_weights())

def train_rnn_kfold_aucval(output_path, patient_record_path, demo_record_path, labels_path, max_epoch, batch_size, gru_units, hidden_units, embedding_dim,
              input_vocabsize, demo_vocabsize, l2_reg=0.001, learning_rate=0.001, dropout_rate=0.3, k=5, pretrained_embedding=None):

    config = locals().copy()

    print("load data...")
    recs, demos, labels = load_data(patient_record_path, demo_record_path, labels_path)

    print("split the dataset into k-fold...")
    recs, demos, labels = shuffle_data(recs, demos, labels)
    chunk_size = int(np.floor(len(labels) / k))
    np.split(np.arange(len(labels)), [chunk_size*i for i in range(k)])
    folds = np.tile(np.split(np.arange(len(labels)), [chunk_size*i for i in range(int(k))])[1:], 2)

    k_fold_auc = []
    k_fold_training_loss = []
    k_fold_validation_auc = []
    time_elapsed = []

    for i in range(k):
        start_time = time.time()
        train_x, valid_x, test_x = recs[np.concatenate(folds[(i%k):(i%k)+k-2])], recs[folds[(i%k)+k-1]], recs[folds[(i%k)+k]]
        train_d, valid_d, test_d = demos[np.concatenate(folds[(i%k):(i%k)+k-2])], demos[folds[(i%k)+k-1]], demos[folds[(i%k)+k]]
        train_y, valid_y, test_y = labels[np.concatenate(folds[(i%k):(i%k)+k-2])], labels[folds[(i%k)+k-1]], labels[folds[(i%k)+k]]

        print("saving test data for {f} fold...".format(f=i))
        save_data(os.path.join(output_path, "RNN_{f}fold_testset.pkl".format(f=i+1)), [test_x, test_d, test_y])

        num_batches = int(np.ceil(float(len(train_x)) / float(batch_size)))
        training_loss = []
        validation_auc = []

        print("build and initialize model for {k}th fold...".format(k=i+1))
        rnn_model = RNN(config)
        if pretrained_embedding != None:
            loaded_embedding = np.load(pretrained_embedding)
            rnn_model.loadParams(loaded_embedding)
        else:
            rnn_model.initParams(config)
    
        best_auc = 0
        best_epoch = 0
        best_model = None
        print("start training...")
        for epoch in range(max_epoch):
            loss_record = []
            progbar = tf.keras.utils.Progbar(num_batches)
        
            for t in random.sample(range(num_batches), num_batches):
                batch_x = train_x[t * batch_size:(t+1) * batch_size]
                batch_d = train_d[t * batch_size:(t+1) * batch_size]
                batch_y = train_y[t * batch_size:(t+1) * batch_size]
            
                x, d, y = pad_matrix(batch_x, batch_d, batch_y, config)
            
                with tf.GradientTape() as tape:
                    batch_cost = compute_loss(rnn_model, x, d, y, training=True)
                gradients = tape.gradient(batch_cost, rnn_model.trainable_variables)
                rnn_model.optimizer.apply_gradients(zip(gradients, rnn_model.trainable_variables))
                
                loss_record.append(batch_cost.numpy())
                progbar.add(1)
        
            print('epoch:{e}, training loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
            training_loss.append(np.mean(loss_record))
            current_auc = calculate_auc(rnn_model, valid_x, valid_d, valid_y, config)
            print('epoch:{e}, validation auc:{l:.6f}'.format(e=epoch+1, l=current_auc))
            validation_auc.append(current_auc)
            if current_auc > best_auc: 
                best_auc = current_auc
                best_epoch = epoch+1
                best_model = rnn_model.get_weights()

        end_time = time.time()
        time_elapsed.append(end_time - start_time)
        print("{t:.6f} seconds for training {k}th fold".format(t=end_time-start_time, k=i+1))

        print('Best model of {k}th fold: at epoch {e}, best model validation loss:{l:.6f}'.format(k=i+1, e=best_epoch, l=best_auc))
        k_fold_training_loss.append(training_loss)
        k_fold_validation_auc.append(validation_auc)

        print("calculate AUC using the best model on the test set")
        rnn_model.set_weights(best_model)
        test_auc = calculate_auc(rnn_model, test_x, test_d, test_y, config)
        print("test auc of {k}th fold: {auc:.6f}".format(k=i+1, auc=test_auc))
        k_fold_auc.append(test_auc)

        print("saving the best model in the fold...")
        np.save(os.path.join(output_path, "RNN_fold{f}_best_model.npy".format(f=i+1)), best_model)
        np.save(os.path.join(output_path, "RNN_fold{f}_training_loss.npy".format(f=i+1)), training_loss)
        np.save(os.path.join(output_path, "RNN_fold{f}_validation_auc.npy".format(f=i+1)), validation_auc)

    print("saving k-fold results...")
    if pretrained_embedding != None:
        mode_name = "preemb"
    else:
        mode_name = "emb"
    np.save(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_training_loss.npy".format(k=k, m=mode_name, l=learning_rate)), k_fold_training_loss)
    np.save(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_validation_auc.npy".format(k=k, m=mode_name, l=learning_rate)), k_fold_validation_auc)
    np.save(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_auc.npy".format(k=k, m=mode_name, l=learning_rate)), k_fold_auc)
    np.save(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_time.npy".format(k=k, m=mode_name, l=learning_rate)), time_elapsed)
    save_data(os.path.join(output_path, "RNN_{m}_{k}fold_l{l}_config.pkl".format(k=k, m=mode_name, l=learning_rate)), config)