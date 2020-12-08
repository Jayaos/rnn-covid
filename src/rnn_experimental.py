import tensorflow as tf
import numpy as np
import pickle
import random
import os
import time
from sklearn.model_selection import train_test_split

class RNN(tf.keras.Model):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.embeddings = None
        self.optimizer = tf.keras.optimizers.Adam(config["learning_rate"])

        self.dropout = tf.keras.layers.Dropout(config["dropout_rate"])
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
        x = self.dropout(x, training)
        x = self.gru(x)
        x = self.mlp1(self.concatenation([x, d]))
        return self.mlp2(x)

def compute_loss(model, x, d, label):
    prediction = model(x, d, training=True)
    loss_sum = tf.negative(tf.add(tf.multiply(5, tf.multiply(label, tf.math.log(prediction))), 
                                  tf.multiply(tf.subtract(1., label), tf.math.log(tf.subtract(1., prediction)))))
    return tf.reduce_mean(loss_sum)
    
def calculate_auc(model, test_x, test_d, test_y, config):
    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    AUC.reset_states()
    x, d, y = pad_matrix(test_x, test_d, test_y, config)
    pred = model(x, d, training=False)
    AUC.update_state(y, pred)

    return AUC.result().numpy()

def get_predictions(model, test_x, test_d, test_y, config):
    x, d, y = pad_matrix(test_x, test_d, test_y, config)
    pred = model(x, d, training=False)

    return pred, y

def train_rnn(output_path, patient_record_path, demo_record_path, labels_path, epochs, batch_size, gru_units, hidden_units, 
              input_vocabsize, demo_vocabsize, l2_reg=0.01, learning_rate=0.001, embedding_dim=256, pretrained_embedding=None,
              measure_time=False):
    
    config = locals().copy()
    
    print("build and initialize model...")
    rnn_model = RNN(config)
    if pretrained_embedding != None:
        pretrained_embedding = np.load(pretrained_embedding)
        rnn_model.loadParams(pretrained_embedding)
    else:
        rnn_model.initParams(config)
    
    print("load data...")
    recs, demos, labels = load_data(patient_record_path, demo_record_path, labels_path)
    train_x, test_x, train_d, test_d, train_y, test_y = train_test_split(recs, demos, labels, test_size=0.2)
    # need to save splitted dataset
    num_batches = int(np.ceil(float(len(train_x)) / float(batch_size)))
    
    best_auc = 0
    best_epoch = 0
    best_model = None
    print("start training...")
    for epoch in range(epochs):
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batches)
        
        for i in random.sample(range(num_batches), num_batches):
            batch_x = np.array(train_x[i * batch_size:(i+1) * batch_size])
            batch_d = np.array(train_d[i * batch_size:(i+1) * batch_size])
            batch_y = np.array(train_y[i * batch_size:(i+1) * batch_size])
            
            x, d, y = pad_matrix(batch_x, batch_d, batch_y, config)
            
            with tf.GradientTape() as tape:
                batch_cost = compute_loss(rnn_model, x, d, y)
                gradients = tape.gradient(batch_cost, rnn_model.trainable_variables)
                rnn_model.optimizer.apply_gradients(zip(gradients, rnn_model.trainable_variables))
                
            loss_record.append(batch_cost.numpy())
            progbar.add(1)
        
        print('epoch:{e}, mean loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
        current_auc = calculate_auc(rnn_model, np.array(test_x), np.array(test_d), np.array(test_y), config)
        print('epoch:{e}, model auc:{l:.6f}'.format(e=epoch+1, l=current_auc))
        if current_auc > best_auc: 
            best_auc = current_auc
            best_epoch = epoch
            best_model = rnn_model.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    rnn_model.set_weights(best_model)

    if measure_time:
        entire_x, entire_d, entire_y = pad_matrix(np.array(recs), np.array(demos), np.array(labels), config)
        start_time = time.time()
        rnn_model(entire_x, entire_d)
        end_time = time.time()
        print("average time for prediction per patient: {}".format((end_time-start_time)/len(entire_y)))
    
    return get_predictions(rnn_model, np.array(test_x), np.array(test_d), np.array(test_y), config)

def load_data(patient_record_path, demo_record_path, labels_path):
    patient_record = pickle.load(open(patient_record_path, 'rb'))
    demo_record = pickle.load(open(demo_record_path, 'rb'))
    labels = pickle.load(open(labels_path, 'rb'))
    
    return patient_record, demo_record, labels

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

def train_rnn_kfold(output_path, patient_record_path, demo_record_path, labels_path, max_epoch, batch_size, gru_units, hidden_units, embedding_dim,
              input_vocabsize, demo_vocabsize, l2_reg=0.001, learning_rate=0.001, k=5, pretrained_embedding=None):
    k_fold_auc = []

    config = locals().copy()

    print("load data...")
    recs, demos, labels = load_data(patient_record_path, demo_record_path, labels_path)

    print("split the dataset into k-fold...")
    recs, demos, labels = shuffle_data(recs, demos, labels)
    chunk_size = int(np.floor(len(labels) / k))
    np.split(np.arange(len(labels)), [chunk_size*i for i in range(k)])
    folds = np.tile(np.split(np.arange(len(labels)), [chunk_size*i for i in range(int(k))])[1:], 2)

    for i in range(k):
        train_x, valid_x, test_x = recs[np.concatenate(folds[(i%k):(i%k)+k-2])], recs[folds[(i%k)+k-1]], recs[folds[(i%k)+k]]
        train_d, valid_d, test_d = demos[np.concatenate(folds[(i%k):(i%k)+k-2])], demos[folds[(i%k)+k-1]], demos[folds[(i%k)+k]]
        train_y, valid_y, test_y = labels[np.concatenate(folds[(i%k):(i%k)+k-2])], labels[folds[(i%k)+k-1]], labels[folds[(i%k)+k]]

        num_batches = int(np.ceil(float(len(train_x)) / float(batch_size)))

        print("build and initialize model...")
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
                    batch_cost = compute_loss(rnn_model, x, d, y)
                    gradients = tape.gradient(batch_cost, rnn_model.trainable_variables)
                    rnn_model.optimizer.apply_gradients(zip(gradients, rnn_model.trainable_variables))
                
                loss_record.append(batch_cost.numpy())
                progbar.add(1)
        
            print('epoch:{e}, mean loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
            current_auc = calculate_auc(rnn_model, valid_x, valid_d, valid_y, config)
            print('epoch:{e}, model auc:{l:.6f}'.format(e=epoch+1, l=current_auc))
            if current_auc > best_auc: 
                best_auc = current_auc
                best_epoch = epoch
                best_model = rnn_model.get_weights()

        print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))

        print("calculate AUC on the best model using the test set")
        rnn_model.set_weights(best_model)
        test_auc = calculate_auc(rnn_model, test_x, test_d, test_y, config)
        print("test auc of {k} fold: {auc:.6f}".format(k=i, auc=test_auc))
        k_fold_auc.append(test_auc)

    print("save k-fold results...")
    np.save(os.path.join(output_path, "{k}_fold_auc.npy".format(k=k)), k_fold_auc)

def train_rnn_kfold_return(output_path, patient_record_path, demo_record_path, labels_path, max_epoch, batch_size, gru_units, hidden_units, embedding_dim,
              input_vocabsize, demo_vocabsize, l2_reg=0.001, learning_rate=0.001, k=5, pretrained_embedding=None):

    config = locals().copy()

    print("load data...")
    recs, demos, labels = load_data(patient_record_path, demo_record_path, labels_path)

    print("split the dataset into k-fold...")
    recs, demos, labels = shuffle_data(recs, demos, labels)
    chunk_size = int(np.floor(len(labels) / k))
    np.split(np.arange(len(labels)), [chunk_size*i for i in range(k)])
    folds = np.tile(np.split(np.arange(len(labels)), [chunk_size*i for i in range(int(k))])[1:], 2)

    j = 0
    train_x, valid_x, test_x = recs[np.concatenate(folds[(j%k):(j%k)+k-2])], recs[folds[(j%k)+k-1]], recs[folds[(j%k)+k]]
    train_d, valid_d, test_d = demos[np.concatenate(folds[(j%k):(j%k)+k-2])], demos[folds[(j%k)+k-1]], demos[folds[(j%k)+k]]
    train_y, valid_y, test_y = labels[np.concatenate(folds[(j%k):(j%k)+k-2])], labels[folds[(j%k)+k-1]], labels[folds[(j%k)+k]]

    num_batches = int(np.ceil(float(len(train_x)) / float(batch_size)))

    print("build and initialize model...")
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
                batch_cost = compute_loss(rnn_model, x, d, y)
                gradients = tape.gradient(batch_cost, rnn_model.trainable_variables)
                rnn_model.optimizer.apply_gradients(zip(gradients, rnn_model.trainable_variables))
                
            loss_record.append(batch_cost.numpy())
            progbar.add(1)
        
        print('epoch:{e}, mean loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
        current_auc = calculate_auc(rnn_model, valid_x, valid_d, valid_y, config)
        print('epoch:{e}, model auc:{l:.6f}'.format(e=epoch+1, l=current_auc))
        if current_auc > best_auc: 
            best_auc = current_auc
            best_epoch = epoch
            best_model = rnn_model.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))

    print("calculate AUC on the best model using the test set")
    rnn_model.set_weights(best_model)
    test_auc = calculate_auc(rnn_model, test_x, test_d, test_y, config)
    print("test auc of a single fold: {auc:.6f}".format(auc=test_auc))

    pred, true_y = get_predictions(rnn_model, test_x, test_d, test_y, config)

    return test_x, test_d, pred, true_y

