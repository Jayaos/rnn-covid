import tensorflow as tf
import numpy as np
import pickle
import random
import os
import time
import argparse

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

def compute_loss(model, x, d, label, training):
    prediction = model(x, d, training)
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

def train_rnn_kfold(output_path, patient_record_path, demo_record_path, labels_path, max_epoch, batch_size, gru_units, hidden_units, embedding_dim,
              input_vocabsize, demo_vocabsize, l2_reg=0.001, learning_rate=0.00001, k=5, pretrained_embedding=None):

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
        print("AUC of {k}th fold: {auc:.6f}".format(k=i+1, auc=test_auc))
        k_fold_auc.append(test_auc)

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

def parse_arguments(parser):
    parser.add_argument("--output_path", type=str, help="the path to output results")
    parser.add_argument("--input_record", type=str, help="the path of training data: patient record")
    parser.add_argument("--input_demo", type=str, help="the path of training data: demographic information")
    parser.add_argument("--input_label", type=str, help="the path of training data: patient label")
    parser.add_argument("--max_epoch", type=int, help="the maximum number of epochs in each fold")
    parser.add_argument("--batch_size", type=int, help="training batch size")
    parser.add_argument("--gru_units", type=int, help="the number of units in the gru layer")
    parser.add_argument("--hidden_units", type=int, help="the number of hidden units in the hidden layer")
    parser.add_argument("--embedding_dim", type=int, help="dimensionality of the embedding")
    parser.add_argument("--input_vocabsize", type=int, help="the number of unique concepts in the data")
    parser.add_argument("--demo_vocabsize", type=int, help="dimensionality of demographic information vector")
    parser.add_argument("--l2_reg", type=float, default=0.001, help="L2 regularization coefficient")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="learning rate for the optimizer")
    parser.add_argument("--k", type=int, default=5, help="k-fold")
    parser.add_argument("--pretrained_embedding", type=str, default=None, help="the path of pretrained embedding")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    train_rnn_kfold(args.output_path, args.input_record, args.input_demo, args.input_label, args.max_epoch,
    args.batch_size, args.gru_units, args.hidden_units, args.embedding_dim, args.input_vocabsize, args.demo_vocabsize, args.l2_reg, 
    args.learning_rate, args.k, args.pretrained_embedding)