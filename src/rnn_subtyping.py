import tensorflow as tf
import numpy as np
import pickle
import random
import os
from sklearn.model_selection import train_test_split

class RNN(tf.keras.Model):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.embeddings = None
        self.optimizer = tf.keras.optimizers.Adam(config["learning_rate"])

        self.dropout = tf.keras.layers.Dropout(0.2)
        self.masking_layer = tf.keras.layers.Masking(mask_value=0.0, name="masking_layer")
        self.gru = tf.keras.layers.GRU(units=config["gru_units"], return_sequences=True, return_state=True, name="gru")
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
        sequences, x = self.gru(x)
        if training:
            x = self.dropout(x)
        x = self.mlp1(self.concatenation([x, d]))
        return self.mlp2(x)
    
    def generateRep(self, x, d):
        x = tf.matmul(x, self.embeddings)
        x = self.masking_layer(x)
        sequences, x = self.gru(x)
        return self.mlp1(self.concatenation([x, d]))

def compute_loss(model, x, d, label):
    prediction = model(x, d, training=True)
    loss_sum = tf.negative(tf.add(tf.multiply(label, tf.math.log(prediction)), 
                                  tf.multiply(tf.subtract(1., label), tf.math.log(tf.subtract(1., prediction)))))
    return tf.reduce_mean(loss_sum)

def calculate_auc(model, test_x, test_d, test_y, config):
    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    AUC.reset_states()
    x, d, y = pad_matrix(test_x, test_d, test_y, config)
    pred = model(x, d, training=False)
    AUC.update_state(y, pred)

    return AUC.result().numpy()

def train_rnn(output_path, patient_record_path, demo_record_path, labels_path, epochs, batch_size, gru_units, hidden_units, embedding_dim,
              input_vocabsize, demo_vocabsize, l2_reg=0.01, learning_rate=0.001, pretrained_embedding=None, generate_rep=False):

    config = locals().copy()
    
    print("build and initialize model...")
    rnn_model = RNN(config)
    if pretrained_embedding != None:
        loaded_embedding = np.load(pretrained_embedding)
        rnn_model.loadParams(loaded_embedding)
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
    
    rnn_model.set_weights(best_model)
    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))

    if generate_rep:
        print("generate patient representation...")
        entire_x, entire_d, entire_y = pad_matrix(np.array(recs), np.array(demos), np.array(labels), config)
        intermediate_rep = rnn_model.generateRep(entire_x, entire_d)
        np.save(os.path.join(output_path, "patient_representation.npy"), intermediate_rep.numpy())
        print("save patient representation...")

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