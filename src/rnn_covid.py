import tensorflow as tf
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split

class RNN(tf.keras.Model):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.embeddings = None
        self.optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        
        self.visit_activation = tf.keras.layers.Activation("tanh", name="visit_activation")
        self.masking_layer = tf.keras.layers.Masking(mask_value=0.0, name="masking_layer")
        self.gru = tf.keras.layers.GRU(units=config["gru_units"], return_sequences=True, return_state=True, name="gru")
        self.concatenation = tf.keras.layers.Concatenate(axis=1, name="concatenation")
        self.mlp1 = tf.keras.layers.Dense(config["mlp_units"], activation=tf.keras.activations.tanh, name="mlp1")
        self.mlp2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name="mlp2")
        
    def initParams(self, config):
        print("use randomly initialzed value...")
        self.embeddings = tf.Variable(tf.random.uniform([config["input_vocabsize"], config["embedding_dim"]], 0.1, -0.1))
        
    def loadParams(self, pretrained_emb):
        print("use pre-trained embeddings...")
        self.embeddings = tf.Variable(pretrained_emb)
        
    def visit_fn(self, x):
        return self.visit_activation(tf.matmul(x, self.embeddings))
            
    def gru_fn(self, x):
        visit_rep = self.visit_fn(x)
        masked_x = self.masking_layer(visit_rep)
        sequences, h = self.gru(masked_x)
        return h # n_patients x emb_dim
    
    def predict_fn(self, h, a, d):
        """
        --h: final output vector from gru
        --a: admission information
        --d: demographic information
        """
        admission_rep = self.visit_fn(a)
        c = self.mlp1(self.concatenation([h, admission_rep]))
        prediction = self.mlp2(self.concatenation([c, d]))
        return prediction
    
    def call(self, x, a, d):
        h = self.gru_fn(x)
        return self.predict_fn(h, a, d)

@tf.function
def compute_loss(model, x, a, d, label):
    prediction = model(x, a, d)
    loss_sum = tf.negative(tf.add(tf.multiply(label, tf.math.log(prediction)), 
                                  tf.multiply(tf.subtract(1., label), tf.math.log(tf.subtract(1., prediction)))))
    return tf.reduce_mean(loss_sum)

def calculate_auc(model, test_x, test_d, test_y, config):

    x, a, d, y = pad_matrix(test_x, test_d, test_y, config)
    
    print("calculate AUC...")
    return model(x, a, d)

def restore_rnn(input_path, patient_record_path, demo_record_path, labels_path, epochs, batch_size, gru_units, mlp_units, 
              input_vocabsize, demo_vocabsize, learning_rate=0.001, embedding_dim=256, pretrained_embedding=None):
    return None

def train_rnn(output_path, patient_record_path, demo_record_path, labels_path, epochs, batch_size, gru_units, mlp_units, 
              input_vocabsize, demo_vocabsize, learning_rate=0.001, embedding_dim=256, pretrained_embedding=None):
    
    config = locals().copy()
    
    print("build and initialize model...")
    rnn_model = RNN(config)
    if pretrained_embedding != None:
        rnn_model.loadParams(pretrained_embedding)
    else:
        rnn_model.initParams(config)
    
    print("load data...")
    recs, demos, labels = load_data(patient_record_path, demo_record_path, labels_path)
    train_x, test_x, train_d, test_d, train_y, test_y = train_test_split(recs, demos, labels, test_size=0.2)
    # need to save splitted dataset
    num_batches = int(np.ceil(float(len(train_x)) / float(batch_size)))
    
    best_auc = 0
    print("start training...")
    for epoch in range(epochs):
        loss_record = []
        progbar = tf.keras.utils.Progbar(num_batches)
        
        for i in random.sample(range(num_batches), num_batches):
            batch_x = train_x[i * batch_size:(i+1) * batch_size]
            batch_d = train_d[i * batch_size:(i+1) * batch_size]
            batch_y = train_y[i * batch_size:(i+1) * batch_size]
            
            x, a, d, y = pad_matrix(batch_x, batch_d, batch_y, config)
            
            with tf.GradientTape() as tape:
                batch_cost = compute_loss(rnn_model, x, a, d, y)
                gradients = tape.gradient(batch_cost, rnn_model.trainable_variables)
                rnn_model.optimizer.apply_gradients(zip(gradients, rnn_model.trainable_variables))
                
            loss_record.append(batch_cost.numpy())
            progbar.add(1)
        
        print('epoch:{e}, mean loss:{l:.6f}'.format(e=epoch, l=np.mean(loss_record)))
    
    print(calculate_auc(rnn_model, test_x, test_d, test_y, config))
        #current_auc = calculate_auc()
        # print('epoch:{e}, model auc:{l:.6f}'.format(e=epoch, l=current_auc))
        #if current_auc > best_auc: 
        #    best_auc = current_auc
        #    best_epoch = epoch
        #    best_model = rnn_model.weights()