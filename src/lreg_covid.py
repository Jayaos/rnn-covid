import tensorflow as tf
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split

class LogisticRegression(tf.keras.Model):
    def __init__(self, config):
        super(LogisticRegression, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(config["learning_rate"])

        self.concatenation = tf.keras.layers.Concatenate(axis=1, name="concatenation")
        self.mlp = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name="mlp")
        
    def call(self, x, d):
        return self.mlp(self.concatenation([x, d]))

def calculate_auc(model, test_x, test_d, test_y, config):
    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    x, d, y = pad_matrix(test_x, test_d, test_y, config)
    pred = model(x, d)
    AUC.update_state(y, pred)

    return AUC.result().numpy()

def train_lreg(output_path, patient_record_path, demo_record_path, labels_path, epochs, batch_size, input_vocabsize, demo_vocabsize, learning_rate=0.001):
    
    config = locals().copy()
    
    print("build model...")
    lr_model = LogisticRegression(config)

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
            batch_x = train_x[i * batch_size:(i+1) * batch_size]
            batch_d = train_d[i * batch_size:(i+1) * batch_size]
            batch_y = train_y[i * batch_size:(i+1) * batch_size]
            
            x, d, y = pad_matrix(batch_x, batch_d, batch_y, config)
            
            with tf.GradientTape() as tape:
                batch_cost = compute_loss(lr_model, x, d, y)
                gradients = tape.gradient(batch_cost, lr_model.trainable_variables)
                lr_model.optimizer.apply_gradients(zip(gradients, lr_model.trainable_variables))
                
            loss_record.append(batch_cost.numpy())
            progbar.add(1)
        
        print('epoch:{e}, mean loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
        current_auc = calculate_auc(lr_model, test_x, test_d, test_y, config)
        print('epoch:{e}, model auc:{l:.6f}'.format(e=epoch+1, l=current_auc))
        if current_auc > best_auc: 
            best_auc = current_auc
            best_epoch = epoch
            best_model = lr_model.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    return lr_model # return trained model

@tf.function
def compute_loss(model, x, d, label):
    prediction = model(x, d)
    loss_sum = tf.negative(tf.add(tf.multiply(label, tf.math.log(prediction)), 
                                  tf.multiply(tf.subtract(1., label), tf.math.log(tf.subtract(1., prediction)))))
    return tf.reduce_mean(loss_sum)

def pad_matrix(records, demos, labels, config):
    n_patients = len(records)
    lengths = np.array([len(rec) for rec in records]) - 1
    input_vocabsize = config["input_vocabsize"]
    demo_vocabsize = config["demo_vocabsize"]

    x = np.zeros((n_patients, input_vocabsize)).astype(np.float32)
    d = np.zeros((n_patients, demo_vocabsize)).astype(np.float32)
    y = np.array(labels).astype(np.float32)
    
    for idx, rec in enumerate(records):
        for visit in rec:
            x[idx, visit] += 1
        
    for idx, demo in enumerate(demos):
        d[idx, demo[:-1]] = 1. # the last element of demos is age 
        d[idx, -1] = demo[-1]
        
    return x, d, y

def load_data(patient_record_path, demo_record_path, labels_path):
    patient_record = pickle.load(open(patient_record_path, 'rb'))
    demo_record = pickle.load(open(demo_record_path, 'rb'))
    labels = pickle.load(open(labels_path, 'rb'))
    
    return patient_record, demo_record, labels