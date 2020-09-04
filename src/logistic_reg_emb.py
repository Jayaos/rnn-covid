import tensorflow as tf
import numpy as np
import pickle
import random
import os
import time
import argparse
from sklearn.model_selection import train_test_split

class LogisticRegression(tf.keras.Model):
    def __init__(self, config):
        super(LogisticRegression, self).__init__()
        self.embeddings = None
        self.optimizer = tf.keras.optimizers.Adam(config["learning_rate"])

        self.concatenation = tf.keras.layers.Concatenate(axis=1, name="concatenation")
        self.mlp = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name="mlp",
        kernel_regularizer=tf.keras.regularizers.L2(l2=config["l2_reg"]))

    def initParams(self, config):
        print("use randomly initialzed value...")
        self.embeddings = tf.Variable(tf.random.normal([config["input_vocabsize"], config["embedding_dim"]], 0, 0.01))
        
    def loadParams(self, pretrained_emb):
        print("use pre-trained embeddings...")
        self.embeddings = tf.Variable(pretrained_emb)

    def call(self, x, d):
        x = tf.matmul(x, self.embeddings)
        x = tf.math.l2_normalize(x)
        return self.mlp(self.concatenation([x, d]))

def calculate_auc(model, test_x, test_d, test_y, config):
    AUC = tf.keras.metrics.AUC(num_thresholds=200)
    AUC.reset_states()
    x, d, y = pad_matrix(test_x, test_d, test_y, config)
    pred = model(x, d)
    AUC.update_state(y, pred)

    return AUC.result().numpy()

def train_lreg(output_path, patient_record_path, demo_record_path, labels_path, epochs, batch_size, 
    input_vocabsize, demo_vocabsize, embedding_dim, l2_reg=0.001, learning_rate=0.001, pretrained_embedding=None,
    measure_time=False):
    
    config = locals().copy()
    
    print("build and initialize model...")
    lr_model = LogisticRegression(config)
    if pretrained_embedding != None:
        pretrained_embedding = np.load(pretrained_embedding)
        lr_model.loadParams(pretrained_embedding)
    else:
        lr_model.initParams(config)

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
                batch_cost = compute_loss(lr_model, x, d, y)
                gradients = tape.gradient(batch_cost, lr_model.trainable_variables)
                lr_model.optimizer.apply_gradients(zip(gradients, lr_model.trainable_variables))
                
            loss_record.append(batch_cost.numpy())
            progbar.add(1)
        
        print('epoch:{e}, mean loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
        current_auc = calculate_auc(lr_model, np.array(test_x), np.array(test_d), np.array(test_y), config)
        print('epoch:{e}, model auc:{l:.6f}'.format(e=epoch+1, l=current_auc))
        if current_auc > best_auc: 
            best_auc = current_auc
            best_epoch = epoch
            best_model = lr_model.get_weights()

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))

    print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))
    lr_model.set_weights(best_model)

    if measure_time:
        entire_x, entire_d, entire_y = pad_matrix(np.array(recs), np.array(demos), np.array(labels), config)
        start_time = time.time()
        lr_model(entire_x, entire_d)
        end_time = time.time()
        print("average time for prediction per patient: {}".format((end_time-start_time)/len(entire_y)))

def compute_loss(model, x, d, label):
    prediction = model(x, d)
    loss_sum = tf.negative(tf.add(tf.multiply(5, tf.multiply(label, tf.math.log(prediction))), 
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
        d[idx, int(demo[:-1])] = 1. # the last element of demos is age 
        d[idx, -1] = demo[-1]
        
    return x, d, y

def load_data(patient_record_path, demo_record_path, labels_path):
    patient_record = pickle.load(open(patient_record_path, 'rb'))
    demo_record = pickle.load(open(demo_record_path, 'rb'))
    labels = pickle.load(open(labels_path, 'rb'))
    
    return patient_record, demo_record, labels

def shuffle_data(data1, data2, data3):
    data1, data2, data3 = np.array(data1), np.array(data2), np.array(data3)
    idx = np.arange(len(data1))
    random.shuffle(idx)

    return data1[idx], data2[idx], data3[idx]

def train_lreg_kfold(output_path, patient_record_path, demo_record_path, labels_path, max_epoch, batch_size,
                input_vocabsize, demo_vocabsize, embedding_dim, l2_reg=0.001, learning_rate=0.001, k=5, pretrained_embedding=None):
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
        lr_model = LogisticRegression(config)
        if pretrained_embedding != None:
            loaded_embedding = np.load(pretrained_embedding)
            lr_model.loadParams(loaded_embedding)
        else:
            lr_model.initParams(config)
    
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
                    batch_cost = compute_loss(lr_model, x, d, y)
                    gradients = tape.gradient(batch_cost, lr_model.trainable_variables)
                    lr_model.optimizer.apply_gradients(zip(gradients, lr_model.trainable_variables))
                
                loss_record.append(batch_cost.numpy())
                progbar.add(1)
        
            print('epoch:{e}, mean loss:{l:.6f}'.format(e=epoch+1, l=np.mean(loss_record)))
            current_auc = calculate_auc(lr_model, valid_x, valid_d, valid_y, config)
            print('epoch:{e}, model auc:{l:.6f}'.format(e=epoch+1, l=current_auc))
            if current_auc > best_auc: 
                best_auc = current_auc
                best_epoch = epoch
                best_model = lr_model.get_weights()

        print('Best model: at epoch {e}, best model auc:{l:.6f}'.format(e=best_epoch, l=best_auc))

        print("calculate AUC on the best model using the test set")
        lr_model.set_weights(best_model)
        test_auc = calculate_auc(lr_model, test_x, test_d, test_y, config)
        print("test auc of {k} fold: {auc:.6f}".format(k=i, auc=test_auc))
        k_fold_auc.append(test_auc)

    print("save k-fold results...")
    np.save(os.path.join(output_path, "logistic_reg_{k}_fold_auc.npy".format(k=k)), k_fold_auc)

def parse_arguments(parser):
    parser.add_argument("--input_record", type=str, help="The path of training data: patient record")
    parser.add_argument("--input_demo", type=str, help="The path of training data: demographic information")
    parser.add_argument("--input_label", type=str, help="The path of training data: patient label")
    parser.add_argument("--output", type=str, help="The path to output results")
    parser.add_argument("--max_epoch", type=int, default=20, help="The maximum number of epochs in each fold")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--inpute_vocabsize", type=int, help="The number of unique concepts in the training data")
    parser.add_argument("--demo_vocabsize", type=int, help="The dimension of demographic vector")
    parser.add_argument("--embedding_dim", type=int, help="The dimension of embedding layer")
    parser.add_argument("--l2_reg", type=float, default=0.01, help="L2 regularization coefficient")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for Adam optimizer")
    parser.add_argument("--k", type=int, default=5, help="k-fold")
    parser.add_argument("--pretrained_embedding", type=str, default=None, help="The path of pretrained-embedding")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    train_lreg_kfold(args.output, args.input_record, args.input_demo, args.input_label, args.max_epoch,
    args.batch_size, args.input_vocabsize, args.demo_vocabsize, args.embedding_dim, args.l2_reg, 
    args.learning_rate, args.k, args.pretrained_embedding)