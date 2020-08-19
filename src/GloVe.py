import numpy as np
import tensorflow as tf
import random
import os
import pickle

class GloVe(tf.keras.Model):
    def __init__(self, config):
        super(GloVe, self).__init__()
        self.final_embeddings = None
        self.vocab_size = config["vocab_size"]
        self.max_vocab_size = config["max_vocab_size"]
        self.embedding_dim = config["embedding_dim"]
        self.scaling_factor = config["scaling_factor"]
        self.optimizer = tf.keras.optimizers.Adagrad(config["learning_rate"])

    def initParams(self):
        with tf.device("/cpu:0"):
            """must be implemented with cpu-only env since this is sparse updating"""
            self.target_embeddings = tf.Variable(tf.random.normal([self.vocab_size, 
            self.embedding_dim], 0, 0.01),
                                                 name="target_embeddings")
            self.context_embeddings = tf.Variable(tf.random.normal([self.vocab_size, 
            self.embedding_dim], 0, 0.01),
                                                  name="context_embeddings")
            self.target_biases = tf.Variable(tf.random.normal([self.vocab_size], 0, 0.01),
                                             name='target_biases')
            self.context_biases = tf.Variable(tf.random.normal([self.vocab_size], 0, 0.01),
                                              name="context_biases")
    def sumEmbeddings(self):
        self.final_embeddings = self.target_embeddings + self.context_embeddings

def train_glove(output_path, patient_record_path, concept2id_path, epochs, batch_size, max_vocab_size=100, scaling_factor=0.75, learning_rate=0.001, embedding_dim=256, use_gpu=True):

    config = locals().copy()

    print("load data...")
    recs = load_data(patient_record_path)
    concept2id = load_data(concept2id_path)
    config["vocab_size"] = len(concept2id)

    print("build and initialize model...")
    glove = GloVe(config)
    glove.initParams()

    print("build co-occurrence matrix...")
    comatrix = build_comatrix(recs, concept2id)

    print("prepare training set...")
    i_ids, j_ids, co_occurrence = prepare_trainingset(comatrix)
    num_batches = int(np.ceil(len(i_ids) / batch_size))

    print("start training...")
    for epoch in range(epochs):
        cost_record = []
        progbar = tf.keras.utils.Progbar(num_batches)

        for i in random.sample(range(num_batches), num_batches): # shuffling the data 
            i_batch = i_ids[i * batch_size : (i+1) * batch_size]
            j_batch = j_ids[i * batch_size : (i+1) * batch_size]
            co_occurrence_batch = co_occurrence[i * batch_size : (i+1) * batch_size]

            with tf.GradientTape() as tape:
                batch_cost = compute_cost(glove, i_batch, j_batch, co_occurrence_batch, use_gpu)
                gradients = tape.gradient(batch_cost, glove.trainable_variables)
                glove.optimizer.apply_gradients(zip(gradients, glove.trainable_variables))

            cost_record.append(batch_cost.numpy())
            progbar.add(1)

        print('epoch:{e}, mean loss:{l:.6f}'.format(e=epoch+1, l=np.mean(cost_record)))
    
    glove.sumEmbeddings()

    print("save trained embeddings...")
    np.save(os.path.join(output_path, 
    "glove_embeddings_e{}_loss{:.4f}.npy".format(epochs, np.mean(cost_record))), glove.final_embeddings) 

def compute_cost(model, i_ids, j_ids, co_occurrence, use_gpu=True):
    if use_gpu:
        device_setting = "/gpu:0"
    else:
        device_setting = "/cpu:0"

    with tf.device(device_setting):
        """x = [target_ind, context_ind, co_occurrence_count]"""
        target_emb = tf.nn.embedding_lookup([model.target_embeddings], i_ids)
        context_emb = tf.nn.embedding_lookup([model.context_embeddings], j_ids)
        target_bias = tf.nn.embedding_lookup([model.target_biases], i_ids)
        context_bias = tf.nn.embedding_lookup([model.context_biases], j_ids)

        weight = tf.math.minimum(1.0, tf.math.pow(tf.math.truediv(co_occurrence, tf.cast(model.max_vocab_size, dtype=tf.float32)),
        model.scaling_factor))
        
        emb_product = tf.math.reduce_sum(tf.math.multiply(target_emb, context_emb), axis=1)
        log_cooccurrence = tf.math.log(tf.add(tf.cast(co_occurrence, dtype=tf.float32), 1))
        
        distance_cost = tf.math.square(
            tf.math.add_n([emb_product, target_bias, context_bias, tf.math.negative(log_cooccurrence)]))
          
    return tf.math.reduce_sum(tf.multiply(weight, distance_cost))

def build_comatrix(records, concept2id):
    comatrix = np.zeros((len(concept2id), len(concept2id)), dtype=np.float32)
    
    percent_count = 0
    for idx, patient in enumerate(records):
        if idx % np.ceil(len(records)/20) == 0:
            print("{} percent done".format(percent_count * 5))
            percent_count += 1
        for record in patient:
            for p in record:
                for k in record:
                    if p != k:
                        comatrix[p, k] += 1.
    return comatrix

def prepare_trainingset(comatrix):
    i_ids = []
    j_ids = []
    co_occurs = []

    for i in range(comatrix.shape[0]):
        for j in range(comatrix.shape[0]):
            i_ids.append(i)
            j_ids.append(j)
            co_occurs.append(comatrix[i, j])
     
    assert len(i_ids) == len(j_ids), "The length of the data are not the same"
    assert len(i_ids) == len(co_occurs), "The length of the data are not the same"
    return i_ids, j_ids, co_occurs

def load_data(data_path):
    my_data = pickle.load(open(data_path, 'rb'))

    return my_data