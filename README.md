## Severity Prediction for COVID-19 Patients via Recurrent Neural Networks

This repository contains source codes for implementing models and reproducing the results in the following [paper](https://www.medrxiv.org/content/10.1101/2020.08.28.20184200v1):

    Severity Prediction for COVID-19 Patients via Recurrent Neural Networks
        Junghwan Lee, Casey Ta, Jae Hyun Kim, Cong Liu, Chunhua Weng
        medRxiv 2020.08.28.20184200; doi: https://doi.org/10.1101/2020.08.28.20184200

### Installation
The source codes were written in Python 3.7.1.
1. Clone this repository.
2. Move to the directory where you clone this repository and install requirements

        Pip install -r requirements.txt

### Prepare Data
The RNN model and baselines require patient records, demographic information of the patients, patient labels, and pre-trained embedding (optional). Example format of the data is available at /data.
* Patient records: medical records of the patients. List of lists saved in pickle file format. For example, there are two patients who made two visits, then corresponding patient records are [[[1,2] , [3,4,5]], [[1,3], [4,5,6]]]. This record can be interpreted as the first patient has concept 1 and 2 at the first visit and concept 3, 4, and 5 at the second visit; and the second patient has concept 1 and 3 at the first visit and concept 4, 5, and 6 at the second visit. For efficient handling of the data, all medical concepts are required to be mapped to integer and a mapping dictionary has to be provided to implement the models. For example, {“concept 1” : 0, “concept 2” : 1} represents “concept 1” and “concept 2” are mapped to integer code 0 and 1. Those mapped integer codes are used in representing patient records, not real value of the concepts. The mapping dictionary is also needed to be saved in pickle file.
* Demographic information of the patents: List of demographic information of the patients saved in pickle file format. You can create your own demographic information vector. We used sex and age information in the paper (e.g., [1, 0, 0.32] for a male patient whose normalized age is 0.32 and [0, 1, 0.53] for a female patient whose normalized age is 0.53.
* Patient labels: List of binary labels saved in pickle file format indicating whether a patient was progressed into a critical status or not. 1 indicates progression of critical status and 0 indicates moderate patient.
* Pre-trained embedding: Pre-trained embedding of medical concepts appeared in the data. This should be saved in numpy data format (.npy). We used [GloVe]() for pre-trained embedding. You can implement models without preparing pre-trained embedding, but pre-trained embedding can improve models performance as described in the paper.

### To implement the RNN model

### To implement the baselines
