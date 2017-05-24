import sys,os,re,time, pickle, sqlite3, random
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import KFold, LeaveOneOut
import utils

"""Classes and methods for training a neural resolution model on the
TAKE dataset. The trained model takes a pair (utterance, object feature)
as input and outputs the probability of the utterrance fitting the object.

Once, learned, the model can then be used to derive a sequential model that 
constructs a distributed representation of the utterance token-by-token."""

from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras import backend as K
from keras.layers import Dense, Input, Embedding, Dropout, GRU, Activation
from keras.layers.merge import Dot, concatenate
from keras.engine import Model
from keras.optimizers import RMSprop
from keras.regularizers import l1,l2
from keras.models import load_model
import tensorflow as tf

GPU_FRACTION=0.50
import keras.backend.tensorflow_backend as KTF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_FRACTION)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(session)

TAKE_DB_FILE = "take.db"

  
class ResolutionModel:
    """Neural resolution model, using Keras"""
    
    def __init__(self, episodes=None, existing_model_file=None):
        """Initialise the model (using either an existing model file, or a set of episodes)"""
        
        if existing_model_file and os.path.exists(existing_model_file):
            self.model = load_model(existing_model_file)
            return
        elif episodes is None:
            raise RuntimeError("must provide either existing model or input data")
        
        # Extract some global parameters from the episodes
        self.max_token_length = max([len(e.utterance) for e in episodes])
        scene_feat_length = len(utils.encode_features(episodes[0].referent))
        max_token_id = max(utils.dico.values())
        
        # Constructs the embedding layer (here we use a dimension=50)
        self.embedding_layer = Embedding(output_dim=50, input_dim=(max_token_id+1),
                                         name="embedding_layer", mask_zero=True)
        
        # Constructs the recurrent layer (with also an output dimension equal to the number of visual features)
        self.recurrent_layer = GRU(scene_feat_length, activation=None, name="recurrent_layer", 
                                   dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l1(0.0001))
        
        # The model takes the sequence of tokens + visual features of the object
        tokens_input = Input(shape=(self.max_token_length,), name="utterance_input", dtype='int32')
        scene_input = Input(shape=(scene_feat_length,), name="scene_input", dtype="float32")

        # We create a vector representation of the utterance
        embeddings = self.embedding_layer(tokens_input)
        predictions = self.recurrent_layer(embeddings)
        
        # And take the dot product of this representation with the visual features
        dotproduct = Dot(axes=1)([predictions, scene_input])
        output = Activation("sigmoid")(dotproduct)
        
        # The model is  optimised to minimise the binary cross-entropy
        rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.001)
        model = Model(inputs=[tokens_input,scene_input], outputs=output)        
        model.compile(optimizer=rmsprop,loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
     
     
    def train(self, episodes, positive_weight=15, batch_size=128, 
              epochs=30, nb_partial_episodes=0, incremental_episodes=True):
        """Train the model using a set of episodes. The positive examples are assigned
        a higher weight since they are less frequent. """

        # If we wish, we can learn the model using partial utterances (either cut after 
        # each word, or at fixed intervals) instead of only the full utterances. 
        # It does not seem to improve the model performance, however.)
        if incremental_episodes:
            episodes = [x for e in episodes for x in e.get_incremental_episodes()]
        
        # Creates the inputs and outputs for the training process
        input_tokens = []
        input_scenes = []
        outputs = []
        for e in episodes:
            tokens = utils.encode_tokens(e.utterance, self.max_token_length)
            all_objects = [e.referent] + e.distractors
            input_tokens += [tokens]*(len(all_objects))
            input_scenes += [utils.encode_features(o) for o in all_objects]
            outputs += [1] + [0]*len(e.distractors)             
             
        inputs = [np.array(input_tokens), np.array(input_scenes)]
        outputs = np.array(outputs)
   
        self.model.fit(inputs, outputs, batch_size=batch_size,epochs=epochs,
                   class_weight={1:positive_weight,0:1})
        
        
    def test(self, episodes):
        """Evaluates the accuracy (ratio of episodes where the correct referent object
        is identified amongst the set of possible candidates) of the model."""
        
        nb_correct = 0.0
        for e in episodes:
            probs = self.predict_prob(e.utterance, [e.referent] + e.distractors)
            if probs.argmax()==0:
                nb_correct += 1
        return nb_correct / len(episodes)
        
    
    def predict_prob(self, utterance, objects):
        """ Given an utterance and a set of possible objects, returns an array
        of probabilities, where each column i indicates the probability that the
        object i is the referent object for the utterance."""
        
        tokens = utils.encode_tokens(utterance, self.max_token_length)
        input_utterances = [tokens]*len(objects)
        input_objects = [utils.encode_features(o) for o in objects]
        inputs = [np.array(input_utterances), np.array(input_objects)]
        predictions = self.model.predict(inputs)
        return predictions.squeeze()

    
    def cross_evaluate(self, episodes, nb_folds=10, nb_epochs=30, 
                       nb_partial_episodes=0):
        """ Performs a cross-evaluation of the model on the set of episodes."""
            
        k_fold = KFold(n_splits=nb_folds, shuffle=True)
        results = []
         
        for i, (train_indices, test_indices) in enumerate(k_fold.split(episodes)):
            self.__init__(episodes=episodes)
            training_data = [episodes[x] for x in train_indices]
            testing_data = [episodes[x] for x in test_indices]

            self.train(training_data, epochs=nb_epochs)
            accuracy = self.test(testing_data)
            
            print("Accuracy on fold %i:"%(i+1), accuracy)
            print("(on training data: %.2f)"% self.test(training_data))
            results.append(accuracy)
            
        print("Average accuracy", np.array(results).mean())
    
    
    def save(self):
        self.model.save("fitmodel.hdf5")

            
    def get_incremental_model(self):
        """Returns an incremental version of the sequential model. The returned model
        takes 2 inputs: a token and a previous state (representing the utterance so far), 
        and returns a new state. """
        
        token_input = Input(shape=(1,), name="incremental_input", dtype='int32')
        state_input = Input(shape=(self.recurrent_layer.units,), name="previous_state", dtype="float32")                                  
        
        embeddings = self.embedding_layer(token_input)
        predictions = self.recurrent_layer(embeddings, initial_state=state_input)
        
        model = Model(inputs=[token_input, state_input], outputs=predictions)        
        model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    
    def get_nonincremental_model(self):
        """Returns a non-incremental version of the model, taking a full sequence of tokens,
        and returning its final state representation."""
        
        token_input = Input(shape=(31,), name="usual_input", dtype='int32')
        
        embeddings = self.embedding_layer(token_input)
        predictions = self.recurrent_layer(embeddings)
        
        model = Model(inputs=token_input, outputs=predictions)        
        model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
        return model



class Episode:
    """Representation of a training episode, composed of an utterance and a list of objects."""
    
    def __init__(self, epid, utterance, objects, referent_id):
        self.epid = epid
        self.utterance = utterance
        self.referent = [o for o in objects if o.name[1]==referent_id][0]
        self.distractors = [o for o in objects if o.name[1]!=referent_id]

    def get_partial_episodes(self, nb_partial_episodes):
        """Returns a list of partial episodes cut at fixed intervals."""
        partial_episodes = []
        for n in range(nb_partial_episodes):
            stop_point = int(len(self.utterance)*(n+1)/(nb_partial_episodes+1))
            partial_utterance = self.utterance[0:stop_point]
            new_ep = Episode("%s-%i"%(self.epid,n), partial_utterance, 
                             [self.referent] + self.distractors, self.referent.name[1])
            partial_episodes.append(new_ep)
        return partial_episodes    

    def get_incremental_episodes(self):
        """Returns a list of partial episodes, extending the utterance token-by-token"""
        partial_episodes = []
        utterance = [x for x in self.utterance if x!="<sil>"]
        for i in range(1, len(utterance)-1):
            partial_utterance = utterance[0:i]
            new_ep = Episode("%s-%i"%(self.epid,i), partial_utterance, 
                             [self.referent] + self.distractors, self.referent.name[1])
            partial_episodes.append(new_ep)
        return partial_episodes    
 
 

def extract_episodes():
    """Extract the training episodes from the TAKE dataset"""
    
    db = sqlite3.connect(TAKE_DB_FILE)
    df_transcriptions = pd.read_sql("SELECT episode_id, inc, word from hand", con=db)
    utterances = {}
    for epid, sub_df in df_transcriptions.groupby("episode_id"):
        sub_df = sub_df.sort_values("inc")
        utterance = []
        for w in sub_df["word"]:
            utterance.append(w)
        utterances[epid] = utterance
    
    df_pieces = pd.read_sql("SELECT * from piece", con=db)
    df_pieces = df_pieces.set_index(["episode_id", "id"])
    objects = {}
    for epid, obj_id in df_pieces.index:
        objects[epid] = objects.get(epid,[]) + [df_pieces.ix[(epid,obj_id)]]

    referents = {}
    df_referent = pd.read_sql("SELECT * from referent", con=db)
    df_referent = df_referent.set_index("episode_id")
    for epid in df_referent.index:
        referents[epid] = df_referent.ix[epid,"object"]
    db.close()    
   
    episodes = []
    for epid in utterances:
        episodes.append(Episode(epid, utterances[epid], objects[epid], referents[epid]))
    
    if not os.path.exists("dico.pkl"):
        print("Regenerating dictionary")
        [utils.encode_tokens(u, None) for u in utterances.values()]
        pickle.dump(utils.dico, open("dico.pkl", "wb"))

    return episodes

