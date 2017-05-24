import sys,os,re,time, pickle, sqlite3, random
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import KFold, LeaveOneOut

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
ALL_FEATURES =  ['col', 'color', 'grid', 'row', 'type', 'x', 'y']

COLOR_CODES = {'blue': 0, 'cyan': 1, 'gray': 2, 'green': 3, 
               'magenta': 4, 'red': 5, 'yellow': 6}
TYPE_CODES = {'F': 0, 'I': 1,  'L': 2, 'N': 3,  'P': 4, 'T': 5,
              'U': 6,'V': 7, 'X': 8, 'Y': 9, 'Z': 10}

dico = {}
stemmer =  SnowballStemmer("german")


class Episode:
    
    def __init__(self, utterance, objects, referent_id=None):
        self.utterance = utterance
        self.objects = objects
        if referent_id:
            self.referent = next([o for o in self.objects if o["id"]==referent_id])
        else:
            self.referent = None
    
    def get_tokens(self, remove_sil=True, length=31):  
        token_list = []
        for w in self.utterance:
            if remove_sil and w=="<sil>":
                continue
            w = stemmer.stem(w)
            if w not in dico:
                dico[w] = len(dico) + 1
            token_list.append(dico[w])
        tokens = token_list[-31:] if len(token_list) > length else token_list
        padded_tokens = np.zeros(length)
        padded_tokens[-len(tokens):] = tokens
        return padded_tokens

    
    def get_bag_of_words(self, nb_tokens=306):
        tokens = self.get_tokens(length=None)
        bag_of_words = np.zeros(nb_tokens)
        for j in tokens:
            if j  > 0:
                bag_of_words[int(j)-1] += 1
        return bag_of_words
               
    def get_referent(self):
        return encode_features(self.referent)
      
    def get_distractors(self):
        return [encode_features(d) for d in self.objects and d is not self.referent]
    
    def get_objects(self):
        return [encode_features(d) for d in self.objects]

      
    def get_partial_episodes(self, nb_partial_episodes):
        partial_episodes = []
        for n in range(nb_partial_episodes):
            stop_point = int(len(self.utterance)*(n+1)/(nb_partial_episodes+1))
            partial_utterance = self.utterance[0:stop_point]
            new_ep = Episode(partial_utterance, self.objects, self.referent["id"])
            partial_episodes.append(new_ep)
        return partial_episodes
 
        

  
class ResolutionModel:

    def __init__(self, episodes=None, existing_model_file=None, bag_of_words=False):
        if existing_model_file and os.path.exists(existing_model_file):
            self.model = load_model(existing_model_file)
            return
        elif episodes is None:
            raise RuntimeError("must provide either existing model or input data")
        
        max_token_length = len(episodes[0].get_tokens())
        scene_feat_length = len(episodes[0].get_referent())    
        max_token_id = int(max([x for e in episodes for x in e.get_tokens()]))
        self.bag_of_words = bag_of_words
        
        if bag_of_words:    
            tokens_input = Input(shape=(max_token_id,), name="utterance_input", dtype='float32')
            tokens = Dropout(0.5)(tokens_input)
            prediction = Dense(scene_feat_length, name="hidden_layer", 
                               kernel_regularizer=l1(0.001))(tokens)
    
        else:
            tokens_input = Input(shape=(max_token_length,), name="utterance_input", dtype='int32')
            embeddings = Embedding(output_dim=scene_feat_length, input_dim=(max_token_id+1),
                                   name="embedding_layer")(tokens_input)
            prediction = GRU(scene_feat_length, activation=None,  name="recurrent_layer", 
                            dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l1(0.001))(embeddings) 

        scene_input = Input(shape=(scene_feat_length,), name="scene_input", dtype="float32")
        dotproduct = Dot(axes=1)([prediction, scene_input])
        output = Activation("sigmoid")(dotproduct)
        rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.001)
        model = Model(inputs=[tokens_input,scene_input], outputs=output)        
        model.compile(optimizer=rmsprop,loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
     
     
    def train(self, episodes, positive_weight=15, batch_size=128, 
              epochs=30, nb_partial_episodes=0):

        if nb_partial_episodes:
            for e in list(episodes):
                episodes += e.get_partial_episodes(nb_partial_episodes)
        
        input_tokens = []
        input_scenes = []
        outputs = []
        for e in episodes:
            tokens = e.get_tokens() if not self.bag_of_words else e.get_bow()
            input_tokens += [tokens]*(1+len(e.distractors))
            input_scenes += [e.get_referent()] + e.get_distractors()
            outputs += [1] + [0]*len(e.distractors)             
            
        inputs = [np.array(input_tokens), np.array(input_scenes)]
        outputs = np.array(outputs)
   
        self.model.fit(inputs, outputs, batch_size=batch_size,epochs=epochs,
                   class_weight={1:positive_weight,0:1})
        
        
    def test(self, episodes):
        nb_correct = 0.0
        for e in episodes:
            probs = self.predict_prob(e)
            if probs.argmax()==0:
                nb_correct += 1
        return nb_correct / len(episodes)
        
    
    def predict_prob(self, episode):
        input_utterance = episode.get_bag_of_words() if self.bag_of_words else episode.get_tokens()
        input_utterances = [input_utterance]*len(episode.objects)  
        input_objects = episode.get_objects()
        inputs = [np.array(input_utterances), np.array(input_objects)]
        predictions = self.model.predict(inputs)
        return predictions.squeeze()

    
    def cross_evaluate(self, episodes, nb_folds=10, nb_epochs=40, 
                       nb_partial_episodes=0):
            
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
        
    
    def get_incremental_model(self):
           
        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        
        token_input = Input(shape=(1,), name="incremental_input", dtype='int32')
                                                         
        if self.bag_of_words:
            pass
        else:
            prev_embeddings = layer_dict["embedding_layer"]
            embeddings = Embedding(output_dim=prev_embeddings.output_dim, input_dim=1,
                                   weights=prev_embeddings.get_weights())(token_input)
            prev_recurrent = layer_dict["recurrent_layer"]
            recurrent = GRU(prev_recurrent.units, weights=prev_recurrent.get_weights())(embeddings)    
    
        model = Model(inputs=token_input, outputs=recurrent)        
        model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    
    def save(self):
        self.model.save("fitmodel.hdf5")



class Interaction:
    
    def __init__(self):
        self.initial_state = np.zeros(31)
        self.iunits = {}
        self.previous = {}
        self.grounding = {}
        self.model = IncrementalModel()
    
    
    def insert_unit(self, token, prev_unit_id=None, prob=1.0):
        
        unit_id = "IU-%i"%(len(self.iunits)+1)
        self.iunits[unit_id] = token
        if prev_unit_id:
            if prev_unit_id in self.grounding:
                prev_state_id = self.grounding[prev_unit_id]
                prev_state = self.iunits[prev_state_id]
                new_state_id = "IState-%i"%(len(self.iunits)+1)
                new_state = self.model.increment(token, prev_state)
                self.iunits[new_state_id] = new_state
                self.previous[new_state_id] = prev_state_id
                self.grounding[new_state_id] = unit_id            
                state_id = 
            self.previous[unit_id] = prev_unit_id
        else:
            new_state = self.model.increment(token, self.initial_state)
            
        return unit_id
    

    def _increment_state(self, token_unit_id, previous_state_id):
        
    
    def predict_prob(self, scenes):
        pass
      

class IncrementalModel:
    
    def __init__(self, trained_model):
    
        layer_dict = dict([(layer.name, layer) for layer in trained_model.model.layers])
        
        tokens_input = Input(shape=(1,), name="incremental_input", dtype='int32')
        state_input
        if trained_model.bag_of_words:
            pass
        else:
            prev_embeddings = layer_dict["embedding_layer"]
            embeddings = Embedding(output_dim=prev_embeddings.output_dim, 
                                   input_dim=prev_embeddings.input_dim,
                                   weights=prev_embeddings.get_weights())(tokens_input)
            prev_recurrent = layer_dict["recurrent_layer"]
            recurrent = GRU(prev_recurrent.units, weights=prev_recurrent.get_weights())(embeddings)
    
        model = Model(inputs=[tokens_input,scene_input], outputs=output)        
        model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        
             
    def increment(self, token, previous_state):
        pass
    
    
    def predict(self, state, scene_input):
        pass



def extract_episodes():
    
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
        episodes.append(Episode(epid, utterances[epid],objects[epid], referents[epid]))

    return episodes

     
    def encode_features(obj):
        one_hot_list = []
        
        one_hot_col = [0]*3
        one_hot_col[obj["col"]] = 1
        one_hot_list += one_hot_col
        
        one_hot_color = [0]*len(COLOR_CODES)
        one_hot_color[COLOR_CODES[obj["color"]]] = 1
        one_hot_list += one_hot_color
        
        if obj["grid"] == "grid1":
            one_hot_list += [1, 0, 1, 0]
        elif obj["grid"] == "grid2":
            one_hot_list += [1, 0, 0, 1]
        elif obj["grid"] == "grid3":
            one_hot_list += [0, 1, 1, 0]
        elif obj["grid"] == "grid4":
            one_hot_list += [0, 1, 0, 1]
            
        one_hot_row = [0]*3
        one_hot_row[obj["row"]] = 1
        one_hot_list += one_hot_row
        
        one_hot_type = [0]*len(TYPE_CODES)
        one_hot_type[TYPE_CODES[obj["type"]]] = 1
        one_hot_list += one_hot_type
        
        one_hot_list += [(obj["x"]-175.0)/1715.0]
        one_hot_list += [(obj["y"]-125.0)/915.0]
        
        return np.array(one_hot_list)
