
# Incremental processing for neural conversation models

## 1) Training

To test the code, you first need to train a neural model (which is optimised on the TAKE dataset):

```
>>> import training
>>> episodes = training.extract_episodes()
>>> mod = training.ResolutionModel(episodes)
>>> mod.train(episodes)
```

You can also perform a k-fold cross-validation of the model using the "cross-evaluate" model.


## 2) Incremental processing

After training, the model has optimised the weights of two trainable layers:
- an embedding layer (mapping tokens to a low-dimensional space)
-  a recurrent layer (in our case a Gated Recurrent Unit) that constructs a fixed-length representation of the utterance from the sequence of embeddings. 

These weights can be applied to map any sequence of tokens to a vector representation. To get a incremental model, one can simply fix the input to be a pair (token, previous state), where token is a new token, and state is the vector representation of the utterancee so far. The model then outputs an updated vector representation.

To test this incremental model, I implemented in interaction.py some simple methods for updating a neural conversation model with incremental units:

```
>>> incr_model = mod.get_incremental_model()
>>> import interaction
>>> dialog = interaction.Interaction()
>>> dialog.insert_unit("das")
>>> dialog.insert_unit_continued("rote")
>>> dialog.insert_unit_continued("kreuz")
...
```

Each new token triggers the neural model to create a new state representation, linked to the previous ones. Each insertion can be associated with a probability (if the probability < 1, the new state is a mixture of (1-p)*old_state + p*new_state). The incremental units can also be revoked.  

## 3) To do:

- Generates the incremental units from the actual speech, using the Google API.
- Test the model on other datasets than TAKE?
