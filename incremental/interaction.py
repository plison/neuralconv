
import numpy as np
import utils
import time
from scipy.special import expit

class IUnit:
    """Representation of an incremental unit, with a type, a payload, a previous
    unit (if available), a grounding unit (if available), and a timestamp."""
    
    counter = 1
    def __init__(self, unit_type, payload, previous=None, grounded_in=None):
        self.id = "%s-%i"%(unit_type, IUnit.counter)
        IUnit.counter += 1
        self.payload = payload
        self.previous = previous
        self.grounded_in = grounded_in
        self.timestamp = time.time()
    
    def __repr__(self):
        return str(self.payload)
        

class Interaction:
    """Representation of a single interaction, made of a expanding set of 
    incremental units, and a neural model processing the tokens."""
    
    def __init__(self, neural_model):
        self.units = {}
        self.neural_model = neural_model
        
    
    def get_successors(self, unit_id):
        """Returns the successors of the incremental unit"""
        successors = []
        for u in self.units.values():
            if u.previous and u.previous.id == unit_id:
                successors.append(u.id)
        return successors
    
     
    def get_grounded(self, unit_id):
        """Returns the units that are grounded in the incremental unit"""
        grounded = []
        for u in self.units.values():
            if u.grounded_in and u.grounded_in.id == unit_id:
                grounded.append(u.id)
        return grounded
         
    
    def insert_unit(self, token, prev_unit_id=None, prob=1.0):
        """Inserts a new token, possibly connected to a previous unit, with a
        particular probability value."""
        
        if prev_unit_id:
            prev_unit = self.units[prev_unit_id]
            prev_state_unit = [x for x in self.units.values() if x.grounded_in==prev_unit][0]   
        else:
            prev_unit, prev_state_unit = None, None 
        new_unit = IUnit("token", token, previous=prev_unit)
        self.units[new_unit.id] = new_unit
        
        if prev_state_unit is None:
            previous_state = np.zeros(30)
        else:
            previous_state = prev_state_unit.payload
        encoded_token = utils.encode_token(token)
        incremental_inputs = [np.reshape(np.array([encoded_token]), (1,1)),
                              np.reshape(previous_state, (1,len(previous_state)))]
        new_state = self.neural_model.predict(incremental_inputs)[0]
        if prob < 1.0:
            new_state = (1-prob)*previous_state + prob*new_state
        new_state_unit = IUnit("state", new_state, previous=prev_state_unit, grounded_in=new_unit)
        self.units[new_state_unit.id] = new_state_unit
            
        return new_unit.id
    
   
    def insert_unit_continued(self, token, prob=1.0):
        """Inserts a new token, attaching it to the last token incremental unit."""
        if self.units:
            last_unit_id = max([x for x in self.units if x.startswith("token")], 
                               key=lambda x : self.units[x].timestamp)
            self.insert_unit(token, last_unit_id, prob)
        else:
            self.insert_unit(token, None, prob)
            
    
    def predict_prob(self, objects):
        """Predicts the probability that each object is a referent for the description
        uttered so far."""
        
        last_states = [self.units[uid].payload for uid in self.units 
                       if uid.startswith("state") and not self.get_successors(uid)]
        if len(last_states) > 1:
            final_state = np.add(last_states) / len(last_states)
        else:
            final_state = last_states[0]
        
        scene_feats = [utils.encode_features(o) for o in objects]
        dot_products = np.array([np.dot(final_state, o) for o in scene_feats])
        return expit(dot_products)
      

    def revoke(self, unit_id):
        """Revoke an incremental unit"""
        if unit_id not in self.units:
            raise RuntimeError("%s not in units"%unit_id)
        to_revoke = [unit_id]
        while len(to_revoke) > 0:
            uid = to_revoke.pop()
            to_revoke += self.get_successors(uid)
            to_revoke += self.get_grounded(uid)
            del self.units[uid]
    
    
    def commit(self, unit_id):
        """Commits an incremental unit 
        (does nothing so far, but we could of course remove the states of all units
        that are in a previous relation to the unit, since they are guaranteed not to
        change anymore)"""
        pass

    
