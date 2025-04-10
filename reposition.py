import numpy as np # type: ignore
import tensorflow as tf
from data import Vehicle
import random

# Simplified environment for simulation
class Environment:
    def __init__(self, stations, vehicles, tasks):
        self.stations = stations
        self.vehicles = vehicles
        self.tasks = tasks
        self.time_slot = 0
    
    def get_state(self, vehicle):
        n_v = [sum(1 for v in self.vehicles if v.station == s.id) for s in self.stations]
        n_t = [sum(1 for t in self.tasks if t.dest == s.id) for s in self.stations]
        time = [1 if i == self.time_slot % 288 else 0 for i in range(288)]
        return np.concatenate([n_v, n_t, time, [1 if s.id == vehicle.station else 0 for s in self.stations]])
    
    def step(self, vehicle, action):
        if action != len(self.stations):  # Not staying
            vehicle.station = self.stations[action].id
        n_v = sum(1 for v in self.vehicles if v.station == vehicle.station)
        n_t = sum(1 for t in self.tasks if t.dest == vehicle.station)
        total_fee = sum(t.fee for t in self.tasks if t.dest == vehicle.station)
        reward = total_fee / n_v if n_v > 0 and n_t > 0 else (-1 if n_v > 0 else 0)
        return self.get_state(vehicle), reward

# Fixed restriction rule
def restriction_rule(state, vehicle, stations):
    F = [1] * (len(stations) + 1)  # +1 for stay
    v_station = vehicle.station
    n_v = state[:len(stations)]
    n_t = state[len(stations):2*len(stations)]
    
    # Find current station index safely
    try:
        current_station_idx = next(i for i, s in enumerate(stations) if s.id == v_station)
    except StopIteration:
        # If vehicle’s station isn’t in stations, default to first station or stay
        current_station_idx = 0  # Fallback
    
    for i in range(len(stations)):
        if n_v[i] <= n_v[current_station_idx] or n_t[i] == 0:
            F[i] = 0
        if vehicle.electricity < 20:
            F[i] = 0
    return np.array(F)

# RDR Algorithm
class RDR:
    def __init__(self, state_size, action_size):
        self.model = self.build_model(state_size, action_size)
        self.target_model = self.build_model(state_size, action_size)
        self.memory = []
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
    
    def build_model(self, state_size, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),  # Fix: Use Input layer
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def act(self, state, vehicle, stations):
        F = restriction_rule(state, vehicle, stations)
        q_values = self.model.predict(np.array([state]), verbose=0)[0] * F
        if random.random() < self.epsilon:
            valid_actions = [i for i, f in enumerate(F) if f == 1]
            return random.choice(valid_actions) if valid_actions else len(stations)  # Stay if no valid actions
        return np.argmax(q_values)
    
    def train(self, env, episodes=5):
        for e in range(episodes):
            for v in env.vehicles:
                state = env.get_state(v)
                action = self.act(state, v, env.stations)
                next_state, reward = env.step(v, action)
                self.memory.append((state, action, reward, next_state))
                
                if len(self.memory) > self.batch_size:
                    batch = random.sample(self.memory, self.batch_size)
                    states = np.array([t[0] for t in batch])
                    next_states = np.array([t[3] for t in batch])
                    targets = self.model.predict(states, verbose=0)
                    target_next = self.target_model.predict(next_states, verbose=0)
                    
                    for i, (s, a, r, ns) in enumerate(batch):
                        targets[i][a] = r + self.gamma * np.max(target_next[i])
                    
                    self.model.fit(states, targets, epochs=1, verbose=0)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if e % 5 == 0:
                self.target_model.set_weights(self.model.get_weights())

# RAR Algorithm
class RAR:
    def __init__(self, state_size, action_size):
        self.actor = self.build_actor(state_size, action_size)
        self.critic = self.build_critic(state_size)
        self.critic_target = self.build_critic(state_size)
        self.memory = []
        self.gamma = 0.95
        self.batch_size = 32
    
    def build_actor(self, state_size, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),  # Fix: Use Input layer
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def build_critic(self, state_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),  # Fix: Use Input layer
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def act(self, state, vehicle, stations):
        F = restriction_rule(state, vehicle, stations)
        probs = self.actor.predict(np.array([state]), verbose=0)[0] * F
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum  # Normalize
            return np.random.choice(len(probs), p=probs)
        return len(stations)  # Stay if no valid actions
    
    def train(self, env, episodes=5):
        for e in range(episodes):
            for v in env.vehicles:
                state = env.get_state(v)
                action = self.act(state, v, env.stations)
                next_state, reward = env.step(v, action)
                self.memory.append((state, action, reward, next_state))
                
                if len(self.memory) > self.batch_size:
                    batch = random.sample(self.memory, self.batch_size)
                    states = np.array([t[0] for t in batch])
                    actions = np.array([t[1] for t in batch])
                    rewards = np.array([t[2] for t in batch])
                    next_states = np.array([t[3] for t in batch])
                    
                    targets = rewards + self.gamma * self.critic_target.predict(next_states, verbose=0).flatten()
                    self.critic.fit(states, targets, epochs=1, verbose=0)
                    
                    advantages = targets - self.critic.predict(states, verbose=0).flatten()
                    action_probs = self.actor.predict(states, verbose=0)
                    for i, a in enumerate(actions):
                        action_probs[i][a] = advantages[i]
                    self.actor.fit(states, action_probs, epochs=1, verbose=0)
            
            if e % 5 == 0:
                self.critic_target.set_weights(self.critic.get_weights())

def reposition_vehicles(stations, vehicles, tasks, assignments, algo='rdr'):
    env = Environment(stations, vehicles, tasks)
    state_size = len(env.get_state(vehicles[0]))
    action_size = len(stations) + 1
    
    if algo == 'rdr':
        agent = RDR(state_size, action_size)
    else:
        agent = RAR(state_size, action_size)
    
    agent.train(env, episodes=5)
    
    repositioning = []
    assigned_vehicles = {a.split(" -> ")[0] for a in assignments}
    idle_vehicles = [v for v in vehicles if v.id not in assigned_vehicles]
    
    for v in idle_vehicles:
        state = env.get_state(v)
        action = agent.act(state, v, stations)
        if action != len(stations):
            repositioning.append(f"{v.id} moves from {v.station} to {stations[action].id}")
            v.station = stations[action].id
    
    return repositioning