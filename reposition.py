import numpy as np
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
        # State: (vehicles per station, tasks per station, time)
        n_v = [sum(1 for v in self.vehicles if v.station == s.id) for s in self.stations]
        n_t = [sum(1 for t in self.tasks if t.dest == s.id) for s in self.stations]
        time = [1 if i == self.time_slot % 288 else 0 for i in range(288)]  # 5-min slots in a day
        return np.concatenate([n_v, n_t, time, [1 if s.id == vehicle.station else 0 for s in self.stations]])
    
    def step(self, vehicle, action):
        # Action: Move to station index or stay
        if action != len(self.stations):  # Not staying
            vehicle.station = self.stations[action].id
        n_v = sum(1 for v in self.vehicles if v.station == vehicle.station)
        n_t = sum(1 for t in self.tasks if t.dest == vehicle.station)
        total_fee = sum(t.fee for t in self.tasks if t.dest == vehicle.station)
        
        # Reward (Eq. 2)
        if n_v != 0 and n_t != 0:
            reward = total_fee / n_v
        elif n_v == 0:
            reward = 0
        else:
            reward = -1
        return self.get_state(vehicle), reward

# Restriction Rule (simplified)
def restriction_rule(state, vehicle, stations):
    F = [1] * (len(stations) + 1)  # +1 for stay
    v_station = vehicle.station
    n_v = state[:len(stations)]
    n_t = state[len(stations):2*len(stations)]
    
    for i, s in enumerate(stations):
        # Vehicle supply-demand
        if n_v[i] <= n_v[stations.index(next(s for s in stations if s.id == v_station))] or n_t[i] == 0:
            F[i] = 0
        # Electricity (assume >20% is fine)
        if vehicle.electricity < 20:
            F[i] = 0
    return np.array(F)

# RDR Algorithm (Algorithm 2)
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
            tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def act(self, state, vehicle, stations):
        F = restriction_rule(state, vehicle, stations)
        q_values = self.model.predict(np.array([state]))[0] * F
        if random.random() < self.epsilon:
            return random.choice([i for i, f in enumerate(F) if f == 1])
        return np.argmax(q_values)
    
    def train(self, env, episodes=10):
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
                    targets = self.model.predict(states)
                    target_next = self.target_model.predict(next_states)
                    
                    for i, (s, a, r, ns) in enumerate(batch):
                        targets[i][a] = r + self.gamma * np.max(target_next[i])
                    
                    self.model.fit(states, targets, epochs=1, verbose=0)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Sync target model
            if e % 5 == 0:
                self.target_model.set_weights(self.model.get_weights())

# RAR Algorithm (Algorithm 3)
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
            tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def build_critic(self, state_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def act(self, state, vehicle, stations):
        F = restriction_rule(state, vehicle, stations)
        probs = self.actor.predict(np.array([state]))[0] * F
        probs = probs / np.sum(probs)  # Normalize
        return np.random.choice(len(probs), p=probs)
    
    def train(self, env, episodes=10):
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
                    
                    # Update critic
                    targets = rewards + self.gamma * self.critic_target.predict(next_states).flatten()
                    self.critic.fit(states, targets, epochs=1, verbose=0)
                    
                    # Update actor
                    advantages = targets - self.critic.predict(states).flatten()
                    action_probs = self.actor.predict(states)
                    for i, a in enumerate(actions):
                        action_probs[i][a] = advantages[i]
                    self.actor.fit(states, action_probs, epochs=1, verbose=0)
            
            # Sync target critic
            if e % 5 == 0:
                self.critic_target.set_weights(self.critic.get_weights())

def reposition_vehicles(stations, vehicles, tasks, assignments, algo='rdr'):
    env = Environment(stations, vehicles, tasks)
    state_size = len(env.get_state(vehicles[0]))
    action_size = len(stations) + 1  # +1 for stay
    
    if algo == 'rdr':
        agent = RDR(state_size, action_size)
    else:
        agent = RAR(state_size, action_size)
    
    # Train for a few episodes
    agent.train(env, episodes=5)
    
    # Apply actions
    repositioning = []
    assigned_vehicles = {a.split(" -> ")[0] for a in assignments}
    idle_vehicles = [v for v in vehicles if v.id not in assigned_vehicles]
    
    for v in idle_vehicles:
        state = env.get_state(v)
        action = agent.act(state, v, stations)
        if action != len(stations):  # Not staying
            repositioning.append(f"{v.id} moves from {v.station} to {stations[action].id}")
            v.station = stations[action].id
    
    return repositioning