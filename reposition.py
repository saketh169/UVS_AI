import numpy as np
import tensorflow as tf
from data import Vehicle, Task
import random
import os

class Environment:
    def __init__(self, stations, vehicles, tasks):
        self.stations = stations
        self.vehicles = vehicles
        self.tasks = tasks
        self.time_slot = 0
    
    def get_state(self, vehicle):
        n_v = [sum(1 for v in self.vehicles if v.station == s.id) for s in self.stations]
        n_t = [sum(1 for t in self.tasks if t.dest == s.id and not t.assigned) for s in self.stations]
        demand = [sum(t.fee for t in self.tasks if t.dest == s.id and not t.assigned) for s in self.stations]
        time = [1 if i == self.time_slot % 24 else 0 for i in range(24)]
        return np.concatenate([n_v, n_t, demand, time, [1 if s.id == vehicle.station else 0 for s in self.stations]])
    
    def step(self, vehicle, action):
        if action != len(self.stations):
            vehicle.station = self.stations[action].id
            vehicle.electricity = max(0, vehicle.electricity - 10)
        n_v = sum(1 for v in self.vehicles if v.station == vehicle.station) + 1
        n_t = sum(1 for t in self.tasks if t.dest == vehicle.station and not t.assigned)
        total_fee = sum(t.fee for t in self.tasks if t.dest == vehicle.station and not t.assigned)
        reward = total_fee / n_v if n_t > 0 else -5
        self.time_slot += 1
        return self.get_state(vehicle), reward

def restriction_rule(state, vehicle, stations):
    F = [1] * (len(stations) + 1)
    v_station = vehicle.station
    n_v = state[:len(stations)]
    n_t = state[len(stations):2*len(stations)]
    
    try:
        current_idx = next(i for i, s in enumerate(stations) if s.id == v_station)
    except StopIteration:
        current_idx = 0
    
    for i in range(len(stations)):
        if n_v[i] >= n_v[current_idx] + 2 or n_t[i] == 0 or vehicle.electricity < 20:
            F[i] = 0
    return np.array(F)

class RDR:
    def __init__(self, state_size, action_size):
        self.model = self.build_model(state_size, action_size)
        self.target_model = self.build_model(state_size, action_size)
        self.memory = []
        self.epsilon = 0.05
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
        if os.path.exists("rdr_model.h5"):
            self.model.load_weights("rdr_model.h5")
            self.target_model.load_weights("rdr_model.h5")
    
    def build_model(self, state_size, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
        return model
    
    def act(self, state, vehicle, stations):
        F = restriction_rule(state, vehicle, stations)
        q_values = self.model.predict(np.array([state]), verbose=0)[0] * F
        valid_actions = [i for i, f in enumerate(F) if f == 1]
        if not valid_actions:
            return len(stations)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        return np.argmax(q_values)
    
    def train(self, env, episodes=5):
        for e in range(episodes):
            for v in env.vehicles:
                state = env.get_state(v)
                action = self.act(state, v, env.stations)
                next_state, reward = env.step(v, action)
                self.memory.append((state, action, reward, next_state))
                
                if len(self.memory) > self.batch_size:
                    batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
                    states = np.array([t[0] for t in batch])
                    next_states = np.array([t[3] for t in batch])
                    targets = self.model.predict(states, verbose=0)
                    target_next = self.target_model.predict(next_states, verbose=0)
                    
                    for i, (s, a, r, ns) in enumerate(batch):
                        targets[i][a] = r + self.gamma * np.max(target_next[i])
                    
                    self.model.fit(states, targets, epochs=1, verbose=0)
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if e % 2 == 0:
                self.target_model.set_weights(self.model.get_weights())

class RAR:
    def __init__(self, state_size, action_size):
        self.actor = self.build_actor(state_size, action_size)
        self.critic = self.build_critic(state_size)
        self.critic_target = self.build_critic(state_size)
        self.memory = []
        self.gamma = 0.95
        self.batch_size = 32
        if os.path.exists("rar_actor.h5"):
            self.actor.load_weights("rar_actor.h5")
        if os.path.exists("rar_critic.h5"):
            self.critic.load_weights("rar_critic.h5")
            self.critic_target.load_weights("rar_critic.h5")
    
    def build_actor(self, state_size, action_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
        return model
    
    def build_critic(self, state_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
        return model
    
    def act(self, state, vehicle, stations):
        F = restriction_rule(state, vehicle, stations)
        probs = self.actor.predict(np.array([state]), verbose=0)[0] * F
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
            return np.random.choice(len(probs), p=probs)
        return len(stations)
    
    def train(self, env, episodes=5):
        for e in range(episodes):
            for v in env.vehicles:
                state = env.get_state(v)
                action = self.act(state, v, env.stations)
                next_state, reward = env.step(v, action)
                self.memory.append((state, action, reward, next_state))
                
                if len(self.memory) > self.batch_size:
                    batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
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
            
            if e % 2 == 0:
                self.critic_target.set_weights(self.critic.get_weights())

def simple_rule_reposition(stations, vehicles, tasks, assignments, task_threshold=1):
    repositioning = []
    assigned_vehicles = {a.split(" -> ")[0] for a in assignments}
    idle_vehicles = [v for v in vehicles if v.id not in assigned_vehicles]
    
    # Count unassigned tasks per station
    unassigned_tasks = {s.id: [] for s in stations}
    for t in tasks:
        if not t.assigned:
            unassigned_tasks[t.dest].append(t)
    
    # Apply rule: Move to stations with >task_threshold unassigned tasks
    for v in idle_vehicles:
        current_station = next((s for s in stations if s.id == v.station), None)
        if not current_station:
            continue
        
        # Find eligible stations
        eligible_stations = []
        for s in stations:
            n_tasks = len(unassigned_tasks[s.id])
            n_vehicles = sum(1 for veh in vehicles if veh.station == s.id)
            free_charging = s.charging_points - n_vehicles
            if (n_tasks > task_threshold and 
                v.electricity >= 20 and 
                free_charging > 0):
                eligible_stations.append(s)
        
        # Choose station with most tasks
        if eligible_stations:
            target = max(eligible_stations, key=lambda s: len(unassigned_tasks[s.id]))
            if target.id != v.station:
                repositioning.append(f"{v.id} moves from {v.station} to {target.id}")
                v.station = target.id
                v.electricity = max(0, v.electricity - 10)
    
    return repositioning, idle_vehicles

def verify_repositioning(repositioning, stations, tasks, vehicles):
    verified = []
    unassigned_tasks = {s.id: sum(1 for t in tasks if t.dest == s.id and not t.assigned) for s in stations}
    vehicle_counts = {s.id: sum(1 for v in vehicles if v.station == s.id) for s in stations}
    
    for r in repositioning:
        v_id, _, to_id = r.split(" moves from ")[0], *r.split(" to ")
        to_id = to_id.strip()
        
        # Common-sense checks
        if (unassigned_tasks.get(to_id, 0) > 0 and  # Has tasks
            vehicle_counts.get(to_id, 0) < unassigned_tasks.get(to_id, 0) + 2):  # Not oversupplied
            verified.append(r)
        else:
            verified.append(f"{v_id} stays at current station (invalid move to {to_id})")
    
    return verified

def reposition_vehicles(stations, vehicles, tasks, assignments, algo='rdr'):
    # Try simple rule first
    repositioning, idle_vehicles = simple_rule_reposition(stations, vehicles, tasks, assignments, task_threshold=1)
    
    # If no moves from rules, use RDR/RAR for remaining idle vehicles
    if idle_vehicles and not repositioning:
        env = Environment(stations, vehicles, tasks)
        state_size = len(env.get_state(vehicles[0]))
        action_size = len(stations) + 1
        
        if algo == 'rdr':
            agent = RDR(state_size, action_size)
        else:
            agent = RAR(state_size, action_size)
        
        for v in idle_vehicles:
            state = env.get_state(v)
            action = agent.act(state, v, stations)
            if action != len(stations):
                target_id = stations[action].id
                repositioning.append(f"{v.id} moves from {v.station} to {target_id}")
                v.station = target_id
                v.electricity = max(0, v.electricity - 10)
    
    # Verify repositioning
    verified_repositioning = verify_repositioning(repositioning, stations, tasks, vehicles)
    
    return verified_repositioning

def pre_train_and_save(stations, vehicles, tasks, algo='rdr', episodes=1000):
    env = Environment(stations, vehicles, tasks)
    state_size = len(env.get_state(vehicles[0]))
    action_size = len(stations) + 1
    
    if algo == 'rdr':
        agent = RDR(state_size, action_size)
        agent.train(env, episodes=episodes)
        agent.model.save_weights("rdr_model.h5")
    else:
        agent = RAR(state_size, action_size)
        agent.train(env, episodes=episodes)
        agent.actor.save_weights("rar_actor.h5")
        agent.critic.save_weights("rar_critic.h5")