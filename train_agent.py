import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import core logic and new Agent
from src.appr_core import GridEnvironment, enrich_data_with_forecast
from src.appr_agent import DQNAgent

# --- PREPARE DATA ---
print("Generating Data...")
HORIZONTE_DIAS = 3
HORAS = 24 * HORIZONTE_DIAS
index = pd.date_range(start='2024-06-01', periods=HORAS, freq='h')
df = pd.DataFrame(index=index)

df['Demanda_MW'] = 70 + 10 * np.sin(2 * np.pi * (df.index.hour) / 24)
df['Generacion_Solar_MW'] = 90 + 30 * np.sin(2 * np.pi * (df.index.hour - 6) / 48)
df['Generacion_Solar_MW'] = df['Generacion_Solar_MW'].clip(lower=0) 

# Add Forecast
df = enrich_data_with_forecast(df, forecast_error_std=5.0)

# --- ENVIRONMENT CONFIG ---
CAPACIDAD_RESTRINGIDA = 80.0
CAPACIDAD_BATERIA_MWh = 60.0
TASA_MAX_MW = 20.0

env = GridEnvironment(
    df=df, 
    capacity_limit=CAPACIDAD_RESTRINGIDA, 
    battery_capacity=CAPACIDAD_BATERIA_MWh,
    battery_rate=TASA_MAX_MW,
    training_mode=True
)

# --- AGENT SETUP ---
agent = DQNAgent(state_size=env.state_space_size, action_size=env.action_space_size)

# --- TRAINING LOOP ---
EPISODES = 100
history = []

print(f"\nStarting PyTorch training for {EPISODES} episodes...")

for e in range(EPISODES):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # Act
        action = agent.act(state)
        
        # Step
        next_state, reward, done, _ = env.step(action)
        
        # Remember
        agent.remember(state, action, reward, next_state, done)
        
        # Learn (Replay)
        agent.replay()
        
        state = next_state
        episode_reward += reward
        
        if done:
            break

    # Epsilon Decay & Target Network Update
    agent.decay_epsilon()
    if e % 10 == 0:
        agent.update_target_network()
        print(f"Episode {e}/{EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    history.append(episode_reward)

print("\nTraining Completed.")
print(f"Example Final State Vector: {state}")
