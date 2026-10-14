import torch
import torch.nn as nn
from benchmark.gymnasium.continuous_mountain_car import ContinuousMountainCar
from methods.reinforcement_learning.actor_critic.network import ActorCriticNetwork
from methods.reinforcement_learning.advantage_estimator.advantage_estimator import GeneralizedAdvantageEstimator

lr = 1e-4
n_episodes = 300
gamma = 0.999

env = ContinuousMountainCar(verbose = True)

model = ActorCriticNetwork(
    state_dim = env.state_dim, 
    hidden_dims = (64, 64), 
    action_dim = env.action_dim, 
    activation = nn.ReLU, 
)
advantage_estimator: GeneralizedAdvantageEstimator = GeneralizedAdvantageEstimator(
    discount_factor = gamma, 
    trace_decay = 1
)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr = lr
)

for episode in range(n_episodes):
    state, info = env.reset()
    episode_reward = 0
    step = 0
    done = False
    
    log_probs = []
    entropies = []
    values = []
    while not done:
        mean, std, value_now = model.forward(torch.as_tensor(state, dtype = torch.float32))
        action_distribution = torch.distributions.Normal(mean, std)
        action = action_distribution.sample()
        next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
        done = terminated or truncated
        
        with torch.no_grad():
            _, _, value_next = model(torch.as_tensor(next_state, dtype=torch.float32))
            if terminated:
                value_next = torch.zeros_like(value_next)

        log_probs.append(action_distribution.log_prob(action).sum())        
        entropies.append(action_distribution.entropy().sum())
        values.append(value_now)
        advantages = advantage_estimator(reward , value_next.detach(), value_now.detach(), terminated, truncated)
        if advantages != None:
            actor_loss = []
            critic_loss = []
            for advantage, log_prob, value in zip(advantages, log_probs, values):            
                actor_loss.append(-advantage * log_prob)
                critic_loss.append((value - value.detach() - advantage).pow(2).mean())
                
            loss = torch.stack(actor_loss).mean() + torch.stack(critic_loss).mean() * 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = next_state
        episode_reward += reward
        step += 1
    
    print(
        f"Episode {episode + 1} | "
        f"Reward={episode_reward:.2f} | "
        f"Steps={step} | "
        f"mean={mean.detach().cpu().numpy()} | "
        f"std={std.detach().cpu().numpy()} | "
        f"terminated={terminated} | "
        f"truncated={truncated}"
    )