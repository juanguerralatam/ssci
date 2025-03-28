import numpy as np
import random
from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
import logging
import json

# Enhanced logging configuration: structured logging with JSON formatting for clarity
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)

logger = logging.getLogger("AgentDeliberation")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("agent_deliberation.log")
stream_handler = logging.StreamHandler()

formatter = JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# 1. Problem Space and Fitness
def nk_landscape(x, k=2, n=None):
    """Simplified NK landscape."""
    if n is None:
        n = len(x)
    fitness = 0
    for i in range(n):
        neighbors = [(i - j) % n for j in range(1, min(k + 1, n))]
        interaction = 1
        for neighbor in neighbors:
            interaction *= np.sin(x[i] + x[neighbor])
        fitness += np.cos(x[i]) + interaction
    return fitness

def generate_beta(n):
    """Generate systematic divergence."""
    return np.random.uniform(-1, 1, n)


# 2. Agents with Diverse Perspectives
class Agent:
    def __init__(self, agent_id, position, beta, n):
        self.agent_id = agent_id
        self.position = position
        self.beta = beta
        self.n = n

    def perceived_payoff(self, x):
        payoff = nk_landscape(x) + np.dot(self.beta, x)
        logger.debug(f"Agent {self.agent_id}: computed perceived payoff {payoff:.4f} for position {x.tolist()}")
        return payoff

    def local_search(self, neighborhood_size=0.5):
        best_position = self.position.copy()
        best_payoff = self.perceived_payoff(self.position)
        for _ in range(10):  # simple local search
            new_position = self.position + np.random.uniform(-neighborhood_size, neighborhood_size, self.n)
            new_payoff = self.perceived_payoff(new_position)
            if new_payoff > best_payoff:
                best_payoff = new_payoff
                best_position = new_position
        logger.debug(f"Agent {self.agent_id}: local search best payoff {best_payoff:.4f} with position {best_position.tolist()}")
        return best_position

    def move_towards(self, target_position, alpha):
        old_position = self.position.copy()
        self.position = (1 - alpha) * self.position + alpha * target_position
        logger.info(f"Agent {self.agent_id}: Moved from {old_position.tolist()} to {self.position.tolist()} towards target {target_position.tolist()} with alpha {alpha}")


# 3. Imagined and Synthetic Deliberation
def imagined_deliberation(agents, rounds, alpha):
    logger.info(f"Starting imagined deliberation with {len(agents)} agents for {rounds} rounds (alpha={alpha})")
    for t in range(rounds):
        logger.info(f"Round {t+1}/{rounds} start")
        selected_agent = random.choice(agents)
        proposed_position = selected_agent.local_search()
        logger.info(f"Round {t+1}: Agent {selected_agent.agent_id} proposed position {proposed_position.tolist()}")
        
        for agent in agents:
            if agent != selected_agent:
                current_local = agent.local_search()
                proposed_payoff = agent.perceived_payoff(proposed_position)
                current_payoff = agent.perceived_payoff(current_local)
                logger.debug(f"Agent {agent.agent_id}: Proposed payoff {proposed_payoff:.4f} vs. current local payoff {current_payoff:.4f}")
                if proposed_payoff > current_payoff:
                    agent.move_towards(proposed_position, alpha)
        logger.info(f"Round {t+1}/{rounds} complete")
    aggregated_position = np.mean([agent.position for agent in agents], axis=0)
    logger.info(f"Imagined deliberation complete. Aggregated position: {aggregated_position.tolist()}")
    return aggregated_position

def synthetic_deliberation_langchain(agents, rounds, alpha, model_name="gpt-3.5-turbo"):
    logger.info(f"Starting synthetic deliberation (LangChain) with {len(agents)} agents for {rounds} rounds (alpha={alpha})")
    llm = ChatOllama(model="llama3.3", temperature=0)
    agent_responses = {}

    # Synthetic deliberation rounds
    for t in range(rounds):
        logger.info(f"Round {t+1}/{rounds} start")
        for agent in agents:
            proposed_position = agent.local_search()
            message = f"Agent {agent.agent_id} proposes position: {proposed_position.tolist()}."
            logger.debug(f"Agent {agent.agent_id}: Proposed position {proposed_position.tolist()}")
            for other_agent in agents:
                if agent != other_agent:
                    system_message = SystemMessage(content=f"You are Agent {other_agent.agent_id}. Agent {agent.agent_id} has proposed position {proposed_position.tolist()}. Your current position is {other_agent.position.tolist()}.")
                    human_message = HumanMessage(content="Do you accept this proposal? If so, indicate the new position. If not, explain why, and keep your current position.")
                    response = llm([system_message, human_message])
                    agent_responses[(t, agent.agent_id, other_agent.agent_id)] = response.content
                    logger.info(f"Round {t+1}: Agent {other_agent.agent_id} response to Agent {agent.agent_id}'s proposal: {response.content}")
                    
                    # Evaluate acceptance using a simple keyword check and payoff improvement
                    if "accept" in response.content.lower():
                        if other_agent.perceived_payoff(proposed_position) > other_agent.perceived_payoff(other_agent.local_search()):
                            other_agent.move_towards(proposed_position, alpha)
        logger.info(f"Round {t+1}/{rounds} complete")

    # Aggregating final positions via LLM call for each agent
    final_positions = []
    for agent in agents:
        system_message = SystemMessage(content=f"You are Agent {agent.agent_id}.")
        prompt = (
            f"In thinking independently about a green technology investment, three executives have different perspectives:\n\n"
            f"Executive A: Values carbon emission reduction but is concerned about shareholder value.\n"
            f"Executive B: Believes in a moral obligation to reduce environmental impact regardless of returns.\n"
            f"Executive C: Questions the technology's effectiveness due to inconsistent test results.\n\n"
            f"After deliberating with a willingness to adjust position (α={alpha}), what is your final position on this investment?\n\n"
            f"IMPORTANT: Respond ONLY with a list of {agents[0].n} numbers representing your coordinates, e.g. [x, y]."
        )
        human_message = HumanMessage(content=prompt)
        
        logger.info(f"Agent {agent.agent_id}: Requesting final position")
        try:
            final_response = llm([system_message, human_message])
            logger.info(f"Agent {agent.agent_id}: Received final response: {final_response.content}")
            # Evaluate response safely: fallback to current position if evaluation fails
            position_list = eval(final_response.content)
            final_positions.append(position_list)
        except Exception as e:
            logger.error(f"Agent {agent.agent_id}: Error evaluating final response. Exception: {e}. Using current position {agent.position.tolist()}")
            final_positions.append(agent.position.tolist())

    aggregated_final = np.mean(final_positions, axis=0)
    logger.info(f"Synthetic deliberation complete. Aggregated final position: {aggregated_final.tolist()}")
    return aggregated_final


# Example Usage
n_dimensions = 2
num_agents = 3
rounds = 5
alpha = 0.5

agents = [
    Agent(i, np.random.uniform(0, 2 * np.pi, n_dimensions), generate_beta(n_dimensions), n_dimensions)
    for i in range(num_agents)
]

# Log initial agent positions for traceability
for agent in agents:
    logger.info(f"Initial position of Agent {agent.agent_id}: {agent.position.tolist()}")

imagined_result = imagined_deliberation(agents.copy(), rounds, alpha)
print("Imagined Deliberation Result:", imagined_result)

synthetic_result_langchain = synthetic_deliberation_langchain(agents.copy(), rounds, alpha)
print("Synthetic Deliberation LangChain Result:", synthetic_result_langchain)