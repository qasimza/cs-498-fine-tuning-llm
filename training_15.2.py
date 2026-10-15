import re
import tinker
from tinker import types
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("TINKER_API_KEY")

# ── Setup ────────────────────────────────────────────────────────────────────
service_client  = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",

)
tokenizer = training_client.get_tokenizer()

VALID_TOOLS = {"calculator", "weather", "search"}

PROMPTS = [
    ("What is 15 plus 27?",               "calculator"),
    ("What is 6 times 8?",                "calculator"),
    ("What is 100 divided by 4?",         "calculator"),
    ("What is the weather in London?",     "weather"),
    ("What is the weather in New York?",   "weather"),
    ("What is the weather in Sydney?",     "weather"),
    ("Search for information about pandas","search"),
    ("Search for the population of Japan", "search"),
]

# Completion-style prompt ending with <tool> so the base model
# immediately continues the XML pattern — avoids reward=0 cold start.
def make_prompt_tokens(task: str) -> list:
    prompt = (
        "Tools: calculator (math), weather (city name), search (query).\n"
        "Format: <tool>NAME</tool><arg>ARG</arg>\n\n"
        "Task: What is 2+3? <tool>calculator</tool><arg>2+3</arg>\n"
        "Task: Weather in Paris? <tool>weather</tool><arg>Paris</arg>\n"
        "Task: Search AI news? <tool>search</tool><arg>AI news</arg>\n"
        f"Task: {task}? <tool>"
    )
    return tokenizer.encode(prompt, add_special_tokens=True)

def make_grpo_datum(prompt_tokens, resp_tokens, advantage):
    w = max(0.0, advantage)
    tokens  = prompt_tokens + resp_tokens
    weights = [0.0] * len(prompt_tokens) + [w] * len(resp_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs=dict(weights=weights[1:], target_tokens=tokens[1:]),
    )

def compute_reward(response: str, correct_tool: str) -> float:
    """
    Return:
      3.0  if <tool> matches correct_tool AND <arg> is non-empty
      2.0  if <tool> is in VALID_TOOLS (but wrong) AND <arg> is non-empty
      1.0  if both tags present but tool not in VALID_TOOLS
      0.0  otherwise (missing tags, empty tool, empty arg)
    """
    tool = re.findall(r"<tool>(.*?)</tool>", response)
    if len(tool) == 0: 
        return 0.0
    tool = tool[0]

    arg = re.findall(r"<arg>(.*?)</arg>", response)
    if len(arg) == 0: 
        return 0.0
    arg = arg[0]

    if tool == "" or arg == "":
        return 0


def compute_advantages(rewards: list) -> list:
    """
    GRPO: normalize rewards to zero mean, unit population std.
    Return all 0.0s if std < 1e-8. Raise ValueError if rewards is empty.
    """
    if len(rewards) == 0:
        raise ValueError("Rewards must not be empty")

    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((reward - mean) ** 2 for reward in rewards) / n
    std = variance ** 0.5

    if std < 1e-8:
        return [0.0] * n

    return [(reward - mean) / std for reward in rewards]

# ── Training loop (provided) ─────────────────────────────────────────────────
G         = 4
NUM_ITERS = 5
LR        = 5e-5

sampling_client = training_client.save_weights_and_get_sampling_client(name="tool-rl-init")

for iteration in range(NUM_ITERS):
    datums, total_reward = [], 0.0

    for task_text, correct_tool in PROMPTS:
        p_tokens = make_prompt_tokens(task_text)
        result   = sampling_client.sample(
            prompt=types.ModelInput.from_ints(p_tokens),
            num_samples=G,
            sampling_params=types.SamplingParams(max_tokens=32, temperature=0.9),
        ).result()

        rewards    = [compute_reward(tokenizer.decode(s.tokens), correct_tool)
                      for s in result.sequences]
        advantages = compute_advantages(rewards)
        total_reward += sum(rewards) / len(rewards)

        for seq, adv in zip(result.sequences, advantages):
            if adv != 0.0:
                datums.append(make_grpo_datum(p_tokens, list(seq.tokens), adv))

    if not datums:
        print(f"Iter {iteration + 1}: no signal — check compute_reward and compute_advantages")
        continue

    training_client.forward_backward(datums, "cross_entropy").result()
    training_client.optim_step(types.AdamParams(learning_rate=LR)).result()
    print(f"Iter {iteration + 1}/{NUM_ITERS}  avg_reward={total_reward / len(PROMPTS):.2f}  datums={len(datums)}")

    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"tool-rl-v{iteration + 1}"
    )

save_result = training_client.save_weights_for_sampler(name="tool-rl-final").result()
print(f"Training complete! Model path: {save_result.path}")