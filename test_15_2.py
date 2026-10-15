import requests

# Few-shot prefix that matches the training prompt distribution.
# Including this in every API call is critical: the model was trained on
# completion-style prompts with these examples, so sending the same format
# at inference time elicits <tool>...</tool><arg>...</arg> responses instead
# of the base model's default <think>... behavior.
FEW_SHOT_PREFIX = (
    "Tools: calculator (math), weather (city name), search (query).\n"
    "Format: <tool>NAME</tool><arg>ARG</arg>\n\n"
    "Task: What is 2+3? <tool>calculator</tool><arg>2+3</arg>\n"
    "Task: Weather in Paris? <tool>weather</tool><arg>Paris</arg>\n"
    "Task: Search AI news? <tool>search</tool><arg>AI news</arg>\n"
)


def call_tool_finetuned_model(prompt, api_key):
    """
    Call your RL-trained Tinker tool-use model via the OpenAI-compatible API.

    Args:
        prompt  (str): the user's task/question (e.g. "What is the weather in Chicago?")
        api_key (str): Tinker API key (loaded from tinker_key.txt by the autograder)

    Returns:
        str: the model's raw response (should contain <tool>...</tool><arg>...</arg>)

    Raises:
        ValueError: if prompt is None or empty
    """
    # TODO: implement this function.
    #
    # Step 1: Validate — if not prompt, raise ValueError("prompt cannot be empty or None").
    #
    # Step 2: Set constants:
    #         TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    #         TINKER_MODEL    = "tinker://YOUR_MODEL_ID/..."  <- replace with your model path
    #
    # Step 3: Build the full prompt by appending the task to FEW_SHOT_PREFIX:
    #         full_prompt = FEW_SHOT_PREFIX + f"Task: {prompt}?"
    #
    # Step 4: Build headers:
    #         {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    #
    # Step 5: Build request body:
    #         {"model": TINKER_MODEL,
    #          "messages": [{"role": "user", "content": full_prompt}],
    #          "temperature": 0,
    #          "max_tokens": 64}
    #
    # Step 6: response = requests.post(f"{TINKER_BASE_URL}/chat/completions",
    #                                  headers=headers, json=data)
    #         response.raise_for_status()
    #
    # Step 7: return response.json()["choices"][0]["message"]["content"]
    
    if not prompt:
        raise ValueError("prompt cannot be empty or None")
        
    TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    TINKER_MODEL    = "tinker://57950f96-4abc-5a2b-a000-516c660c7324:train:0/sampler_weights/tool-rl-final"  
    
    full_prompt = FEW_SHOT_PREFIX + f"Task: {prompt}?"
    
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    
    data = {
        "model": TINKER_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0,
        "max_tokens": 64,
    }

    response = requests.post(f"{TINKER_BASE_URL}/chat/completions",
                                      headers=headers, json=data)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    key = os.getenv('TINKER_API_KEY')
    print('key loaded:', bool(key))
    text = call_tool_finetuned_model('What is the weather in Chicago?', key)
    print('returned:', repr(text))