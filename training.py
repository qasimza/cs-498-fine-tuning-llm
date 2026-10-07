import json
import tinker 
import torch
import transformers
import os
from dotenv import load_dotenv



# Load variables from .env into os.environ
load_dotenv()
api_key = os.getenv("TINKER_API_KEY")

def create_training_data(tokenizer):

    data = list()

    with open("./cot_train.jsonl") as file:
        
        for line in file:
            
            example = json.loads(line)

            messages = example["messages"]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True
            ) 

            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            )

            input_tokens = token_ids[:-1]
            target_tokens = token_ids[1:]
            
            datum = tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(
                    input_tokens
                ),
                loss_fn_inputs={
                    "target_tokens": tinker.types.TensorData.from_torch(
                        torch.tensor(target_tokens, dtype=torch.long)
                    ),
                    "weights": tinker.types.TensorData.from_torch(
                        torch.ones(
                            len(target_tokens),
                            dtype=torch.float32
                        )
                    ),
                },
            )

            data.append(datum)

    return data       

async def train_model(training_client, training_data):
    for step in range(20):

        print(f"Training step {step + 1}/20")

        await training_client.forward_backward_async(
            training_data,
            "cross_entropy",
        )

        await training_client.optim_step_async(
            tinker.types.AdamParams(
                learning_rate=1e-4,
            )
        )

async def main():
    
    service_client = tinker.ServiceClient()
    
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-8B",
        rank=16
    )

    tokenizer = training_client.get_tokenizer()

    training_data = create_training_data(tokenizer)

    print(f"Loaded {len(training_data)} training examples.")

    await train_model(training_client, training_data)

    print("Model Training Completed!")

    sampler_future = training_client.save_weights_for_sampler(
    "cot-reasoning-model"
)
    checkpoint = await sampler_future.result_async()

    print("Training complete!")
    print("Console URL:", training_client.get_console_url())
    print("Model path:", checkpoint.path)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

