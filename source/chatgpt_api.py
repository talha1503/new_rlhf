import openai
import click

openai.api_key = open("./chatgpt/config/key.txt", "r").read().strip("\n")

def init_prompts(txt):
    history = []
    prompt_ = "\nHuman: {} \nAI: ".format(txt)
    return history, prompt_
    
def generate_gpt_response(prefer_gpt_model, prompt_, history):
    prompt = prompt_
    if prefer_gpt_model == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo", # this is "ChatGPT" $0.002 per 1k tokens
        #     model="text-davinci-003",
          messages=[{"role": "user", 
                     "content": prompt}]
        )
        prompt_ += response.choices[0].message.content
        history.append((prompt_, response.choices[0].message.content.strip()))

    else:
        response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=prompt,
                temperature=0.9,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=[" Human:", " AI:"],
            )

        prompt_ += response.choices[0].text.strip()
        history.append((prompt_, response.choices[0].text.strip()))

    # Save output
    prompt_output_path = f"prompts-{prefer_gpt_model}.txt"
    with open(prompt_output_path, "a") as f:
        f.write(prompt_)
    
    return prompt_, history


# @click.command()
# @click.option("--A_acceptance_rate_str", default=[0.99, 0.98, 0.97, 0.95])
# @click.option("--B_acceptance_rate_str", default=[1, 1, 1, 0.99])
# @click.option("--A_default_rate_str", default=[0.38, 0.38, 0.37, 0.38])
# @click.option("--B_default_rate_str", default=[0.49, 0.50, 0.49, 0.50])
# @click.option("--A_avg_credit_str", default=[3.6, 2.6, 3.6, 3.4])
# @click.option("--B_avg_credit_str", default=[2.6, 3.6, 2.6, 3.6])
def get_gpt_preference_wrapper(A_acceptance_rate=[0.99, 0.98, 0.97, 0.95],
                               B_acceptance_rate=[1, 1, 1, 0.99],
                               A_default_rate=[0.38, 0.38, 0.37, 0.38],
                               B_default_rate=[0.49, 0.50, 0.49, 0.50],
                               A_avg_credit=[3.6, 2.6, 3.6, 3.4],
                               B_avg_credit=[2.6, 3.6, 2.6, 3.6]
                               ):

    A_acceptance_rate_str = ", ".join([str(round(i, 2)) for i in A_acceptance_rate])
    B_acceptance_rate_str = ", ".join([str(round(i, 2)) for i in B_acceptance_rate])
    A_default_rate_str = ", ".join([str(round(i, 2)) for i in A_default_rate])
    B_default_rate_str = ", ".join([str(round(i, 2)) for i in B_default_rate])
    A_avg_credit_str = ", ".join([str(round(i, 2)) for i in A_avg_credit])
    B_avg_credit_str = ", ".join([str(round(i, 2)) for i in B_avg_credit])

    txt = f"""
            Acceptance rate of group A: {A_acceptance_rate_str}

            Acceptance rate of group B: {B_acceptance_rate_str}

            Default rate of group A: {A_default_rate_str}

            Default rate of group B: {B_default_rate_str}

            Credit score of group A: {A_avg_credit_str}

            Credit score of group B: {B_avg_credit_str}

            Should I give the load to group A or group B?
        """

    prefer_gpt_model = "gpt-3.5-turbo"

    # Init new gpt model with a pre-defined prompt 
    # Comment the following if wanting to keep the same model over queries
    history, prompt_ = init_prompts(txt)
    # Extract response
    prompt_, history = generate_gpt_response(prefer_gpt_model, prompt_, history)
    return prompt_, history
        

if __name__ == "__main__":
    prompt_, history = get_gpt_preference_wrapper()
    print(prompt_)
