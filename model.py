import openai

class Model:
    def __init__(self, model_name, temperature):

        self.model = model_name
        self.temperature = temperature

    def ask_chatgpt(self, prompt):

        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temperature,
            top_p=1,
            max_tokens=256,
            frequency_penalty=0,
            presence_penalty=0,
            messages=prompt,
            api_key='',  # Replace with your actual API key
            n=1,
        )
        return response.choices[0].message['content'].lower()
