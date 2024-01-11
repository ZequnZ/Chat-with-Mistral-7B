from typing import Iterator
import gradio as gr
import random
import time

from text_generation import Client

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

API_URL = "https://api-inference.huggingface.co/models/" + model_id
HF_TOKEN = "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"
SYSTEM_PROMPT = "I want you to act as a great assistant. You will provide trustful information and can inspire me to think more using supportive languages."

client = Client(
    API_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
)
EOS_STRING = "</s>"
EOT_STRING = "<EOT>"

generate_kwargs = dict(
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
    top_k=20,
    temperature=0.6,
)


def generate_prompts(
    sys_prompt: str, input: str, history: list[tuple[str, str]]
) -> str:
    prompt = f"<s>[INST] {sys_prompt} [/INST]</s>\n\n"
    context = ""
    for user_input, model_output in history:
        # prompt+=f"[INST]{input} {model_output}[/INST]"
        # prompt+=f"[User input]{user_input} [Model output]{model_output}\n\n"
        if user_input != "":
            context += f"{user_input}:\n{model_output}\n"
    if context != "":
        prompt += "[INST] Below are some Context between me and you, which can be used as reference to answer [Next user input] and stop when finishing answering:\n"
        prompt += context
        prompt += f"[/INST]\n\n[Next user input]:\n\n"
    prompt += f"{input}\n"
    return prompt


# theme = gr.themes.Base()
theme = "WeixuanYuan/Soft_dark"

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Chat with Mistral-7B\n[Github](https://github.com/ZequnZ/Chat-with-Mistral-7B)")
    with gr.Row():
        chatbot = gr.Chatbot(scale=6)

        with gr.Column(variant="compact", scale=1):
            gr.Markdown("## Parameters:")
            max_new_tokens = gr.Slider(
                label="Max new tokens",
                minimum=1,
                maximum=1024,
                step=1,
                value=128,
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=2,
                step=0.1,
                value=0.6,
            )
            top_p = gr.Slider(
                label="Top-p (nucleus sampling)",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.9,
            )
            top_k = gr.Slider(
                label="Top-k",
                minimum=1,
                maximum=100,
                step=1,
                value=10,
            )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="What do you wanna ask?",
            scale=10,
        )
        submit_bt = gr.Button("✔️ Submit", scale=1, variant=1)
    with gr.Row():
        clear_bt = gr.Button("🗑️ Clear")
        remove_bt = gr.Button("← Remove last input")
        retry_bt = gr.Button("🔄 Retry")

    system_prompt = gr.Textbox(
        label="System prompt/Instruction",
        value=SYSTEM_PROMPT,
        lines=3,
        interactive=True,
    )

    # Submit the message in textbox
    def sub_msg(user_message, history) -> tuple[str, list[tuple[str, str]]]:
        if not history == None:
            return "", history + [[user_message, None]]
        else:
            return "", [[user_message, None]]

    def remove_last_dialogue(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if history:
            history.pop()
        return history

    def remove_last_output(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if history:
            last_dialogue = history.pop()
            history.append([last_dialogue[0], None])
        return history

    def output_messages(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return history

    def bot(history: list[tuple[str, str]]) -> Iterator[list[tuple[str, str]]]:
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    def call_llm(
        history: list[tuple[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: float,
        sys_prompt: str,
    ) -> Iterator[list[tuple[str, str]]]:
        generate_kwargs = dict(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        if history:
            prompt = generate_prompts(sys_prompt, history[-1][0], history[:-1])
            history[-1][1] = ""
            print("prompt: ", prompt)

            stream = client.generate_stream(prompt, **generate_kwargs)
            time.sleep(3)

            for response in stream:
                if response.token.text != EOS_STRING:
                    history[-1][1] += response.token.text
                    time.sleep(0.05)
                yield history
        return []

    textbox.submit(sub_msg, [textbox, chatbot], [textbox, chatbot], queue=False).then(
        fn=call_llm,
        inputs=[chatbot, max_new_tokens, temperature, top_p, top_k, system_prompt],
        outputs=chatbot,
    )
    submit_bt.click(
        sub_msg, [textbox, chatbot], [textbox, chatbot], queue=False, show_progress=True
    ).then(
        fn=call_llm,
        inputs=[chatbot, max_new_tokens, temperature, top_p, top_k, system_prompt],
        outputs=chatbot,
    )

    # CLear all the history
    clear_bt.click(lambda: None, None, chatbot, queue=False)

    remove_bt.click(remove_last_dialogue, [chatbot], [chatbot], queue=False).then(
        output_messages, chatbot, chatbot
    )

    retry_bt.click(
        fn=remove_last_output, inputs=[chatbot], outputs=[chatbot], queue=False
    ).then(
        fn=call_llm,
        inputs=[chatbot, max_new_tokens, temperature, top_p, top_k, system_prompt],
        outputs=chatbot,
    )


if __name__ == "__main__":
    demo.launch()
