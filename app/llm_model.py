# app/llm_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class FineTunedGPT2:
    def __init__(self, model_dir: str = "models/gpt2_squad"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()

        # Ensure pad token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def answer_question(self, question: str, max_new_tokens: int = 64) -> str:
        """
        Given a question, generate an answer in the trained format:
        'That is a great question. ... Let me know if you have any other questions.'
        """
        prompt = (
        f"That is a great question. Here is the answer:\n"
        f"Question: {question}\nAnswer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Strip the prompt, keep only the generated part after "Answer: "
        if "Answer:" in full_text:
            answer_part = full_text.split("Answer:")[1].strip()
        else:
            answer_part = full_text

        return answer_part