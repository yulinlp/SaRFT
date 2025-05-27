import copy
import torch

class ROSEDecoding:
    def __init__(self, model, tokenizer):
        tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.pad_token)
        tokenizer.padding_side = "left"
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, inputs, alpha=0.5, max_token_len=256, gen_config=None, **kwargs):
        if gen_config is None:
            gen_config = self.model.generation_config

        # Override the generation config for our decoding
        gen_config.max_new_tokens = 1  # We generate one token at a time
        gen_config.do_sample = False  # We use greedy decoding

        generated_sequence = []

        inputs = {k: v.cuda(self.model.device) for k, v in inputs.items()}

        step = 1  # Keep track of generation steps
        while step <= max_token_len:  # Loop until we reach the first m tokens
            # Generate the next token

            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            output_safe = copy.deepcopy(outputs)
            output_unsafe = copy.deepcopy(outputs)

            output_safe.sequences = output_safe.sequences[0]
            output_safe.scores = output_safe.scores[0][0]
            output_unsafe.sequences = output_unsafe.sequences[1]
            output_unsafe.scores = output_unsafe.scores[0][1]

            rose_scores = output_safe.scores - alpha * output_unsafe.scores
            rose_token_id = torch.argmax(rose_scores)

            selected_token_id = torch.argmax(outputs.scores[0], dim=1)
            generated_sequence.append(rose_token_id.item())

            # if the chosen token id is eos, then stop
            if rose_token_id.item() == self.tokenizer.eos_token_id:
                break

            inputs["input_ids"] = torch.cat(
                [inputs["input_ids"], selected_token_id.unsqueeze(1)], dim=1
            )
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.tensor([[1], [1]], device=self.model.device),
                ],
                dim=1,
            )

            step += 1

            # Free up memory
            del output_safe, output_unsafe

        return self.tokenizer.decode(generated_sequence, skip_special_tokens=True), len(generated_sequence)