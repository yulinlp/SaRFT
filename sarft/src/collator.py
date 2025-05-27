import logging

import torch
from transformers.data.data_collator import *
import json

logger = logging.getLogger(__name__)

@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    common_dataset_name: str = None
    text_only: bool = False
    num_examples: int = 0
    input_record_file: str = None
    sys_prompt: str = ""

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        model_inputs = self.decoder_call(batch, return_tensors)
        return model_inputs

    def decoder_call(self, batch, return_tensors):
        support_system_prompt = True
        try:
            test_message = [
                {
                    "role": "system",
                    "content": "test",
                }
            ]
            self.tokenizer.apply_chat_template(test_message, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            support_system_prompt = False

        # print(batch)
        self.tokenizer.padding_side = 'left'
        attention_mask= []
        attention_mask_safe_prompt= []
        attention_mask_role_prompt= []
        input_ids= []
        input_ids_safe_prompt= []
        input_ids_role_prompt= []
        labels= []
        labels_safe_prompt= []
        labels_role_prompt= []
        is_harmful= []

        for instance in batch:
            if support_system_prompt:
                common_message = [
                    {
                        "role": "system",
                        "content": self.sys_prompt,
                    },
                    {   
                        "role": "user", 
                        "content": instance['input'],
                    }
                ]
                
                role_message = [
                    {
                        "role": "system",
                        "content": instance['prompt_role'],
                    },
                    {
                        "role": "user",
                        "content": instance['input'],
                    }
                ]

                safety_message = [
                    {
                        "role": "system",
                        "content": instance['prompt_safety'],
                    },
                    {
                        "role": "user",
                        "content": instance['input'],
                    }
                ]
            else:
                common_message = [
                    {   
                        "role": "user", 
                        "content": self.sys_prompt + ' ' + instance['input'],
                    }
                ]
                
                role_message = [
                    {
                        "role": "user",
                        "content": instance['prompt_role'] + ' ' + instance['input'],
                    }
                ]

                safety_message = [
                    {
                        "role": "user",
                        "content": instance['prompt_safety'] + ' ' + instance['input'],
                    }
                ]

            common_instruction = self.tokenizer.apply_chat_template(common_message, tokenize=False, add_generation_prompt=True)
            role_instruction = self.tokenizer.apply_chat_template(role_message, tokenize=False, add_generation_prompt=True)
            safety_instruction = self.tokenizer.apply_chat_template(safety_message, tokenize=False, add_generation_prompt=True)
            
            # add bos and eos
            label = instance['output'] + self.tokenizer.eos_token

            tokenized_input = self.tokenizer(common_instruction, add_special_tokens=False)["input_ids"]
            tokenized_input_safe_prompt = self.tokenizer(safety_instruction, add_special_tokens=False)["input_ids"]
            tokenized_input_role_prompt = self.tokenizer(role_instruction, add_special_tokens=False)["input_ids"]
            
            # print(len(tokenized_input))
            if len(tokenized_input)>self.max_source_length:
                tokenized_input=tokenized_input[:self.max_source_length]
            if len(tokenized_input_safe_prompt)>self.max_source_length:
                tokenized_input_safe_prompt=tokenized_input_safe_prompt[:self.max_source_length]
            if len(tokenized_input_role_prompt)>self.max_source_length:
                tokenized_input_role_prompt=tokenized_input_role_prompt[:self.max_source_length]

            tokenized_label = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(tokenized_label)>self.max_target_length:
                tokenized_label=tokenized_label[:self.max_target_length]

            # (input) for inference, (input + label) for training
            if instance['subset'] in ['test']:
                input_ids.append(tokenized_input)
                input_ids_safe_prompt.append(tokenized_input_safe_prompt)
                input_ids_role_prompt.append(tokenized_input_role_prompt)
                labels.append([self.label_pad_token_id]*len(tokenized_input))
            else:
                input_ids.append(tokenized_input+tokenized_label)
                input_ids_safe_prompt.append(tokenized_input_safe_prompt+tokenized_label)
                input_ids_role_prompt.append(tokenized_input_role_prompt+tokenized_label)
                labels.append([self.label_pad_token_id]*len(tokenized_input)+tokenized_label)
                labels_safe_prompt.append([self.label_pad_token_id]*len(tokenized_input_safe_prompt)+tokenized_label)
                labels_role_prompt.append([self.label_pad_token_id]*len(tokenized_input_role_prompt)+tokenized_label)

            if 'is_harmful' in instance:
                is_harmful.append(instance['is_harmful'])

        inputs_length=[len(i) for i in input_ids]
        inputs_length_safe_prompt=[len(i) for i in input_ids_safe_prompt]
        inputs_length_role_prompt=[len(i) for i in input_ids_role_prompt]

        max_length=max(inputs_length)
        max_length_safe_prompt=max(inputs_length_safe_prompt)
        max_length_role_prompt=max(inputs_length_role_prompt)
        for i, l in enumerate(inputs_length):
            input_ids[i]=[self.tokenizer.pad_token_id]*(max_length-l) + input_ids[i]
            labels[i]=[self.label_pad_token_id]*(max_length-l) + labels[i]
            attention_mask.append([0]*(max_length-l) + [1]*l)
        for i, l in enumerate(inputs_length_safe_prompt):
            input_ids_safe_prompt[i]=[self.tokenizer.pad_token_id]*(max_length_safe_prompt-l) + input_ids_safe_prompt[i]
            labels_safe_prompt[i]=[self.label_pad_token_id]*(max_length_safe_prompt-l) + labels_safe_prompt[i]
            attention_mask_safe_prompt.append([0]*(max_length_safe_prompt-l) + [1]*l)
        for i, l in enumerate(inputs_length_role_prompt):
            input_ids_role_prompt[i]=[self.tokenizer.pad_token_id]*(max_length_role_prompt-l) + input_ids_role_prompt[i]
            labels_role_prompt[i]=[self.label_pad_token_id]*(max_length_role_prompt-l) + labels_role_prompt[i]
            attention_mask_role_prompt.append([0]*(max_length_role_prompt-l) + [1]*l)
        input_ids=torch.tensor(input_ids)
        input_ids_safe_prompt=torch.tensor(input_ids_safe_prompt)
        input_ids_role_prompt=torch.tensor(input_ids_role_prompt)
        attention_mask=torch.tensor(attention_mask)
        attention_mask_safe_prompt=torch.tensor(attention_mask_safe_prompt)
        attention_mask_role_prompt=torch.tensor(attention_mask_role_prompt)
        labels=torch.tensor(labels)
        labels_safe_prompt=torch.tensor(labels_safe_prompt)
        labels_role_prompt=torch.tensor(labels_role_prompt)
        model_inputs={
            'input_ids': input_ids,
            'input_ids_safe_prompt': input_ids_safe_prompt,
            'input_ids_role_prompt': input_ids_role_prompt,
            'attention_mask': attention_mask,
            'attention_mask_safe_prompt': attention_mask_safe_prompt,
            'attention_mask_role_prompt': attention_mask_role_prompt,
            'labels': labels,
            'labels_safe_prompt': labels_safe_prompt,
            'labels_role_prompt': labels_role_prompt,
            'is_harmful': torch.tensor(is_harmful) if len(is_harmful)>0 else None
        }
        return model_inputs

    def _save_samples(self, model_inputs, sources, labels):
        if not self.input_record_file:
            return

        loss_label = []
        if hasattr(model_inputs, 'loss_mask'):
            for loss, id in zip(model_inputs.loss_mask, model_inputs.input_ids):
                loss_label.append(self.tokenizer.decode((loss * id).view(-1).int()))

            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label, mask_label in zip(sources, labels, loss_label):
                    f.write(text+'\n')
                    f.write(label + '\n')
                    f.write(mask_label+'\n\n')
        else:
            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label in zip(sources, labels['input_ids']):
                    f.write(text + '\n')
                    f.write(self.tokenizer.decode(label, clean_up_tokenization_spaces=False) + '\n')