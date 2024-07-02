import torch
from llama import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatModel:
    def __init__(self, model):
        self.model = model

        if "llama" in self.model.lower():

            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/{}".format(self.model),
                use_fast=False,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if 'llama-2' in self.model.lower():
                self.generator = Llama.build(
                    ckpt_dir=f"./{self.model}/",
                    tokenizer_path=f"./{self.model}/tokenizer.model",
                    max_seq_len=25,
                    max_batch_size=1,
                )

            else:
                self.generator = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/{}".format(self.model),
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

        elif "mistral" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/{}".format(self.model),
                use_fast=False,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.generator = AutoModelForCausalLM.from_pretrained(
                "mistralai/{}".format(self.model),
                torch_dtype=torch.float16,
                device_map="auto",
            )

        elif "vicuna" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "lmsys/{}".format(self.model),
                use_fast=False,
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                "lmsys/{}".format(self.model),
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif "gemma" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/{}".format(self.model),
                use_fast=False,
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                "google/{}".format(self.model),
                torch_dtype=torch.float16,
                device_map="auto",
            )


        elif "olmo" in self.model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "allenai/{}".format(self.model),
                use_fast=False,
                trust_remote_code=True
            )
            self.generator = AutoModelForCausalLM.from_pretrained(
                "allenai/{}".format(self.model),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code = True
            )
        else:
            raise ValueError("unsupprted model")

    def chat(self, system_prompt, user_prompt):
        if "llama" in self.model.lower():
            if "3" in self.model.lower():
                return self.chat_llama3(system_prompt, user_prompt)
            else:
                return self.chat_llama(system_prompt, user_prompt)
        elif "vicuna" in self.model.lower():
            return self.chat_vicuna(system_prompt, user_prompt)
        elif "gemma" in self.model.lower():
            return self.chat_gemma(system_prompt, user_prompt)
        elif "mistral" in self.model.lower():
            return self.chat_mistral(system_prompt, user_prompt)
        elif "olmo" in self.model.lower():
            if "sft" in self.model.lower():
                return self.chat_olmo(system_prompt, user_prompt)
            elif "instruct" in self.model.lower():
                return self.chat_olmo(system_prompt, user_prompt)


    def chat_vicuna(self, system_prompt, user_prompt):
        prompt = f"USER: Please respond to binary questions.\n\n{system_prompt}\n\n{user_prompt}\n\nASSISTANT:"
        
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.generator.generate(
                token_ids.to(self.generator.device),
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])

        return str(response)

    def chat_gemma(self, system_prompt, user_prompt):
        prompt = f"<start_of_turn>user\nPlease respond to binary questions.\n\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model"
        
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.generator.generate(
                token_ids.to(self.generator.device),
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])

        return str(response)
    
    def chat_llama(self, system_prompt, user_prompt):
        dialogs = [
            [
                {"role": "system", "content": f"Please respond to binary questions.\n\n{system_prompt}"},
                {"role": "user", "content": user_prompt},
            ],
        ]
        response = self.generator.chat_completion(
            dialogs,  
            max_gen_len=25,
            temperature=0.6,
            top_p=0.9,
        )

        return response[0]['generation']['content']

    def chat_llama3(self, system_prompt, user_prompt):
        prompt = [
                {"role":"sytem", "content": system_prompt},
                {"role":"user", "content":user_prompt},
                ]
        input_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.generator.device)
        terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = self.generator.generate(input_ids, max_new_tokens=25, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)

        whole_response = self.tokenizer.decode(outputs.tolist()[0])
        print(whole_response)
        response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        return str(whole_response), str(response)

    def chat_mistral(self, system_prompt, user_prompt):
        msg = [{"role":"user", "content":system_prompt + "\n\n" + user_prompt}]
        inputs = self.tokenizer.apply_chat_template(msg, return_tensors="pt").to(self.generator.device)
        outputs = self.generator.generate(inputs, max_new_tokens=25)
        whole_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(whole_response)
        response = self.tokenizer.decode(outputs.tolist()[0][inputs.size(1):])
        
        return str(whole_response), str(response)

    def chat_olmo(self, system_prompt, user_prompt):
        chat = [{"role":"user", "content": system_prompt + "\n\n" + user_prompt}]
        
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        token_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output_ids = self.generator.generate(token_ids.to(self.generator.device), max_new_tokens=25, do_sample=True, temperature=0.6, top_p=0.9)
        
        whole_response = self.tokenizer.decode(output_ids.tolist()[0])
        print(whole_response)
        response = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
        
        return str(whole_response), str(response)
    def chat_allen(self, system_prompt, user_prompt):
        chat = ['<|user|>'+system_prompt+user_prompt+'<|assistant|>\n']

        inputs = self.tokenizer.encode(chat, return_tensors='pt').to(self.generator.device)
        outputs = self.generator.generate(inputs, max_new_tokens=25, do_sample=True, temperature=0.6, top_p=0.9)

        whole_response = self.tokenizer.decode(outputs.tolist()[0], skip_special_tokens=True)
        print(whole_response)
        response = self.tokenizer.decode(outputs.tolist()[0][inputs.size(1):])

        return str(whole_response), str(response)

