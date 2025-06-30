import tiktoken
import os 
import glob
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template
import numpy as np
import argparse
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

from datetime import datetime, timezone
import time
import torch

def reset_rope(model, model_max_train_len, scaling_factor):
    """
    Not sure if I need this
    """
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device="cpu", dtype=torch.float32)
    return

class LLMNeedleHaystackTester:
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 
                 model_name='',
                 results_version = 1,

                 context_lengths_min = 1000,
                 context_lengths_max = 128000,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,

                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 print_ongoing_status = True):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.

        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.

        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question

        self.model_name = model_name
        self.results_version = results_version

        self.save_results = save_results
        self.save_contexts = save_contexts
        self.print_ongoing_status = print_ongoing_status

        self.final_context_length_buffer = final_context_length_buffer

        self.testing_results = []

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths
        
        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type not in ["linear", "sigmoid"]:
                    raise ValueError("document_depth_percent_interval_type must be either 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents
        
        self.load_model()

        self.evaluation_model = None
        self.debug='debug'
        
    def load_model(self):
        assert self.model_name in ('mistral-7B-instruct-v0.2',
                                   'llama-2-7B-32k-instruct',
                                   'longchat-v1.5-7b-32k',
                                   'vicuna-v1.5-7b-16k',
                                   'lwm-text-chat-1m'), (
                                       f'You are using model: {self.model_name} that we do not support.'
                                    )
        
        model2path = json.load(open("/root/SnapKV/experiments/LongBench/config/model2path.json", "r"))
        path = model2path[self.model_name]

        self.model_to_test = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            use_flash_attention_2=True
        ).eval()

        if self.model_name == 'mistral-7B-instruct-v0.2':
            self.enc = AutoTokenizer.from_pretrained(
                path,
                padding_side="right",
                use_fast=False,
            )
        else:
            self.enc = AutoTokenizer.from_pretrained(
                path,
                use_fast=False,
            )

        # ---------- Not sure if I need them ----------
        # scaling_factor = 10 # hardcode
        # reset_rope(self.model_to_test, model_max_train_len=81920, scaling_factor=scaling_factor)
        # import tensor_parallel as tp
        # self.model_to_test = tp.tensor_parallel(self.model_to_test, sharded=True)
        # ---------------------------------------------

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)
    
    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len:
                continue

            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)
    
    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure

        # You may also try:
        # (f"You are a helpful AI bot that answers questions for a user. Keep your response short and direct."
        #  f"\n\n{context}\n\n"
        #  f"Question: {self.retrieval_question} Don't give information outside the document or repeat your findings. "
        #  f"The document definitely contains the answer, and I'm 100% sure. So try your best to find it.")
        
        # test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
        test_format=f"This is a very long story book:\n\n{context}\n\nBased on the content of the book, Question: {self.retrieval_question}\n\nAnswer:"

        if self.model_name == 'longchat-v1.5-7b-32k' or self.model_name == 'vicuna-v1.5-7b-16k':
            # you may consider to remove "Answer:" in this case. Because it already have "ASSISTANT:"
            test_format = test_format[:len('Answer:')]

            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], test_format)
            conv.append_message(conv.roles[1], None)
            test_format = conv.get_prompt()
        elif self.model_name == 'llama-2-7B-32k-instruct' or self.model_name == 'lwm-text-chat-1m':
            test_format = f"[INST] {test_format} [/INST]"
        elif self.model_name == 'mistral-7B-instruct-v0.2':
            test_format = test_format
        else:
            raise NotImplementedError(f'We currently do not support {self.model_name}.')

        return test_format
    
    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")
        
        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)

        # ----------- Start generate -----------
        test_start_time = time.time()

        input = self.enc(prompt, truncation=False, return_tensors="pt").to(self.model_to_test.device)
        input_ids_len = input.input_ids.shape[-1]

        outputs = self.model_to_test.generate(
            **input,
            max_new_tokens=50,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=input_ids_len+1,
        )
        response = self.enc.decode(outputs[0][input_ids_len:], skip_special_tokens=True).strip()
        
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        # ----------------------------------------

        score = scorer.score(self.needle, response)['rouge1'].fmeasure*10

        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }
        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")
        
        context_file_location = f'{self.model_name}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            # not recommended -> too long
            results['file_name'] = context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            with open(f'contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
        
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results'):
                os.makedirs('results')

            # Save the result to file for retesting
            p = f'results/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)
