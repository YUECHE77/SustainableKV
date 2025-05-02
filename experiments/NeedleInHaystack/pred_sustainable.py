import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tiktoken
import pdb
import glob
import jieba
import json
import numpy as np
import argparse
import time
from rouge_score import rouge_scorer
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from SustainableKV.monkeypatch import sustainablekv_replace_mistral, sustainablekv_replace_llama, sustainablekv_replace_mixtral

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="data/PaulGrahamEssays", # PaulGrahamEssays  
                 retrieval_question="The best thing to do in San Francisco is: ",

                 context_lengths_min=None,
                 context_lengths_max=None,
                 context_lengths_num_intervals=40,
                 context_lengths=None,
                 step=100, 

                 document_depth_percent_min=0,
                 document_depth_percent_max=100,
                 document_depth_percent_intervals=10,
                 document_depth_percents=None,
                 document_depth_percent_interval_type="linear",

                 model_name='',
                 results_version=1,
                 final_context_length_buffer=200,

                 save_results=True,
                 save_contexts=True,
                 print_ongoing_status=True,
                 
                 use_compress=True,
                 save_folder=None):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.

        :param context_lengths_min: The minimum length of the context. Default is 0.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param step: The "step" in np.arrange()

        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.

        :param model_name: The name of the model.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens

        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        if save_folder is None:
            raise ValueError('You must provide the path to your output folder!')

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question

        self.step = step

        self.model_name = model_name
        self.results_version = results_version
        self.final_context_length_buffer = final_context_length_buffer

        self.save_results = save_results
        self.save_contexts = save_contexts
        self.print_ongoing_status = print_ongoing_status
        
        self.use_compress = use_compress
        self.save_folder = save_folder

        self.testing_results = []
        
        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                # self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
        else:
            self.context_lengths = context_lengths
        
        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents
        
        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

        self.load_model()
        self.reset_kv_args()

    def reset_kv_args(self):
        if self.use_compress:
            num_layers = len(self.model_to_test.model.layers)

            all_window_size = [16] * num_layers
            all_subseq_len = [4096*2] * num_layers
            all_attn_sink_tok = [4] * num_layers
            desired_cache_size = [2048] * num_layers
            pooling = 'maxpool'
            all_kernel_size = [7] * num_layers
            all_recycling_percent = [0.5] * num_layers
            merge = 'average'

            for i in range(num_layers):
                self.model_to_test.model.layers[i].self_attn.config.window_size = all_window_size[i]
                self.model_to_test.model.layers[i].self_attn.config.subseq_len = all_subseq_len[i]
                self.model_to_test.model.layers[i].self_attn.config.attn_sink_tok = all_attn_sink_tok[i]
                self.model_to_test.model.layers[i].self_attn.config.desired_cache_size = desired_cache_size[i]
                self.model_to_test.model.layers[i].self_attn.config.pooling = pooling
                self.model_to_test.model.layers[i].self_attn.config.kernel_size = all_kernel_size[i]
                self.model_to_test.model.layers[i].self_attn.config.recycling_percent = all_recycling_percent[i]
                self.model_to_test.model.layers[i].self_attn.config.merge = merge
        else:
            pass

    def load_model(self):
        model2path = json.load(open("/root/SnapKV/experiments/LongBench/config/model2path.json", "r"))
        
        if self.use_compress:
            sustainablekv_replace_llama()
            sustainablekv_replace_mistral()
            sustainablekv_replace_mixtral()

        self.model_to_test = AutoModelForCausalLM.from_pretrained(
            model2path[self.model_name],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            use_flash_attention_2=True
        ).eval()

        if "mistral" in self.model_name.lower():
            self.enc = AutoTokenizer.from_pretrained(
                model2path[self.model_name],
                padding_side="right",
                use_fast=False,
            )
        else:
            self.enc = AutoTokenizer.from_pretrained(
                model2path[self.model_name],
                use_fast=False,
            )
            
    def logistic(self, x, L=100, x0=50, k=.1):
        """
        When document_depth_percent_interval_type == 'sigmoid'
        """
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def bound_evaluate_and_log(self, *args):
        self.reset_kv_args()
        self.evaluate_and_log(*args)
    
    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len:
                continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)
    
    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
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

        input = self.enc(prompt, return_tensors="pt").to(self.model_to_test.device)
        context_length = input.input_ids.shape[-1]

        test_start_time = time.time()
        output = self.model_to_test.generate(
            **input,
            output_attentions=False,
            max_new_tokens=30,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            min_length=context_length+1,
            eos_token_id=[self.enc.eos_token_id, self.enc.encode("\n", add_special_tokens=False)[-1]],
        )[0]
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        response = self.enc.decode(output[context_length:], skip_special_tokens=True).strip()
        print(response)

        if len(response) != 0:
            score = scorer.score(self.needle, response)['rouge1'].fmeasure * 10
        else:
            score = 0.0
        
        results = {
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'), 
        }
        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'len_{context_length}_depth_{int(depth_percent*100)}'

        # if self.save_contexts:
        #     results['file_name'] = context_file_location

        #     # Save the context to file for retesting
        #     if not os.path.exists('sustainablekv_results_needle/contexts'):
        #         os.makedirs('sustainablekv_results_needle/contexts')

        #     if not os.path.exists(f'sustainablekv_results_needle/contexts/{self.model_name}'):
        #         os.makedirs(f'sustainablekv_results_needle/contexts/{self.model_name}')

        #     with open(f'sustainablekv_results_needle/contexts/{self.model_name}/{context_file_location}_context.txt', 'w') as f:
        #         f.write(context)
        
        if self.save_results:
            # Save the context to file for retesting
            # if not os.path.exists('sustainablekv_results_needle/results'):
            #     os.makedirs('sustainablekv_results_needle/results')
            
            # if not os.path.exists(f'sustainablekv_results_needle/results/{self.model_name}'):
            #     os.makedirs(f'sustainablekv_results_needle/results/{self.model_name}')

            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)

            # Save the result to file for retesting
            p = self.save_folder + os.sep + f'{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f, ensure_ascii=False)
        
    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """
        # results_dir = 'sustainablekv_results_needle/results/' + self.model_name
        results_dir = self.save_folder
        print("Searching existing results at %s" % results_dir)

        if not os.path.exists(results_dir):
            return False
        
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def encode_text_to_tokens(self, text):
        return self.enc.encode(text, add_special_tokens=False)
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]
        
        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if "mistral" in self.model_name.lower():
                period_tokens = [842, 28723]
            elif "llama2" in self.model_name.lower() or "llama-2" in self.model_name.lower():
                period_tokens = [13]
            elif "glm" in self.model_name.lower():
                period_tokens = [918, 30930]
            else:
                period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]
            
            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]
        
        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return len(self.enc.encode(context, add_special_tokens=False))
    
    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()

        return context
    
    def get_tokens_from_context(self, context):
        return self.enc.encode(context, add_special_tokens=False)
    
    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length], skip_special_tokens=True)
    
    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)

        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)

        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")
    
    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()

        #asyncio.run(self.run_test())
        self.run_test(args)
    
if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default=None, help='name of model', choices=[
        "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", 
        "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
        "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1", "llama-2-7B-32k-instruct", 
        "mixtral-8x7B-instruct-v0.1","lwm-text-chat-1m", "lwm-text-1m"])
    
    parser.add_argument('-s', '--s-len', metavar='N', type=int, default=1*1000, help='a number')
    parser.add_argument('-e', '--e-len', metavar='N', type=int, default=380*1000,help='a number')
    parser.add_argument('--num-intervals', type=int, default=34)
    parser.add_argument('--step', type=int, default=1000, help='Do not use this. Use num_intervals instead.')
    
    parser.add_argument("--compress", action="store_true")
    parser.add_argument('--save-folder', type=str, default=None, help='path to your output folder')
    
    args = parser.parse_args()

    needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    retrieval_question="The best thing to do in San Francisco is: ",

    ht = LLMNeedleHaystackTester(
        needle=needle,
        haystack_dir=r'/root/SnapKV/experiments/NeedleInHaystack/PaulGrahamEssays',
        retrieval_question=retrieval_question,

        context_lengths_min=args.s_len,
        context_lengths_max=args.e_len,
        context_lengths_num_intervals=args.num_intervals,
        step=args.step,

        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_intervals=10,
        document_depth_percent_interval_type='linear',

        model_name=args.model_name,
        results_version=2,
        final_context_length_buffer=200,

        save_contexts=False,
        save_results=True,
        print_ongoing_status=True,

        use_compress=args.compress,
        save_folder=args.save_folder,
    )

    ht.start_test(args)
        