from openprompt import PromptDataLoader
# from pipeline_base import PromptDataLoader
from openprompt.data_utils import PROCESSORS
from openprompt.prompts import ManualVerbalizer
from openprompt.prompts import SoftTemplate, MixedTemplate
from openprompt import PromptForClassification
from openprompt.utils.logging import logger
from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate
from typing import List, Dict, Callable, Sequence

from tokenizers import PreTokenizedString
from tqdm import tqdm
import torch
import argparse
import numpy as np
import time
import os
import re
import json
import random

from openprompt.utils.reproduciblity import set_seed
from openprompt.plms import load_plm
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5
from sklearn.metrics import classification_report #Classification Report

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.prompts import PTRTemplate
from discoverbalizer import DiscoVerbalizer
from data_utils import ANLIProcessor


parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", action="store_true", help="Whether to tune the plm, default to False")
# parser.add_argument("--few_shot", action="store_true", help="Whether to tune the plm, default to False")
parser.add_argument("--shot", type=int, default=-1)
# parser.add_argument("--few_shot_data",  type=str, default="./DiscoPrompt/PDTB2-Imp-Fewshot")

parser.add_argument("--model", type=str, default='t5', help="We test both t5 in this scripts, the corresponding tokenizerwrapper will be automatically loaded.")
parser.add_argument("--model_name_or_path", default='/model_path/t5-base')
parser.add_argument("--project_root", default='./AbductionPrompt', help="The project root in the file system")
parser.add_argument("--template_file", default="./AbductionPrompt/abduction_template.txt", help="The project root in the file system")                    
parser.add_argument("--template_id", type=int, default=0)
parser.add_argument("--verbalizer_id", type=int, default=0)

parser.add_argument("--setting",type=str, default= "anli" )
parser.add_argument("--result_file", type=str, default="./AbductionPrompt/result/results.txt")
parser.add_argument("--ckpt_file", type=str, default="AbductionPromptClassification_result")

parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--max_steps", default=30000, type=int)
parser.add_argument("--max_seq_length", default=350, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--model_lr", type=float, default=3e-5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--batch_size_e", type=int, default=8)
parser.add_argument("--eval_every_steps", type=int, default=250)

parser.add_argument("--dataset_decoder_max_length", default=50, type=int)
parser.add_argument("--decoder_max_length", default=10, type=int)
parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="Adafactor")
parser.add_argument("--warmup_step_prompt", type=int, default=0)
parser.add_argument("--init_from_vocab", action="store_false")
args = parser.parse_args()

content_write = "="*20+"\n"
# content_write += f"dataset {args.dataset}\t"
content_write += f"setting {args.setting}\t"
content_write += f"temp_file {args.template_file}\t"                 
content_write += f"temp {args.template_id}\t"
content_write += f"verb {args.verbalizer_id}\t"
content_write += f"model {args.model_name_or_path}\t"
content_write += f"seed {args.seed}\t"
content_write += f"tune_plm {args.tune_plm}\t"
# content_write += f"few_shot {args.few_shot}\t"
# content_write += f"shot {args.shot}\t"
content_write += f"max_steps {args.max_steps}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"prompt_lr {args.prompt_lr}\t"
content_write += f"batch_size {args.batch_size}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"ckpt_file {args.ckpt_file}\t"
content_write += "\n"
print(content_write)

if  args.setting == "anli":
    data_dir_train = "./data"
    data_dir_dev = "./data"
    data_dir_test = "./data"
    Processor = ANLIProcessor()
    
    dataset = {}
    dataset['train'] = Processor.get_examples(data_dir_train, 'train')
    dataset['validation'] = Processor.get_examples(data_dir_dev, 'dev')
    dataset['test'] = Processor.get_examples(data_dir_test, 'test')
    class_labels =Processor.get_labels()
    
    label_words = {
    0 :["choice1"],
    1 :["choice2"],
    }

elif args.setting == "anli-3":
    data_dir_train = "./ANLI_dataset_RankingRandom" #"./ANLI_dataset_RankingRandom" #"./anli_data"  # datas is orgial data, data is augumented
    data_dir_dev = "./ANLI_dataset_RankingRandom"
    data_dir_test = "./ANLI_dataset_RankingRandom"
    Processor = ANLIProcessor()
    
    dataset = {}
    dataset['train'] = Processor.get_examples(data_dir_train, 'train')
    dataset['validation'] = Processor.get_examples(data_dir_dev, 'dev')
    dataset['test'] = Processor.get_examples(data_dir_test, 'test')
    class_labels =Processor.get_labels()
    
    label_words = {
    0 :["plausible", "not plausible", "choice1"],
    1 :["not plausible", "plausible", "choice2"],
    }

elif args.setting == "anli-4":
    data_dir_train = "./AbductionPrompt/data"  # datas is orgial data, data is augumented
    data_dir_dev = "./AbductionPrompt/data"
    data_dir_test = "./AbductionPrompt/data"
    Processor = ANLIProcessor()
    
    dataset = {}
    dataset['train'] = Processor.get_examples(data_dir_train, 'train')
    dataset['validation'] = Processor.get_examples(data_dir_dev, 'dev')
    dataset['test'] = Processor.get_examples(data_dir_test, 'test')
    class_labels =Processor.get_labels()
    
    label_words = {
    0 :["A"],
    1 :["B"],
    }

elif args.setting == "anli-5":
    data_dir_train = "./AbductionPrompt/data"  # datas is orgial data, data is augumented
    data_dir_dev = "./AbductionPrompt/data"
    data_dir_test = "./AbductionPrompt/data"
    Processor = ANLIProcessor()
    
    dataset = {}
    dataset['train'] = Processor.get_examples(data_dir_train, 'train')
    dataset['validation'] = Processor.get_examples(data_dir_dev, 'dev')
    dataset['test'] = Processor.get_examples(data_dir_test, 'test')
    class_labels =Processor.get_labels()
    
    label_words = {
    0 :["plausible", "not plausible", "A"],
    1 :["not plausible", "plausible", "B"],
    }

set_seed(args.seed)   
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

max_seq_l = args.max_seq_length
dataset_decoder_max_length = args.dataset_decoder_max_length
decoder_max_length = args.decoder_max_length
batchsize_t = args.batch_size
batchsize_e = args.batch_size_e
gradient_accumulation_steps = args.gradient_accumulation_steps
model_parallelize = True

template_id = args.template_id
mytemplate = PTRTemplate(model=plm, tokenizer=tokenizer).from_file("./abduction_template.txt", choice=template_id)
myverbalizer = DiscoVerbalizer(tokenizer, classes=class_labels, num_classes=2, label_words = label_words)
print(mytemplate.text)

use_cuda = True
tune_plm = args.tune_plm #True
plm_eval_mode = True

if tune_plm == False:
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=(not tune_plm), plm_eval_mode=plm_eval_mode)
elif tune_plm == True:
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=(not tune_plm)) # plm_eval_mode = False

if use_cuda:
    prompt_model= prompt_model.cuda()#.half()

if model_parallelize:
    prompt_model.parallelize()

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=decoder_max_length,
    batch_size=batchsize_t,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail") #, num_workers =16

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=decoder_max_length,
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=decoder_max_length,
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
print("truncate rate: {}".format(train_dataloader.tokenizer_wrapper.truncate_rate), flush=True)
print("truncate rate: {}".format(validation_dataloader.tokenizer_wrapper.truncate_rate), flush=True)
print("truncate rate: {}".format(test_dataloader.tokenizer_wrapper.truncate_rate), flush=True)

generation_arguments = {
    "max_length": dataset_decoder_max_length,
}

def evaluate(prompt_model, dataloader, test = False):
    prompt_model.eval()
    predictions = []
    ground_truths = []

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.to(prompt_model.device)
        logits, _, each_logits = prompt_model(inputs)
        predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        ground_truths.extend(inputs['label'])
    assert len(predictions)==len(ground_truths), (len(predictions), len(ground_truths))
#     predictions = [prediction.strip() for prediction in predictions]
#     ground_truths = [ground_truth.strip() for ground_truth in ground_truths]
    predictions = [int(prediction) for prediction in predictions]
    ground_truths = [int(ground_truth) for ground_truth in ground_truths]
    # shown one example
    print(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")
    score = crossfit_evaluate(predictions, ground_truths, metric="ACC")
#     score = crossfit_evaluate(predictions, ground_truths, metric="Classification-F1")
    
    if test:
        print("4 Class Classification Report")
        print(classification_report(ground_truths, predictions, digits = 4))    
    return score


max_steps = args.max_steps
loss_func = torch.nn.CrossEntropyLoss()
tot_step = max_steps

optimizer = "Adafactor"
prompt_lr = args.prompt_lr
print(prompt_lr)
warmup_step_prompt = args.warmup_step_prompt

if tune_plm: # normally we freeze the model when using soft_template. However, we keep the option to tune plm
    no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=500, num_training_steps=tot_step)
else:
    optimizer1 = None
    scheduler1 = None


optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
if optimizer.lower() == "adafactor":
    optimizer2 = Adafactor(optimizer_grouped_parameters2,
                            lr=prompt_lr,
                            relative_step=False,
                            scale_parameter=False,
                            warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    scheduler2 = get_constant_schedule_with_warmup(optimizer2, num_warmup_steps=warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
elif optimizer.lower() == "adamw":
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=prompt_lr) # usually lr = 0.5
    scheduler2 = get_linear_schedule_with_warmup(
                    optimizer2,
                    num_warmup_steps=warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500


print(mytemplate.get_default_soft_token_ids())
print(optimizer_grouped_parameters2[0]['params'][0].size())
print("num of trainable parameters: %d" % ( sum(p.numel() for p in prompt_model.parameters() if p.requires_grad)))

eval_every_steps = args.eval_every_steps
project_root = args.project_root
this_run_unicode = args.ckpt_file
result_file = args.result_file

tot_loss = 0
log_loss = 0
best_val_acc = 0
glb_step = 0
actual_step = 0
leave_training = False

acc_traces = []
tot_train_time = 0
pbar_update_freq = 10
prompt_model.train()

pbar = tqdm(total=tot_step, desc="Train")
for epoch in range(1000000):
    print(f"Begin epoch {epoch}")
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.to(prompt_model.device)
        tot_train_time -= time.time()
        
        # For Classification Param
        logits, _, each_logits = prompt_model(inputs) 
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        actual_step += 1        

        if actual_step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss) / pbar_update_freq
                pbar.update(10)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss

            if optimizer1 is not None:
                optimizer1.step()
                optimizer1.zero_grad()
            if scheduler1 is not None:
                scheduler1.step()
            if optimizer2 is not None:
                optimizer2.step()
                optimizer2.zero_grad()
            if scheduler2 is not None:
                scheduler2.step()

        tot_train_time += time.time()

        if actual_step % gradient_accumulation_steps == 0 and glb_step > 0 and glb_step % eval_every_steps == 0:
            val_acc = evaluate(prompt_model, validation_dataloader)
            if val_acc >= best_val_acc:
                # torch.save(prompt_model.state_dict(), f"{project_root}/ckpts/{this_run_unicode}.ckpt")
                torch.save(prompt_model.state_dict(), f"/home/data/ckchancc/AbductionPrompt/ckpt/{this_run_unicode}.ckpt")

                best_val_acc = val_acc
            acc_traces.append(val_acc)
            print(
            "Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time / actual_step), flush
            =True)
            prompt_model.train()

        if glb_step > max_steps:
            leave_training = True
            break

    if leave_training:
        break


# tokenizer.save_vocabulary("{project_root}/")
# prompt_model.config.to_json_file("{project_root}/")

# # super_glue test split can not be evaluated without submitting the results to their website. So we skip it here and keep them as comments.
prompt_model.load_state_dict(torch.load(f"/home/data/ckchancc/AbductionPrompt/ckpt/{this_run_unicode}.ckpt"))
prompt_model = prompt_model.cuda()#.half()

# a simple measure for the convergence speed.
thres99 = 0.99 * best_val_acc
thres98 = 0.98 * best_val_acc
thres100 = best_val_acc
step100 = step98 = step99 = max_steps
for val_time, acc in enumerate(acc_traces):
    if acc >= thres98:
        step98 = min(val_time * eval_every_steps, step98)
        if acc >= thres99:
            step99 = min(val_time * eval_every_steps, step99)
            if acc >= thres100:
                step100 = min(val_time * eval_every_steps, step100)

content_write = " "
content_write += f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\tcritical_steps:{[step98, step99, step100]}\n"
content_write += "\n"

print(content_write)


test_acc = evaluate(prompt_model, test_dataloader, test = True)
print("Testing Accuacry")
print(test_acc)

with open(f"{result_file}", "a") as fout:
    fout.write(content_write)
