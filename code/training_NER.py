import json
import logging
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils import data
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup

from architectures.BERT_model_Multitask import BERT_model_Multitask
from architectures.BERT_model import BERT_model
from architectures.architectures_utils import train_bert_model, predict, train_bert_model_multitask
from eval_metrics.evaluateNER import EvaluateNER
from utils.BatcherBERT import BatcherBERT
from utils.ReadNERData import Dataset
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt='%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments(parser):
	parser.add_argument("--file_extension", type=str, default="txt")
	parser.add_argument("--experiment", type=str, default="baseline")
	parser.add_argument("--saving_path", type=str, default="")
	parser.add_argument("--train_file", type=str, default="train")
	parser.add_argument("--data_path", type=str, default="train")
	parser.add_argument("--train", action="store_true")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--force_size", action="store_true")
	parser.add_argument("--special_labels", action="store_true")
	parser.add_argument("--masking_percentage", type=float, default=0.0)
	parser.add_argument("--seed", type=int, default=12)  # 12 seems to be a good seed
	parser.add_argument("--early_stop", type=int, default=0)
	parser.add_argument("--tags2use", type=str, default="NER_IOB2")
	parser.add_argument("--separator", type=str, default=" ")
	parser.add_argument("--print_predictions", action="store_true")
	parser.add_argument("--test_file", type=str, default="test")
	parser.add_argument("--dev_file", type=str, default="valid")
	parser.add_argument("--predict", type=str, default="None",
						choices=["None", "Train", "Test", "Dev", "All", "DevTest", "ExtraTest"])
	parser.add_argument("--crf", action="store_true")
	parser.add_argument("--update_masking", action="store_true")
	parser.add_argument("--predict_masked", action="store_true")
	parser.add_argument("--uppercase", action="store_true")
	parser.add_argument("--multitask", action="store_true")
	parser.add_argument("--lr", type=float, default=2e-5)
	parser.add_argument("--epsilon", type=float, default=1e-8)
	parser.add_argument("--bert_model", type=str, default="bert-base-cased")
	parser.add_argument("--token_col", type=int, default=0)
	parser.add_argument("--ner_col", type=int, default=3)
	parser.add_argument("--extra_test_file", type=str, default=None)
	parser.add_argument("--evaluate", action="store_true", help="It will be activated if training is activated")
	parser.add_argument("--mask_tokens", action="store_true",
						help="If activated, instead of masking based on entities, it will mask based on tokens.")
	parser.add_argument("--train_batch_size", type=int, default=32)
	parser.add_argument("--comment_line", type=str, default="-DOCSTART")
	parser.add_argument("--sequence_size", type=int, default=128)
	parser.add_argument("--multi_gpu", action="store_true")
	parser.add_argument("--subwords_mask", action="store_true")
	parser.add_argument("--no_redundant_uppercase", action="store_true")
	parser.add_argument("--bert_hidden_size", type=int, default=768)
	parser.add_argument("--no_dev", action="store_true")
	parser.add_argument("--uppercase_percentage", type=float, default=0.0)

	args = parser.parse_args()
	for k in args.__dict__:
		print(k + ": " + str(args.__dict__[k]))

	return args


parser = argparse.ArgumentParser()
opt = parse_arguments(parser)

seed = opt.seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

extension = opt.file_extension
columns = None
masking = False
update_masking = False
predict_masked = False
mask_entities = False
subwords_mask = False
do_predictions_on = []
if opt.predict != "None":
	if opt.predict == "All":
		do_predictions_on = ["Dev", "Test", "Train"]
		if opt.extra_test_file:
			do_predictions_on.append("ExtraTest")
	elif opt.predict == "DevTest":
		do_predictions_on = ["Dev", "Test"]
		if opt.extra_test_file:
			do_predictions_on.append("ExtraTest")
	else:
		do_predictions_on = [opt.predict]

if opt.train:
	opt.evaluate = True

multitask = False
multitask_dict = None
if opt.multitask:
	multitask = True
if opt.masking_percentage > 0:
	masking = True
	predict_masked = opt.predict_masked
	mask_entities = not opt.mask_tokens
	subwords_mask = opt.subwords_mask

columns = {
	opt.token_col: 'tokens',
	opt.ner_col: 'NER_IOB'
}
experiment_name = opt.experiment
tags2use = opt.tags2use
saving_path = opt.saving_path
train_file = opt.train_file
data_path = opt.data_path
if opt.separator == "\\t":
	opt.separator = "\t"

if opt.no_dev:
	opt.early_stop = 0

if not os.path.exists(f"{saving_path}/{experiment_name}/"):
	os.makedirs(f"{saving_path}/{experiment_name}/")
with open(f"{saving_path}/{experiment_name}/params-{experiment_name}.txt", "w") as output_file:
	output_file.write(f"seed: {seed}\n")
	output_file.write(f"file_extension: {extension}\n")
	output_file.write(f"experiment_name: {experiment_name}\n")
	output_file.write(f"tags2use: {tags2use}\n")
	output_file.write(f"special_labesl: {opt.special_labels}\n")
	output_file.write(f"force_size: {opt.force_size}\n")
	output_file.write(f"train_file: {train_file}\n")
	output_file.write(f"masking_percentage: {opt.masking_percentage}\n")
	output_file.write(f"seed: {seed}\n")
	output_file.write(f"epochs: {opt.epochs}\n")
	output_file.write(f"early_stop: {opt.early_stop}\n")
	output_file.write(f"separator: {opt.separator}\n")
	output_file.write(f"test_file: {opt.test_file}\n")
	output_file.write(f"dev_file: {opt.dev_file}\n")
	output_file.write(f"crf: {opt.crf}\n")
	output_file.write(f"update_masking: {update_masking}\n")
	output_file.write(f"uppercase: {opt.uppercase}\n")
	output_file.write(f"predict_masked: {predict_masked}\n")
	output_file.write(f"multitask: {multitask}\n")
	output_file.write(f"lr: {opt.lr}\n")
	output_file.write(f"epsilon: {opt.epsilon}\n")
	output_file.write(f"bert_model: {opt.bert_model}\n")
	output_file.write(f"extra_test_file: {opt.extra_test_file}\n")
	output_file.write(f"mask_entities: {mask_entities}\n")
	output_file.write(f"mask_tokens: {opt.mask_tokens}\n")
	output_file.write(f"train_batch_size: {opt.train_batch_size}\n")
	output_file.write(f"comment_line: {opt.comment_line}\n")
	output_file.write(f"sequence_size: {opt.sequence_size}\n")
	output_file.write(f"multi_gpu: {opt.multi_gpu}\n")
	output_file.write(f"subwords_mask: {opt.subwords_mask}\n")
	output_file.write(f"no_redundant_uppercase: {opt.no_redundant_uppercase}\n")
	output_file.write(f"bert_hidden_size: {opt.bert_hidden_size}\n")
	output_file.write(f"no_dev: {opt.no_dev}\n")
	output_file.write(f"uppercase_percentage: {opt.uppercase_percentage}\n")

dataset_config = {
	'columns': columns,
	'columnsSeparator': opt.separator,
	'basePath': data_path,
	'dataType': {
		'Train': {
			'path': f"{train_file}.{extension}",
			'labelsAsTraining': True
		},
		'Test': {
			'path': f"{opt.test_file}.{extension}",
			'labelsAsTraining': False
		}
	},
	'commentSymbol': [opt.comment_line],
	'specialLabels': opt.special_labels
}

if not opt.no_dev:
	dataset_config["dataType"]["Dev"] = {
		'path': f"{opt.dev_file}.{extension}",
		'labelsAsTraining': False
	}

if opt.extra_test_file is not None:
	dataset_config["dataType"]["ExtraTest"] = {
		'path': f"{opt.extra_test_file}.{extension}",
		'labelsAsTraining': False
	}

bert_base_model = opt.bert_model
tokenizer = BertTokenizer.from_pretrained(bert_base_model, do_lower_case=False)

data_processor = Dataset(dataset_config, tokenizer=tokenizer)
mapping = data_processor.getMapping()
dataset = data_processor.getProcessedData()



inverse_tagset = dict(map(reversed, mapping[tags2use].items()))



evaluateNER = EvaluateNER(inverse_tagset)

if multitask:
	if tags2use == "NER_IOB2":
		multitask_dict = mapping["NER_IOBA"]
	if tags2use == "NER_IOBES":
		multitask_dict = mapping["NER_IOBESA"]

resize_vocab_size = None

add_tokens = data_processor.getTokensToAdd()

if len(add_tokens) > 0:
	tokenizer.add_tokens(add_tokens)
	resize_vocab_size = len(tokenizer)

if opt.uppercase:
	special_tokens_dict = {'additional_special_tokens': ['[UP]', '[up]']}

	num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

	if num_added_toks > 0:
		resize_vocab_size = len(tokenizer)

tokenizer.save_vocabulary(f"{saving_path}/{experiment_name}/")


def Train():
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	type_crf_constraints = None
	if opt.crf:
		if tags2use in ["NER_IOB2", "NER_IOBA"]:
			type_crf_constraints = "BIO"
		elif tags2use in ["NER_IOBES", "NER_IOBESA"]:
			type_crf_constraints = "BIOES"

	bert_config = BertConfig.from_pretrained(bert_base_model, num_labels=len(mapping[tags2use]) + 1,
											 finetuning_task=experiment_name,
											 task_specific_params={"crf": opt.crf,
																   "type_crf_constraints": type_crf_constraints,
																   "predict_masked": predict_masked,
																   "labels_boundaries": multitask_dict,
																   "bert_hidden_size": opt.bert_hidden_size},
											 label2id=mapping[tags2use])

	params_train = {'batch_size': opt.train_batch_size,
					'shuffle': True}


	print("Processing train")
	training_batcher = BatcherBERT(dataset["Train"], tokenizer, tags_field=tags2use,
								   max_length=opt.sequence_size,
								   inverse_tagset=inverse_tagset,
								   tagset=mapping[tags2use], force_size=opt.force_size,
								   special_labels=opt.special_labels,
								   mask_percentage=opt.masking_percentage,
								   uppercase=opt.uppercase,
								   predict_masked=predict_masked, multitask=opt.multitask,
								   multitask_dict=multitask_dict,
								   mask_entities=mask_entities, subwords_mask=subwords_mask,
								   no_redundant_uppercase=opt.no_redundant_uppercase,
								   uppercase_percentge=opt.uppercase_percentage)

	training_batcher.createBatches()

	if opt.no_dev:
		dev_generator = None
		dev_aligner = None
	else:
		params_dev = {'batch_size': 8,
					  'shuffle': False}

		print("Processing dev")
		dev_set = BatcherBERT(dataset["Dev"], tokenizer, tags_field=tags2use,
							  max_length=opt.sequence_size, test=True, tagset=mapping[tags2use], force_size=opt.force_size,
							  special_labels=opt.special_labels, uppercase=opt.uppercase,
							  no_redundant_uppercase=opt.no_redundant_uppercase)

		dev_generator = data.DataLoader(dev_set, **params_dev, collate_fn=dev_set.collate_fn)

		dev_aligner = dev_set.getAligner()

	if multitask:
		model = BERT_model_Multitask.from_pretrained(bert_base_model,
													 from_tf=False,
													 config=bert_config)
	else:
		model = BERT_model.from_pretrained(bert_base_model,
										   from_tf=False,
										   config=bert_config)

	if resize_vocab_size is not None:
		model.resize_embedding_and_fc(resize_vocab_size)

	if opt.multi_gpu and torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs")
		model = nn.DataParallel(model)
		opt.multi_gpu = True
	else:
		opt.multi_gpu = False

	num_train_epochs = opt.epochs
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.weight']
	weight_decay = 0.01
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(
			nd in n for nd in no_decay)], 'weight_decay': weight_decay},
		{'params': [p for n, p in param_optimizer if any(
			nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]

	warmup_proportion = 0.1
	gradient_accumulation_steps = 1
	learning_rate = opt.lr
	num_train_optimization_steps = int(
		len(training_batcher) / params_train['batch_size'] / gradient_accumulation_steps) * num_train_epochs
	warmup_steps = int(warmup_proportion * num_train_optimization_steps)
	optimizer = AdamW(optimizer_grouped_parameters,
					  lr=learning_rate, eps=opt.epsilon,
					  correct_bias=True)  # If we would like to replicate BERT, we need to set the compensate_bias as false
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
												num_training_steps=num_train_optimization_steps)

	if multitask:
		train_bert_model_multitask(model, experiment_name, num_train_epochs,
								   optimizer, scheduler, training_batcher, params_train, dev_generator,
								   evaluateNER.calculate, saving_path, use_gpu=True, masking=masking, early_stop=opt.early_stop,
								   update_masking=update_masking, dev_aligner=dev_aligner, multi_gpu=opt.multi_gpu,
								   bert_hidden_size=opt.bert_hidden_size, uppercase_percentage=opt.uppercase_percentage)
	else:
		train_bert_model(model, experiment_name, num_train_epochs,
						 optimizer, scheduler, training_batcher, params_train, dev_generator, evaluateNER.calculate,
						 saving_path, use_gpu=True, masking=masking, early_stop=opt.early_stop, bert_hidden_size=opt.bert_hidden_size,
						 update_masking=update_masking, dev_aligner=dev_aligner, multi_gpu=opt.multi_gpu, uppercase_percentage=opt.uppercase_percentage)


def test(model, data_split):
	params_test = {'batch_size': 8,
				   'shuffle': False}

	print(f"Processing {data_split}")
	test_set = BatcherBERT(dataset[data_split], tokenizer, tags_field=tags2use,
						   max_length=opt.sequence_size, test=True, tagset=mapping[tags2use], force_size=opt.force_size,
						   special_labels=opt.special_labels, uppercase=opt.uppercase,
						   no_redundant_uppercase=opt.no_redundant_uppercase)

	test_generator = data.DataLoader(test_set, **params_test, collate_fn=test_set.collate_fn)

	print(f"Evaluating Model with {data_split}")
	_, _, report, david_metrics = predict(model, test_generator, tagged=True, evaluation_function=evaluateNER.calculate,
										  use_gpu=True, test_aligner=test_set.getAligner(), multi_gpu=opt.multi_gpu, bert_hidden_size=opt.bert_hidden_size)

	with open(f"{saving_path}/{experiment_name}/{data_split.lower()}-{experiment_name}-results.txt", "w") as output_file:
		output_file.write(report)
		output_file.write("\n")
	if david_metrics is not None:
		with open(f"{saving_path}/{experiment_name}/{data_split.lower()}-{experiment_name}-results_david.json",
				  "w") as output_file:
			json.dump(david_metrics, output_file)


def printCommentLines(output_file, lines_array):
	for line in lines_array:
		output_file.write(f"{line}\n")


def testAndPredict(model, iob2_inverse_tagset, data_split):
	params_test = {'batch_size': 8,
				   'shuffle': False}

	print(f"Processing {data_split}")
	test_set = BatcherBERT(dataset[data_split], tokenizer, tags_field=tags2use,
						   max_length=opt.sequence_size, test=True, tagset=mapping[tags2use], force_size=opt.force_size,
						   special_labels=opt.special_labels, uppercase=opt.uppercase,
						   no_redundant_uppercase=opt.no_redundant_uppercase)

	test_generator = data.DataLoader(test_set, **params_test, collate_fn=test_set.collate_fn)

	print(f"Evaluating Model with {data_split}")
	predictions, _, report, david_metrics = predict(model, test_generator, tagged=True,
													evaluation_function=evaluateNER.calculate, use_gpu=True,
													test_aligner=test_set.getAligner(), multi_gpu=opt.multi_gpu,
													bert_hidden_size=opt.bert_hidden_size)

	with open(f"{saving_path}/{experiment_name}/{data_split.lower()}-{experiment_name}-results.txt", "w") as output_file:
		output_file.write(report)
		output_file.write("\n")
	if david_metrics is not None:
		with open(f"{saving_path}/{experiment_name}/{data_split.lower()}-{experiment_name}-results_david.json",
				  "w") as output_file:
			json.dump(david_metrics, output_file)

	with open(f"{saving_path}/{experiment_name}/{data_split.lower()}-{experiment_name}-predictions.txt",
			  "w") as output_file:
		if -1 in dataset[f"{data_split}_comments"]:
			printCommentLines(output_file, dataset[f"{data_split}_comments"][-1])

		for sentence_id, sentence in enumerate(dataset[data_split]):
			assert (len(predictions[sentence_id]) == len(sentence["tokens"]))
			if iob2_inverse_tagset is None and tags2use is not None:
				for (token, predicted_tag, ner_tag) in zip(sentence["tokens"], predictions[sentence_id],
														   sentence[tags2use]):
					output_file.write(
						f"{token}{opt.separator}{predicted_tag}{opt.separator}{inverse_tagset[ner_tag]}\n")
			elif tags2use == "NER_IOBA":
				for (token, predicted_tag, ner_tag) in zip(sentence["tokens"], predictions[sentence_id],
														   sentence["NER_IOB2"]):
					output_file.write(
						f"{token}{opt.separator}{predicted_tag}{opt.separator}X{opt.separator}{iob2_inverse_tagset[ner_tag]}\n")
			elif tags2use == "NER_IOBESA":
				for (token, predicted_tag, ner_tag) in zip(sentence["tokens"], predictions[sentence_id],
														   sentence["NER_IOBES"]):
					output_file.write(
						f"{token}{opt.separator}{predicted_tag}{opt.separator}X{opt.separator}{iob2_inverse_tagset[ner_tag]}\n")
			else:
				for (token, predicted_tag, ner_tag) in zip(sentence["tokens"], predictions[sentence_id]):
					output_file.write(f"{token}{opt.separator}{predicted_tag}\n")
			output_file.write(f"\n")
			if sentence_id in dataset[f"{data_split}_comments"]:
				printCommentLines(output_file, dataset[f"{data_split}_comments"][sentence_id])


def getPredictions(model, iob2_inverse_tagset, data_split):
	params_test = {'batch_size': 8,
				   'shuffle': False}

	print(f"Processing {data_split}")
	test_set = BatcherBERT(dataset[data_split], tokenizer, tags_field=None, 
						   max_length=opt.sequence_size, test=True, tagset=None, force_size=opt.force_size,
						   special_labels=opt.special_labels, uppercase=opt.uppercase,
						   no_redundant_uppercase=opt.no_redundant_uppercase)

	test_generator = data.DataLoader(test_set, **params_test, collate_fn=test_set.collate_fn)

	predictions = predict(model, test_generator, tagged=False, evaluation_function=evaluateNER.calculate, use_gpu=True,
						  test_aligner=test_set.getAligner(), multi_gpu=opt.multi_gpu, bert_hidden_size=opt.bert_hidden_size)

	with open(f"{saving_path}/{experiment_name}/{data_split.lower()}-{experiment_name}-predictions.txt",
			  "w") as output_file:

		if -1 in dataset[f"{data_split}_comments"]:
			printCommentLines(output_file, dataset[f"{data_split}_comments"][-1])

		for sentence_id, sentence in enumerate(dataset[data_split]):
			assert (len(predictions[sentence_id]) == len(sentence["tokens"]))
			if iob2_inverse_tagset is None and tags2use is not None:
				for (token, predicted_tag, ner_tag) in zip(sentence["tokens"], predictions[sentence_id],
														   sentence[tags2use]):
					if predicted_tag not in inverse_tagset:
						predicted_tag = "O"
					else:
						predicted_tag = inverse_tagset[predicted_tag]
					output_file.write(
						f"{token}{opt.separator}{predicted_tag}{opt.separator}{inverse_tagset[ner_tag]}\n")
			elif tags2use == "NER_IOBA":
				for (token, predicted_tag, ner_tag) in zip(sentence["tokens"], predictions[sentence_id],
														   sentence["NER_IOB2"]):
					if predicted_tag not in inverse_tagset:
						predicted_tag = "O"
					else:
						predicted_tag = inverse_tagset[predicted_tag]
					output_file.write(
						f"{token}{opt.separator}{predicted_tag}{opt.separator}X{opt.separator}{iob2_inverse_tagset[ner_tag]}\n")
			elif tags2use == "NER_IOBESA":
				for (token, predicted_tag, ner_tag) in zip(sentence["tokens"], predictions[sentence_id],
														   sentence["NER_IOBES"]):
					if predicted_tag not in inverse_tagset:
						predicted_tag = "O"
					else:
						predicted_tag = inverse_tagset[predicted_tag]
					output_file.write(
						f"{token}{opt.separator}{predicted_tag}{opt.separator}X{opt.separator}{iob2_inverse_tagset[ner_tag]}\n")
			else:
				for (token, predicted_tag, ner_tag) in zip(sentence["tokens"], predictions[sentence_id]):
					if predicted_tag not in inverse_tagset:
						predicted_tag = "O"
					else:
						predicted_tag = inverse_tagset[predicted_tag]
					output_file.write(f"{token}{opt.separator}{predicted_tag}\n")
			output_file.write(f"\n")
			if sentence_id in dataset[f"{data_split}_comments"]:
				printCommentLines(output_file, dataset[f"{data_split}_comments"][sentence_id])


if opt.train:
	Train()

if opt.evaluate and len(do_predictions_on) > 0:
	if multitask:
		model = BERT_model_Multitask.from_pretrained(f"{saving_path}/{experiment_name}/", from_tf=False)
	else:
		model = BERT_model.from_pretrained(f"{saving_path}/{experiment_name}/", from_tf=False)

	if opt.multi_gpu and torch.cuda.device_count() > 1:
		print(f"Using {torch.cuda.device_count()} GPUs")
		model = nn.DataParallel(model)
		opt.multi_gpu = True
	else:
		opt.multi_gpu = False

	iob2_inverse_tagset = None
	if tags2use == "NER_IOBA":
		iob2_inverse_tagset = dict(map(reversed, mapping["NER_IOB2"].items()))
	elif tags2use == "NER_IOBESA":
		iob2_inverse_tagset = dict(map(reversed, mapping["NER_IOBES"].items()))
	for data_split in do_predictions_on:
		if data_split == "Train":
			continue
		print(f"Prediction on: {data_split}")
		testAndPredict(model, iob2_inverse_tagset, data_split)

else:
	if opt.evaluate:
		if multitask:
			model = BERT_model_Multitask.from_pretrained(f"{saving_path}/{experiment_name}/", from_tf=False)
		else:
			model = BERT_model.from_pretrained(f"{saving_path}/{experiment_name}/", from_tf=False)

		if opt.multi_gpu and torch.cuda.device_count() > 1:
			print(f"Using {torch.cuda.device_count()} GPUs")
			model = nn.DataParallel(model)
			opt.multi_gpu = True
		else:
			opt.multi_gpu = False

		test(model, "Test")
		if opt.extra_test_file is not None:
			test(model, "ExtraTest")

	if len(do_predictions_on) > 0:
		iob2_inverse_tagset = None
		if tags2use == "NER_IOBA":
			iob2_inverse_tagset = dict(map(reversed, mapping["NER_IOB2"].items()))
		elif tags2use == "NER_IOBESA":
			iob2_inverse_tagset = dict(map(reversed, mapping["NER_IOBES"].items()))
		if multitask:
			model = BERT_model_Multitask.from_pretrained(f"{saving_path}/{experiment_name}/", from_tf=False)
		else:
			model = BERT_model.from_pretrained(f"{saving_path}/{experiment_name}/", from_tf=False)

		if opt.multi_gpu and torch.cuda.device_count() > 1:
			print(f"Using {torch.cuda.device_count()} GPUs")
			model = nn.DataParallel(model)
			opt.multi_gpu = True
		else:
			opt.multi_gpu = False

		for data_split in do_predictions_on:
			print(f"Prediction on: {data_split}")
			getPredictions(model, iob2_inverse_tagset, data_split)
