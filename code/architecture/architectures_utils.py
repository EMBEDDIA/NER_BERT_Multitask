import torch
from tqdm import tqdm
from torch.utils import data
import os

from utils.EarlyStopper import EarlyStopper


def train_bert_model_multitask(model, experiment_name, epochs, optimizer, scheduler, train_batcher,
							   train_batching_params, dev_dataloader,
							   evaluation_function, saving_path, early_stop=0, use_gpu=True, gpu_device="cuda:0",
							   masking=False, update_masking=False, dev_aligner=None, multi_gpu=False, bert_hidden_size=768, uppercase_percentage=0.0):
	using_cuda = False
	if torch.cuda.is_available() and use_gpu:
		device = torch.device(gpu_device)
		model.to(device)
		using_cuda = True
	else:
		device = torch.device("cpu")

	gradient_accumulation_steps = 1
	step = 0
	eval_score = 0.0
	report = ""
	train_dataloader = data.DataLoader(train_batcher, **train_batching_params, collate_fn=train_batcher.collate_fn)

	early_stopper = EarlyStopper(patience=early_stop)

	for epoch in range(epochs):

		if epoch > 0 and (masking is True or uppercase_percentage > 0.0):
			if update_masking and eval_score >= 0.90:
				print("Updating masking")
				train_batcher.updateMasking(None, False)
				update_masking = False
				early_stop = 5
			print("Batching training dataset")
			train_batcher.createBatches()
			train_dataloader = data.DataLoader(train_batcher, **train_batching_params,
											   collate_fn=train_batcher.collate_fn)

		print(f"\nTraining Model: {epoch + 1}")
		running_loss = 0.0
		model.train()
		for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
			tokens, attention_masks, token_type_ids, tags, tokens_mask, labelling_mask, lm_mask, lm_labels, labels_boundaries = batch
			batch_size, sequence_size = tokens.shape
			valid_output_tonkens = torch.zeros(batch_size, sequence_size, bert_hidden_size, dtype=torch.float32, device=device)
			valid_output_predict = None
			if lm_mask is not None:
				valid_output_predict = torch.zeros(batch_size, sequence_size, bert_hidden_size, dtype=torch.float32, device=device)
			if using_cuda:
				tokens = tokens.to(device)
				tags = tags.to(device)
				attention_masks = attention_masks.to(device)
				labelling_mask = labelling_mask.to(device)
				tokens_mask = tokens_mask.to(device)
				token_type_ids = token_type_ids.to(device)
				labels_boundaries = labels_boundaries.to(device)
				if lm_mask is not None:
					lm_mask = lm_mask.to(device)
					lm_labels = lm_labels.to(device)

			loss, _ = model(tokens, token_type_ids=token_type_ids, attention_mask=attention_masks, labels=tags,
							tokens_mask=tokens_mask, labelling_mask=labelling_mask, lm_mask=lm_mask,
							lm_labels=lm_labels,
							labels_boundaries=labels_boundaries, valid_output_tokens=valid_output_tonkens,
							valid_output_predict=valid_output_predict)
			if multi_gpu:
				loss.sum().backward()
				running_loss += loss.sum().item()
			else:
				loss.backward()
				running_loss += loss.item()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			if (step + 1) % gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
		if step == 0:
			step = 1
		print(f"Loss:\t{running_loss / step}")

		if dev_dataloader is not None:
			print("Evaluating Model with Dev")

			model.eval()

			eval_score, report, loss = predict(model, dev_dataloader, tagged=True, evaluation_function=evaluation_function,
											   calculate_loss=True, multi_gpu=multi_gpu, bert_hidden_size=bert_hidden_size,
											   use_gpu=use_gpu, gpu_device=gpu_device, test_aligner=dev_aligner)

			print(f"F-score: {eval_score}\tLoss: {loss}")
			if saving_path is not None and early_stop > 0:
				if early_stopper.checkImprovement(loss, eval_score):
					print(early_stopper.getCounter())
					if not os.path.exists(f"{saving_path}/{experiment_name}/"):
						os.makedirs(f"{saving_path}/{experiment_name}/")
					if multi_gpu:
						model.module.save_pretrained(f"{saving_path}/{experiment_name}/")
					else:
						model.save_pretrained(f"{saving_path}/{experiment_name}/")
					with open(f"{saving_path}/{experiment_name}/dev-{experiment_name}-results.txt", "w") as output_file:
						output_file.write(report)
						output_file.write("\n")
					with open(f"{saving_path}/{experiment_name}/best-{experiment_name}-epoch.txt", "w") as output_file:
						output_file.write(f"best: {epoch + 1}\n")
				else:
					print(early_stopper.getCounter())
					if early_stopper.stopTraining():
						with open(f"{saving_path}/{experiment_name}/best-{experiment_name}-epoch.txt", "a") as output_file:
							output_file.write(f"last: {epoch + 1}\n")
							output_file.write(f"reason: {early_stopper.getCounter()}\n")
						print(f"Early stop as there have been {early_stop} epochs withouth change")
						break
	if early_stop == 0:
		if not os.path.exists(f"{saving_path}/{experiment_name}/"):
			os.makedirs(f"{saving_path}/{experiment_name}/")
		if multi_gpu:
			model.module.save_pretrained(f"{saving_path}/{experiment_name}/")
		else:
			model.save_pretrained(f"{saving_path}/{experiment_name}/")
		if dev_dataloader is not None:
			with open(f"{saving_path}/{experiment_name}/dev-{experiment_name}-results.txt", "w") as output_file:
				output_file.write(report)
				output_file.write("\n")


def train_bert_model(model, experiment_name, epochs, optimizer, scheduler, train_batcher, train_batching_params,
					 dev_dataloader,
					 evaluation_function, saving_path, early_stop=0, use_gpu=True, gpu_device="cuda:0", masking=False,
					 update_masking=False, dev_aligner=None, multi_gpu=False, bert_hidden_size=768, uppercase_percentage=0.0):
	using_cuda = False
	if torch.cuda.is_available() and use_gpu:
		device = torch.device(gpu_device)
		model.to(device)
		using_cuda = True
	else:
		device = torch.device("cpu")

	gradient_accumulation_steps = 1
	step = 0
	eval_score = 0.0
	report = ""
	train_dataloader = data.DataLoader(train_batcher, **train_batching_params, collate_fn=train_batcher.collate_fn)

	early_stopper = EarlyStopper(patience=early_stop)

	for epoch in range(epochs):

		if epoch > 0 and (masking is True or uppercase_percentage > 0.0):
			if update_masking and eval_score >= 0.90:
				print("Updating masking")
				train_batcher.updateMasking(None, False)
				update_masking = False
				early_stop = 5
			print("Batching training dataset")
			train_batcher.createBatches()
			train_dataloader = data.DataLoader(train_batcher, **train_batching_params,
											   collate_fn=train_batcher.collate_fn)

		print(f"\nTraining Model: {epoch + 1}")
		running_loss = 0.0
		model.train()
		for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
			tokens, attention_masks, token_type_ids, tags, tokens_mask, labelling_mask, lm_mask, lm_labels = batch
			batch_size, sequence_size = tokens.shape
			valid_output_tonkens = torch.zeros(batch_size, sequence_size, bert_hidden_size, dtype=torch.float32, device=device)
			valid_output_predict = None
			if lm_mask is not None:
				valid_output_predict = torch.zeros(batch_size, sequence_size, bert_hidden_size, dtype=torch.float32, device=device)
			if using_cuda:
				tokens = tokens.to(device)
				tags = tags.to(device)
				attention_masks = attention_masks.to(device)
				labelling_mask = labelling_mask.to(device)
				tokens_mask = tokens_mask.to(device)
				token_type_ids = token_type_ids.to(device)
				if lm_mask is not None:
					lm_mask = lm_mask.to(device)
					lm_labels = lm_labels.to(device)

			loss, _ = model(tokens, token_type_ids=token_type_ids, attention_mask=attention_masks, labels=tags,
							tokens_mask=tokens_mask, labelling_mask=labelling_mask, lm_mask=lm_mask,
							lm_labels=lm_labels, valid_output_tokens=valid_output_tonkens,
							valid_output_predict=valid_output_predict)

			if multi_gpu:
				loss.sum().backward()
				running_loss += loss.sum().item()
			else:
				loss.backward()
				running_loss += loss.item()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			if (step + 1) % gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
		step += 1
		print(f"Loss:\t{running_loss / step}")

		if dev_dataloader is not None:
			print("Evaluating Model with Dev")
			model.eval()

			eval_score, report, loss = predict(model, dev_dataloader, tagged=True, evaluation_function=evaluation_function,
											   calculate_loss=True, multi_gpu=multi_gpu, bert_hidden_size=bert_hidden_size,
											   use_gpu=use_gpu, gpu_device=gpu_device, test_aligner=dev_aligner)
			print(f"F-score: {eval_score}\tLoss: {loss}")
			if saving_path is not None and early_stop > 0:
				if early_stopper.checkImprovement(loss, eval_score):
					print(early_stopper.getCounter())
					if not os.path.exists(f"{saving_path}/{experiment_name}/"):
						os.makedirs(f"{saving_path}/{experiment_name}/")
					if multi_gpu:
						model.module.save_pretrained(f"{saving_path}/{experiment_name}/")
					else:
						model.save_pretrained(f"{saving_path}/{experiment_name}/")
					with open(f"{saving_path}/{experiment_name}/dev-{experiment_name}-results.txt", "w") as output_file:
						output_file.write(report)
						output_file.write("\n")
					with open(f"{saving_path}/{experiment_name}/best-{experiment_name}-epoch.txt", "w") as output_file:
						output_file.write(f"best: {epoch + 1}\n")
				else:
					print(early_stopper.getCounter())
					if early_stopper.stopTraining():
						with open(f"{saving_path}/{experiment_name}/best-{experiment_name}-epoch.txt", "a") as output_file:
							output_file.write(f"last: {epoch + 1}\n")
							output_file.write(f"reason: {early_stopper.getCounter()}\n")
						print(f"Early stop as there have been {early_stop} epochs withouth change")
						break
	if early_stop == 0:
		if not os.path.exists(f"{saving_path}/{experiment_name}/"):
			os.makedirs(f"{saving_path}/{experiment_name}/")
		if multi_gpu:
			model.module.save_pretrained(f"{saving_path}/{experiment_name}/")
		else:
			model.save_pretrained(f"{saving_path}/{experiment_name}/")
		if dev_dataloader is not None:
			with open(f"{saving_path}/{experiment_name}/dev-{experiment_name}-results.txt", "w") as output_file:
				output_file.write(report)
				output_file.write("\n")


def predict(model, test_dataloader, tagged=False, evaluation_function=None, use_gpu=False, calculate_loss=False,
			gpu_device="cuda:0", test_aligner=None, multi_gpu=False, bert_hidden_size=768):
	predictions = []
	gold_standard = []
	using_cuda = False
	if torch.cuda.is_available() and use_gpu:
		device = torch.device(gpu_device)
		model.to(device)
		using_cuda = True
	else:
		device = torch.device("cpu")
	model.eval()
	if multi_gpu:
		hasCRF = model.module.hasCRF()
	else:
		hasCRF = model.hasCRF()
	for step, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):

		tokens, attention_masks, token_type_ids, tags, tokens_mask, labelling_mask = batch
		batch_size, sequence_size = tokens.shape
		valid_output_tonkens = torch.zeros(batch_size, sequence_size, bert_hidden_size, dtype=torch.float32, device=device)
		if using_cuda:
			tokens = tokens.to(device)
			attention_masks = attention_masks.to(device)
			if calculate_loss:
				labelling_mask = labelling_mask.to(device)
				tags = tags.to(device)
			tokens_mask = tokens_mask.to(device)
			token_type_ids = token_type_ids.to(device)

		with torch.no_grad():
			if calculate_loss:
				loss, logits = model(tokens, token_type_ids=token_type_ids, attention_mask=attention_masks,
									 labels=tags, tokens_mask=tokens_mask, labelling_mask=labelling_mask,
									 valid_output_tokens=valid_output_tonkens)
				if multi_gpu:
					loss = loss.sum()
			else:
				logits = model(tokens, token_type_ids=token_type_ids, attention_mask=attention_masks,
							   labels=None, tokens_mask=tokens_mask, labelling_mask=None,
							   valid_output_tokens=valid_output_tonkens)

		if hasCRF:
			if multi_gpu:
				logits = model.module.getCRFtags(logits, labelling_mask.to(device))
			else:
				logits = model.getCRFtags(logits, labelling_mask.to(device))
			predictions.extend(logits)
		else:
			logits = torch.argmax(torch.log_softmax(logits, dim=2), dim=2)
			logits = logits.detach().cpu().numpy()

		if tagged:
			if using_cuda and calculate_loss:
				labelling_mask = labelling_mask.detach().cpu()
				tags = tags.detach().cpu()
			for i in range(len(tags)):
				active_tokens = labelling_mask[i] == 1
				active_tags = ((tags[i])[active_tokens])
				gold_standard.append(active_tags.tolist())
				if not hasCRF:
					active_logits = ((logits[i])[active_tokens])
					predictions.append(active_logits.tolist())
		elif not hasCRF:
			for i in range(len(logits)):
				active_tokens = labelling_mask[i] == 1
				active_logits = ((logits[i])[active_tokens])
				predictions.append(active_logits.tolist())
	if test_aligner is not None:
		predictions, gold_standard = sentenceAligner(test_aligner, predictions, gold_standard)
	if tagged and evaluation_function is not None:
		eval_score, report, david_metrics = evaluation_function(predictions, gold_standard)
		if calculate_loss:
			return eval_score, report, loss / (step + 1)
		else:
			return predictions, eval_score, report, david_metrics
	return predictions


def sentenceAligner(aligner, predictions, gold_standard):
	sentence_offset = 0
	final_predictions = []
	final_gold_standard = []
	print("Aligning predictions")
	for sentence_id, sentence in enumerate(tqdm(aligner, total=len(aligner))):
		sentence_prediction = []
		sentence_gold = []
		while len(sentence_prediction) < aligner[sentence_id]:
			sentence_prediction.extend(predictions[sentence_id + sentence_offset])
			if len(gold_standard) > 0:
				sentence_gold.extend(gold_standard[sentence_id + sentence_offset])
			if len(sentence_prediction) < aligner[sentence_id]:
				sentence_offset += 1
		assert (len(sentence_prediction) == aligner[sentence_id])
		if len(gold_standard) > 0:
			assert (len(sentence_gold) == aligner[sentence_id])
			final_gold_standard.append(sentence_gold)
		final_predictions.append(sentence_prediction)
	predictions = final_predictions
	if len(gold_standard) > 0:
		gold_standard = final_gold_standard
	return predictions, gold_standard
