import math
import random

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils import data
import torch


# class BatcherBERT(data.Dataset):
from tqdm import tqdm


class BatcherBERT(data.Dataset):
    def __init__(self, dataset, bert_tokenizer, tags_field="tags", max_length=None, tagset=None, inverse_tagset=None,
                 special_labels=False, force_size=None, test=False, mask_percentage=0.0, uppercase=False, predict_masked=False,
                 multitask=False, multitask_dict=None, create_aligner=True, mask_entities=False, subwords_mask=False, no_redundant_uppercase=False, uppercase_percentge=0.0):
        self.__bert_tokenizer = bert_tokenizer
        self.__max_length = max_length
        self.__tags_field = tags_field
        self.__special_labels = special_labels
        self.__tagset = tagset
        self.__inverse_tagset = inverse_tagset
        self.__mask_percentage = 0.0
        self.__predict_masked = False
        self.__uppercase = uppercase
        self.__uppercase_percentage = uppercase_percentge
        self.__no_redundant_uppercase = no_redundant_uppercase
        self.__multitask = False
        self.__multitask_dict = multitask_dict
        self.__force_size = force_size
        self.__test = test
        self.__create_aligner = False
        self.__aligner = []
        self.__masked_entities = False
        self.__mask_all_subtokens = False
        if self.__test:
            self.__create_aligner = create_aligner
            self.__data = self.__prepareTokens(dataset)
            self.__size = len(self.__data)
            self.__entries_ids = list(range(self.__size))
        else:
            self.__multitask = multitask
            self.__multitask_convertor = self.__createConvertorMultitask()
            self.__mask_percentage = mask_percentage
            if self.__mask_percentage > 0.0:
                self.__predict_masked = predict_masked
                self.__mask_entities = mask_entities
                self.__subwords_mask = subwords_mask
            self.__dataset = dataset
            self.__data = None
            self.__size = 0
            self.__entries_ids = []

    def __createConvertorMultitask(self):
        if not self.__multitask:
            return None
        conversion_dict = {}
        for tag in self.__tagset:
            if tag == "O":
                conversion_dict[self.__tagset["O"]] = self.__multitask_dict["O"]
            elif tag[0] == "B":
                conversion_dict[self.__tagset[tag]] = self.__multitask_dict["B-A"]
            elif tag[0] == "I":
                conversion_dict[self.__tagset[tag]] = self.__multitask_dict["I-A"]
            elif tag[0] == "S":
                conversion_dict[self.__tagset[tag]] = self.__multitask_dict["S-A"]
            elif tag[0] == "E":
                conversion_dict[self.__tagset[tag]] = self.__multitask_dict["E-A"]
            else:
                conversion_dict[self.__tagset[tag]] = self.__multitask_dict[tag]
        return conversion_dict

    def getAligner(self):
        if self.__create_aligner:
            return self.__aligner
        else:
            return None

    def updateMasking(self, mask_percentage):
        if mask_percentage is not None:
            self.__mask_percentage = mask_percentage

    def createBatches(self):
        self.__data = self.__prepareTokens(self.__dataset)
        self.__size = len(self.__data)
        self.__entries_ids = list(range(self.__size))

    def __len__(self):
        return self.__size

    def __entryGenerator(self):
        bert_tokens_list = []
        new_entry = {}
        # This will be the CLS token
        if not self.__special_labels:
            new_entry["bert_tokens_mask"] = [0]
            new_entry["labelling_mask"] = []
            if self.__tags_field is not None:
                new_entry["bert_tags"] = []
            if self.__predict_masked:
                new_entry["lm_mask"] = []
                new_entry["lm_labels"] = []
        else:
            new_entry["bert_tokens_mask"] = [1]
            #We don't want to predict this token during testing
            if self.__test:
                new_entry["labelling_mask"] = [0]
            else:
                new_entry["labelling_mask"] = [1]
            if self.__tags_field is not None:
                new_entry["bert_tags"] = [self.__tagset["[CLS]"]]   # During testing this will disappear with the labelling mask
            if self.__predict_masked:
                new_entry["lm_mask"] = [0]
                new_entry["lm_labels"] = []
        new_entry["token_type_ids"] = [0]  # This will be the CLS token
        return bert_tokens_list, new_entry

    def __processEntry(self, bert_tokens_list, new_entry, overflow=-1):
        encoding = self.__bert_tokenizer.encode_plus(bert_tokens_list,
                                                     max_length=self.__max_length,
                                                     pad_to_max_length=True,
                                                     return_attention_masks=True,
                                                     add_special_tokens=True,
                                                     )
        new_entry["bert_tokens"] = encoding["input_ids"]
        new_entry["attention_mask"] = encoding["attention_mask"]

        # This is the SEP
        if self.__special_labels:
            new_entry["bert_tokens_mask"].append(1)

            if overflow == -1:
                overflow = len(new_entry["labelling_mask"])

            if self.__tags_field is not None:
                new_entry["bert_tags"].insert(overflow, self.__tagset["[SEP]"])

            if self.__test:
                new_entry["labelling_mask"].insert(overflow, 0)
            else:
                new_entry["labelling_mask"].insert(overflow, 1)

        if self.__multitask:
            new_entry["multitask_tags"] = []
            for tag in new_entry["bert_tags"]:
                if tag == 0:
                    new_entry["multitask_tags"].append(0)
                else:
                    new_entry["multitask_tags"].append(self.__multitask_convertor[tag])

        if self.__tags_field is not None and self.__max_length is not None:
            for i in range(len(new_entry["bert_tags"]), self.__max_length):
                new_entry["bert_tags"].append(0)
            assert (len(new_entry["bert_tags"]) == self.__max_length)

        if self.__max_length is not None:
            for i in range(len(new_entry["bert_tokens_mask"]), self.__max_length):
                new_entry["bert_tokens_mask"].append(0)
        assert (len(new_entry["bert_tokens_mask"]) == self.__max_length)

        if self.__max_length is not None:
            for i in range(len(new_entry["labelling_mask"]), self.__max_length):
                new_entry["labelling_mask"].append(0)
        assert (len(new_entry["labelling_mask"]) == self.__max_length)

        assert (len(new_entry["labelling_mask"]) == len(new_entry["bert_tokens_mask"]))

        if self.__max_length is not None:
            for i in range(len(new_entry["token_type_ids"]), self.__max_length):
                new_entry["token_type_ids"].append(0)

        if self.__predict_masked and self.__max_length is not None:
            for i in range(len(new_entry["lm_mask"]), self.__max_length):
                new_entry["lm_mask"].append(0)
            assert (len(new_entry["lm_mask"]) == self.__max_length)
            for i in range(len(new_entry["lm_labels"]), self.__max_length):
                new_entry["lm_labels"].append(0)
            assert (len(new_entry["lm_labels"]) == self.__max_length)

        if self.__multitask:
            for i in range(len(new_entry["multitask_tags"]), self.__max_length):
                new_entry["multitask_tags"].append(0)
            assert (len(new_entry["multitask_tags"]) == self.__max_length)

        assert (len(new_entry["token_type_ids"]) == self.__max_length)

        assert (len(new_entry["token_type_ids"]) == len(new_entry["bert_tokens_mask"]))

        #assert (sum(new_entry["bert_tokens_mask"]) == sum(new_entry["labelling_mask"]))


        # if __debug__:
        #     print("*** Example ***")
        #     print(f"tokens: {len(bert_tokens_list)}")
        #     print("tokens: %s" % " ".join(
        #         [str(x) for x in bert_tokens_list]))
        #     print(f"input_ids: {len(new_entry['bert_tokens'])}")
        #     print("input_ids: %s" %
        #           " ".join([str(x) for x in new_entry["bert_tokens"]]))
        #     print(f"bert_tags_masks: {len(new_entry['bert_tags_masks'])}")
        #     print("bert_tags_masks: %s" %
        #           " ".join([str(x) for x in new_entry["bert_tags_masks"]]))
        #     print(f"label_masks: {len(new_entry['label_masks'])}")
        #     print(
        #         "label_masks: %s" % " ".join([str(x) for x in new_entry["label_masks"]]))
        #     print(f"attention_masks: {len(new_entry['attention_masks'])}")
        #     print(
        #         "attention_masks: %s" % " ".join([str(x) for x in new_entry["attention_masks"]]))
        #     print(f"bert_tags: {len(new_entry['bert_tags'])}")
        #     print(
        #         "bert_tags: %s" % " ".join([str(x) for x in new_entry["bert_tags"]]))
        #     print(f"token_type_ids: {len(new_entry['token_type_ids'])}")
        #     print(
        #         "token_type_ids: %s" % " ".join([str(x) for x in new_entry["token_type_ids"]]))

        return new_entry

    def __prepareTokens(self, dataset):
        modified_dataset = []

        entries_to_upper = []
        if self.__uppercase_percentage > 0.0:
            entries_to_upper = random.sample(list(range(len(dataset))),
                                             int(round(len(dataset) * self.__uppercase_percentage)))
            entries_to_upper.sort()

        for (entry_id, entry) in enumerate(tqdm(dataset, total=len(dataset))):
            split_in_more = 0
            entities_to_mask = []

            if self.__create_aligner:
                self.__aligner.append(len(entry["tokens"]))

            if self.__mask_percentage > 0.0 and not self.__test:
                if self.__mask_entities:
                    entity_sequence = self.__findNamedEntities(entry[self.__tags_field])
                    entity_sequence_length = len(entity_sequence)
                    if entity_sequence_length > 3:
                        random_types_id = random.sample(list(range(entity_sequence_length)),
                                                        int(round(entity_sequence_length * self.__mask_percentage)))
                        random_types_id.sort()
                        for type_id in random_types_id:
                            entities_to_mask.extend(entity_sequence[type_id])
                else:
                    tokens_sequence_length = len(entry["tokens"])
                    if tokens_sequence_length > 3:
                        entities_to_mask = random.sample(list(range(tokens_sequence_length)),
                                                         int(round(tokens_sequence_length * self.__mask_percentage)))
                        entities_to_mask.sort()

            bert_tokens_list, new_entry = self.__entryGenerator()
            if self.__tags_field is not None:
                #Added for Cleopatra
                if entry_id in entries_to_upper:
                    upper_tokens = []
                    for token in entry["tokens"]:
                        upper_tokens.append(token.upper())
                    zipped_info = zip(upper_tokens, entry[self.__tags_field])
                else:
                    zipped_info = zip(entry["tokens"], entry[self.__tags_field])
                flag = 2
            else:
                zipped_info = entry["tokens"]
                flag = 3

            tag = None
            size_modifier = 0
            sentence_limit = False
            overflow = -1
            stop_process = False
            for info_id, info in enumerate(zipped_info):
                if flag == 2:
                    token, tag = info
                else:
                    token = info

                bert_tokens = []
                if len(entities_to_mask) > 0 and info_id in entities_to_mask:
                    if self.__predict_masked:
                        bert_tokens = self.__bert_tokenizer.tokenize(token)
                        if len(bert_tokens) == 0:  # If for whatever reason bert_tokens is empty, then, we need to consider that this characters is unknown
                            bert_tokens = ["[UNK]"]
                        bert_tokens_ids = self.__bert_tokenizer.convert_tokens_to_ids(bert_tokens)
                        if self.__subwords_mask:
                            token_to_mask = 0
                            if len(bert_tokens_ids) > 1:
                                token_to_mask = random.randint(0, len(bert_tokens_ids)-1)
                            bert_tokens[token_to_mask] = "[MASK]"
                            new_entry["lm_labels"].append(bert_tokens_ids[token_to_mask])
                        else:
                            # Deliverable and ECIR 2021
                            bert_tokens = []
                            for ids in bert_tokens_ids:
                                new_entry["lm_labels"].append(ids)
                                bert_tokens.append("[MASK]")
                    else:
                        quantity = random.choices([1, 2, 3], weights=[0.76, 0.12, 0.12], k=1)[0]
                        for mask_pos in range(quantity):
                            bert_tokens.append("[MASK]")
                else:
                    #Version submiited in deliverable and ECIR2021
                    #if self.__uppercase and token.isupper():
                    #   token = f"[UP] {token} {token.title()} {token.lower()} [up]"
                    if self.__uppercase and token.isupper():
                        if self.__no_redundant_uppercase:
                            token_title = token.title()
                            if token_title != token_title:
                                token = f"[UP] {token} {token.title()} {token.lower()} [up]"
                            else:
                                token = f"[UP] {token} {token.lower()} [up]"
                        else:
                            token = f"[UP] {token} {token.title()} {token.lower()} [up]"
                    bert_tokens = self.__bert_tokenizer.tokenize(token)

                if len(bert_tokens) == 0:  # If for whatever reason bert_tokens is empty, then, we need to consider that this characters is unknown
                    bert_tokens = ["[UNK]"]
                if flag <= 1 and not sentence_limit:
                    size_modifier = 0
                # The reson for self.__max_length-1 is because, we have an extra token at the end for BERT, thus, a string can be only 126 bert tokens
                #As we might need at least one space for adding <eT> we need to add the size modifier
                if len(new_entry["token_type_ids"]) + len(bert_tokens) > self.__max_length-1-size_modifier:
                    split_in_more += 1
                    if sentence_limit:
                        if len(new_entry["labelling_mask"]) + 1 > self.__max_length-1:
                            print(f"Warning, dev/test sentence representation overflowing, trunking to the maximum number of real tokens allowed: {self.__max_length-1}")
                            stop_process = True
                        else:
                            self.__processOverloadTokensToEntry(new_entry, bert_tokens, tag)
                    elif self.__force_size:
                        difference = self.__max_length-1 - size_modifier - len(new_entry["token_type_ids"])
                        if difference < 0:
                            difference = 0
                        else:
                            if difference > 0:
                                bert_tokens_in = bert_tokens[:difference]
                                bert_tokens_list.extend(bert_tokens_in)
                                self.__processTokensToEntry(new_entry, bert_tokens_in, tag)
                        if self.__test:
                            overflow = len(new_entry["labelling_mask"])
                            self.__processOverloadTokensToEntry(new_entry, bert_tokens[difference:], tag)
                        else:
                            print(f"Warning, training sentence representation overflowing, trunking to {self.__max_length}")
                            stop_process = True
                        sentence_limit = True
                    else:
                        new_entry = self.__processEntry(bert_tokens_list, new_entry)
                        modified_dataset.append(new_entry)
                        bert_tokens_list, new_entry = self.__entryGenerator()
                if not sentence_limit:
                    bert_tokens_list.extend(bert_tokens)
                    self.__processTokensToEntry(new_entry, bert_tokens, tag)
                if stop_process:
                    break

            new_entry = self.__processEntry(bert_tokens_list, new_entry, overflow=overflow)

            modified_dataset.append(new_entry)

            #if split_in_more > 0:
            #    print(f"{split_in_more}\t{' '.join(entry['tokens'])}")

        return modified_dataset

    def __findNamedEntities(self, tags_sequence, skip_o=False, transform_labels=True):
        entity_sequence = []
        original_pairing = []
        for tag_position, tag in enumerate(tags_sequence):
            if transform_labels:
                tag = self.__inverse_tagset[tag]
            if tag[0] == 'O':
                if len(original_pairing) > 0:
                    entity_sequence.append(original_pairing)
                if not skip_o:
                    entity_sequence.append([tag_position])
                original_pairing = []
            elif tag[0] in ['B', 'S']:
                if len(original_pairing) > 0:
                    entity_sequence.append(original_pairing)
                original_pairing = [tag_position]
            else:
                original_pairing.append(tag_position)
        if len(original_pairing) > 0:
            entity_sequence.append(original_pairing)
        return entity_sequence

    def __processOverloadTokensToEntry(self, new_entry, bert_tokens, tag):
        if len(bert_tokens) > 0:
            new_entry["labelling_mask"].append(1)
            if self.__tags_field is not None:
                new_entry["bert_tags"].append(tag)

    def __processTokensToEntry(self, new_entry, bert_tokens, tag):
        first_true_token = 0
        for i in range(len(bert_tokens)):
            new_entry["token_type_ids"].append(0)
            if i == first_true_token:
                new_entry["bert_tokens_mask"].append(1)
                new_entry["labelling_mask"].append(1)
                if self.__tags_field is not None:
                    new_entry["bert_tags"].append(tag)
            else:
                new_entry["bert_tokens_mask"].append(0)
            if self.__predict_masked:
                if bert_tokens[i] == "[MASK]":
                    new_entry["lm_mask"].append(1)
                else:
                    new_entry["lm_mask"].append(0)



    def __getitem__(self, index):
        id_ = self.__entries_ids[index]
        return self.__data[id_]

    def collate_fn(self, batch):
        temp_batch = {}
        for entry in batch:
            for field_name in entry:
                if field_name not in temp_batch:
                    temp_batch[field_name] = []
                temp_batch[field_name].append(entry[field_name])
        batch = temp_batch
        del temp_batch
        bert_tokens = torch.tensor([f for f in batch['bert_tokens']], dtype=torch.long)
        attention_mask = torch.tensor([f for f in batch["attention_mask"]], dtype=torch.long)
        token_type_ids = torch.tensor([f for f in batch["token_type_ids"]], dtype=torch.long)
        bert_tags = None
        if self.__tags_field is not None:
            bert_tags = torch.tensor([f for f in batch["bert_tags"]], dtype=torch.long)
        bert_tokens_mask = torch.tensor([f for f in batch["bert_tokens_mask"]], dtype=torch.long)
        labelling_mask = torch.tensor([f for f in batch["labelling_mask"]], dtype=torch.long)
        lm_mask = None
        lm_labels = None
        if self.__predict_masked:
            lm_mask = torch.tensor([f for f in batch["lm_mask"]], dtype=torch.long)
            lm_labels = torch.tensor([f for f in batch["lm_labels"]], dtype=torch.long)
        labels_boundaries = None
        if self.__multitask:
            labels_boundaries = torch.tensor([f for f in batch["multitask_tags"]], dtype=torch.long)
        if self.__test:
            return bert_tokens, attention_mask, token_type_ids, bert_tags, bert_tokens_mask, labelling_mask
        else:
            if self.__multitask:
                return bert_tokens, attention_mask, token_type_ids, bert_tags, bert_tokens_mask, labelling_mask, lm_mask, lm_labels, labels_boundaries
            else:
                return bert_tokens, attention_mask, token_type_ids, bert_tags, bert_tokens_mask, labelling_mask, lm_mask, lm_labels

