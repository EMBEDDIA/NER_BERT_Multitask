import pickle as pkl

from regex import regex


class Dataset:

    mapping = {}

    def __init__(self, dataset, predifined_mapping=None, tokenizer=None):
        self.__mappings = Dataset.mapping
        self.__split_by = dataset['columnsSeparator']
        self.__special_labels = dataset["specialLabels"]
        self.__processed_data = None
        self.__tokenizer = tokenizer
        self.__add_tokens = set()

        if predifined_mapping is not None:
            self.__mappings = predifined_mapping

        dataset_columns = dataset['columns']
        comment_symbols = tuple(dataset['commentSymbol'])
        base_path = dataset["basePath"]
        paths = []
        labels_as_training = []
        data_type_names = []
        for data_type_name, data_type in dataset["dataType"].items():
            if "labelsAsTraining" in data_type and data_type['labelsAsTraining']:
                paths.insert(0, f"{base_path}/{data_type['path']}")
                labels_as_training.insert(0, data_type['labelsAsTraining'])
                data_type_names.insert(0, data_type_name)
            else:
                paths.append(f"{base_path}/{data_type['path']}")
                labels_as_training.append(False)
                data_type_names.append(data_type_name)

        dataset_elements = zip(data_type_names, paths, labels_as_training)

        self.__processed_data = self.__readData(dataset_elements, dataset_columns, comment_symbols)

    def getMapping(self):
        return self.__mappings

    def getProcessedData(self):
        return self.__processed_data

    def getTokensToAdd(self):
        return list(self.__add_tokens)

    def __readData(self, dataset_elements, cols, commentSymbols):

        data = {}
        for name, path, labels_as_training in dataset_elements:
            print(f"File processed: {path}")
            sentences, comments = self.__readCoNLL(path, cols, commentSymbols)
            self.__vectorize(sentences, training=labels_as_training)
            #sentences["comments"] = comments
            data[name] = sentences
            data[f"{name}_comments"] = comments
        return data

    def __readCoNLL(self, inputPath, cols, commentSymbols=None):
        """
        Reads in a CoNLL file and returns a list with sentences (each sentence is a list of tokens)
        """
        sentences = []
        comments = {}

        sentenceTemplate = {name: [] for name in cols.values()}

        sentence = {name: [] for name in sentenceTemplate.keys()}
        commented_lines = []
        comments_counter = -1
        newData = False
        lastval = 'O'
        for line in open(inputPath):
            line = line.strip("\n")
            line = line.strip("\r")
            if len(line) == 0 or (commentSymbols is not None and line.startswith(commentSymbols)):
                if commentSymbols is not None and line.startswith(commentSymbols):
                    commented_lines.append(line)
                if newData:
                    sentences.append(sentence)
                    sentence = {name: [] for name in sentenceTemplate.keys()}
                    if len(commented_lines) > 0:
                        comments[comments_counter] = commented_lines
                        commented_lines = []
                    comments_counter += 1
                    newData = False
                continue

            splits = line.split(self.__split_by)
            #print(splits)

            for colIdx, colName in cols.items():
                val = splits[colIdx]
                sentence[colName].append(val)

            newData = True

        if newData:
            sentences.append(sentence)
        if len(commented_lines) > 0:
            comments[comments_counter] = commented_lines

        for name in cols.values():
            if name.endswith('_IOB'):

                #Experimental Adrián: only borders detection
                className = name[0:-4] + '_IOBA'
                for sentence in sentences:
                    sentence[className] = []
                    lastval = 'O'
                    for val in sentence[name]:
                        newval = val
                        if (lastval == 'O' and val[0] == 'I') or (
                                val != 'O' and lastval != 'O' and lastval[1:] != val[1:]):
                            newval = 'B-A'
                        elif val[0] != 'O':
                            newval = val[0] + '-A'
                        lastval = val
                        val = newval
                        sentence[className].append(val)

                # FIX using IOB2 or BIO: if 'I-MISC', 'I-MISC', 'O', 'I-PER', 'I-PER', -> converts into -> 'B-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER',
                className = name[0:-4] + '_IOB2'
                for sentence in sentences:
                    sentence[className] = []
                    lastval = 'O'
                    for val in sentence[name]:
                        newval = val
                        if (lastval == 'O' and val[0] == 'I') or (
                                val != 'O' and lastval != 'O' and lastval[1:] != val[1:]):
                            newval = 'B' + val[1:]
                        lastval = val
                        val = newval
                        sentence[className].append(val)

                # Add class
                className = name[0:-4] + '_class'
                for sentence in sentences:
                    sentence[className] = []
                    for val in sentence[name]:
                        valClass = val[2:] if val != 'O' else 'O'
                        sentence[className].append(valClass)

                # Add IOB encoding
                iobName = name[0:-4] + '_IOBX'
                for sentence in sentences:
                    sentence[iobName] = []
                    oldVal = 'O'
                    for val in sentence[name]:
                        newVal = val

                        if newVal[0] == 'B':
                            if oldVal != 'I' + newVal[1:]:
                                newVal = 'I' + newVal[1:]

                        sentence[iobName].append(newVal)
                        oldVal = newVal

                # Add IOBES encoding
                iobesName = name[0:-4] + '_IOBES'
                className = name[0:-4] + '_IOB2'
                for sentence in sentences:
                    sentence[iobesName] = []

                    for pos in range(len(sentence[className])):
                        val = sentence[className][pos]
                        nextVal = sentence[className][pos +
                                                 1] if (pos + 1) < len(sentence[className]) else 'O'
                        prevVal = sentence[className][pos - 1] if pos > 0 else 'O'

                        newVal = val
                        if val[0] == 'B' and nextVal[0] != 'I':
                            newVal = 'S' + val[1:]
                        elif val[0] == 'I' and nextVal[0] != 'I':
                            newVal = 'E' + val[1:]
                        sentence[iobesName].append(newVal)

                # Add IOBESA encoding (only borders)
                iobesName = name[0:-4] + '_IOBESA'
                className = name[0:-4] + '_IOBES'
                for sentence in sentences:
                    sentence[iobesName] = []

                    for pos in range(len(sentence[className])):
                        val = sentence[className][pos]

                        newVal = val
                        if val != 'O':
                            newVal = f"{val[0]}-A"

                        sentence[iobesName].append(newVal)

        return sentences, comments

    def __vectorize(self, sentences, training=False):
        sentenceKeys = list(sentences[0].keys())
        for sentence in sentences:
            for name in sentenceKeys:
                if name == "NER_IOBX":
                    continue
                if name.startswith("NER_"):
                    if name not in self.__mappings and training:
                        self.__mappings[name] = {"O": 1}
                        if self.__special_labels:
                            self.__mappings[name]["[CLS]"] = len(self.__mappings[name]) + 1
                            self.__mappings[name]["[SEP]"] = len(self.__mappings[name]) + 1
                    for (id_, item) in enumerate(sentence[name]):
                        if item not in self.__mappings[name]:
                            if training:
                                self.__mappings[name][item] = len(self.__mappings[name]) + 1
                            else:
                                print(f"Issue with the label {item} in {name}")
                                exit(1)
                        sentence[name][id_] = self.__mappings[name][item]
                if name == "tokens" and training and self.__tokenizer is not None:
                    for token in sentence["tokens"]:
                        bert_tokens = self.__tokenizer.tokenize(token)
                        if self.__validateBertTokens(token, bert_tokens) == 1 and token not in self.__add_tokens:
                            if regex.search("\p{P}|\p{S}", token):
                                new_tokens = list(filter(None, regex.split("(\p{P}|\p{S})", token)))
                                for sub_token in new_tokens:
                                    bert_tokens = self.__tokenizer.tokenize(sub_token)
                                    if self.__validateBertTokens(sub_token, bert_tokens) == 1:
                                        self.__add_tokens.add(sub_token)
                            else:
                                self.__add_tokens.add(token)

    def __validateBertTokens(self, token, bert_tokens):
        if len(bert_tokens) == 0:
            self.__add_tokens.add(token)
            return 0
        elif len(bert_tokens) == 1 and bert_tokens[0] == "[UNK]":
            return 1
        else:
            return 0
