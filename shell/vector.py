import torch, numpy, random
def to_idx_torch(tokens, tokenizer):
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(ids) > 512:
        ids = ids[:512]
    return torch.LongTensor(ids)

def vectorize(ex, tokenizer):
    questionid, scenario, passage, question, answer1, answer2, label = ex
    k = [answer1, answer2]
    random.shuffle(k)
    questioninfo = question + ["[SEP]"] + k[0] + ["[SEP]"] + k[1]
    qanswer1 = question + ["[SEP]"] + answer1
    qanswer2 = question + ["[SEP]"] + answer2

    passage = to_idx_torch(passage, tokenizer)
    question = to_idx_torch(question, tokenizer)
    answer1 = to_idx_torch(answer1, tokenizer)
    answer2 = to_idx_torch(answer2, tokenizer)
    questioninfo = to_idx_torch(questioninfo, tokenizer)
    qanswer1 = to_idx_torch(qanswer1, tokenizer)
    qanswer2 = to_idx_torch(qanswer2, tokenizer)

    label = torch.LongTensor(1).fill_(label)

    return [questionid, [passage, question, questioninfo, answer1, answer2, qanswer1, qanswer2], label]

def tomask(texts):
    # Batch questions
    max_length = max([q.size(0) for q in texts])
    x = torch.LongTensor(len(texts), max_length).zero_()
    x_mask = torch.ByteTensor(len(texts), max_length).fill_(0)
    for i, q in enumerate(texts):
        x[i, :q.size(0)].copy_(q)
        x_mask[i, :q.size(0)].fill_(1)
    return x, x_mask

def batchify(batch):
    ids = [ex[0] for ex in batch]
    input_num = len(batch[0][1])
    inputs = [[ex[1][k] for ex in batch] for k in range(input_num)]
    passage, question, questioninfo, answer1, answer2, qanswer1, qanswer2 = inputs

    targets = torch.cat([ex[2] for ex in batch])

    passage, passage_mask = tomask(passage)
    question, question_mask = tomask(question)
    questioninfo, questioninfo_mask = tomask(questioninfo)
    answer1, answer1_mask = tomask(answer1)
    answer2, answer2_mask = tomask(answer2)
    qanswer1, qanswer1_mask = tomask(qanswer1)
    qanswer2, qanswer2_mask = tomask(qanswer2)

    return [ids, [passage, passage_mask, question, question_mask, questioninfo, questioninfo_mask,
                  answer1, answer1_mask, answer2, answer2_mask,
                  qanswer1, qanswer1_mask, qanswer2, qanswer2_mask], targets]
