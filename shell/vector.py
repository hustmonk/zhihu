import torch, numpy, random

def to_idx_torch(tokens1, tokens2, CLS, SEP, lenanswer):
    ids = torch.LongTensor(CLS + tokens1 + SEP + tokens2 + SEP)
    segment_ids = torch.LongTensor([0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1))
    answer_core_mask = torch.LongTensor([0] * (lenanswer + 1))
    passage_mask = [1] * ids.size(0)
    for i in range(len(tokens1) + 2, ids.size(0)):
        passage_mask[i] = 0
    passage_mask = torch.LongTensor(passage_mask)
    return ids, segment_ids, answer_core_mask, passage_mask

def vectorize(ex, tokenizer):
    questionid, scenario, passage, question, answer1, answer2, label = ex

    answer1 = tokenizer.convert_tokens_to_ids(answer1)
    passage = tokenizer.convert_tokens_to_ids(passage)
    question = tokenizer.convert_tokens_to_ids(question)
    answer2 = tokenizer.convert_tokens_to_ids(answer2)
    if len(passage) > 400:
        passage = passage[:400]

    CLS = tokenizer.convert_tokens_to_ids(["[CLS]"])
    SEP = tokenizer.convert_tokens_to_ids(["[SEP]"])
    answer1_info = to_idx_torch(answer1, question + SEP + passage, CLS, SEP, len(answer1))
    answer2_info = to_idx_torch(answer2, question + SEP + passage, CLS, SEP, len(answer2))
    qanswer1_info = to_idx_torch(answer1, question, CLS, SEP, len(answer1))
    qanswer2_info = to_idx_torch(answer2, question, CLS, SEP, len(answer2))
    label = torch.LongTensor([label])

    return [questionid, [answer1_info, answer2_info, qanswer1_info, qanswer2_info], label]

def tomask(info):
    ids, segment_ids, answer_core_mask, passage_mask = info
    # Batch questions
    max_length = max([q.size(0) for q in ids])
    x = torch.LongTensor(len(ids), max_length).zero_()
    x_segments = torch.LongTensor(len(ids), max_length).zero_()
    x_mask = torch.ByteTensor(len(ids), max_length).fill_(0)
    x_core = torch.ByteTensor(len(ids), max_length).fill_(1)
    x_passage_mask = torch.ByteTensor(len(ids), max_length).fill_(1)
    for i, q in enumerate(ids):
        x[i, :q.size(0)].copy_(q)
        x_segments[i, :q.size(0)].copy_(segment_ids[i])
        x_mask[i, :q.size(0)].fill_(1)
        x_core[i, :len(answer_core_mask[i])].copy_(answer_core_mask[i])
        x_passage_mask[i, :len(passage_mask[i])].copy_(passage_mask[i])

    return x, x_segments, x_mask, x_core, x_passage_mask

def batchify(batch):
    ids = [ex[0] for ex in batch]
    num1 = len(batch[0][1])
    inputs = []
    for i in range(num1):
        num2 = len(batch[0][1][0])
        info = [[ex[1][i][k] for ex in batch] for k in range(num2)]
        inputs.append(info)

    answer1_info, answer2_info, qanswer1_info, qanswer2_info = inputs

    targets = torch.cat([ex[2] for ex in batch])

    answer1_info = tomask(answer1_info)
    answer2_info = tomask(answer2_info)

    qanswer1_info = tomask(qanswer1_info)
    qanswer2_info = tomask(qanswer2_info)

    return [ids, [answer1_info, answer2_info, qanswer1_info, qanswer2_info], targets]
