import torch, numpy, random

def to_idx_torch(tokens1, tokens2, tokenizer):
    CLS = tokenizer.convert_tokens_to_ids(["[CLS]"])
    SEP = tokenizer.convert_tokens_to_ids(["[SEP]"])
    tokens1 = tokenizer.convert_tokens_to_ids(tokens1)
    tokens2 = tokenizer.convert_tokens_to_ids(tokens2)
    length = 3 + len(tokens1) + len(tokens2)
    if length > 512:
        tokens2length = 512 - 3 - len(tokens1)
        tokens2 = tokens2[:tokens2length]
    ids = CLS + tokens1 + SEP + tokens2 + SEP
    segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)
    return torch.LongTensor(ids), torch.LongTensor(segment_ids)

def vectorize(ex, tokenizer):
    questionid, scenario, passage, question, answer1, answer2, label = ex

    answer1, answer1_segment_ids = to_idx_torch(question + answer1, passage, tokenizer)
    answer2, answer2_segment_ids = to_idx_torch(question + answer2, passage, tokenizer)

    label = torch.LongTensor(1).fill_(label)

    return [questionid, [answer1, answer1_segment_ids, answer2, answer2_segment_ids], label]

def tomask(texts, segment_ids):
    # Batch questions
    max_length = max([q.size(0) for q in texts])
    x = torch.LongTensor(len(texts), max_length).zero_()
    x_segments = torch.LongTensor(len(texts), max_length).zero_()
    x_mask = torch.ByteTensor(len(texts), max_length).fill_(0)
    for i, q in enumerate(texts):
        x[i, :q.size(0)].copy_(q)
        x_segments[i, :q.size(0)].copy_(segment_ids[i])
        x_mask[i, :q.size(0)].fill_(1)
    return x, x_segments, x_mask

def batchify(batch):
    ids = [ex[0] for ex in batch]
    input_num = len(batch[0][1])
    inputs = [[ex[1][k] for ex in batch] for k in range(input_num)]
    answer1, answer1_segment_ids, answer2, answer2_segment_ids = inputs

    targets = torch.cat([ex[2] for ex in batch])

    answer1, answer1_segment_ids, answer1_mask = tomask(answer1, answer1_segment_ids)
    answer2, answer2_segment_ids, answer2_mask = tomask(answer2, answer2_segment_ids)

    return [ids, [answer1, answer1_segment_ids, answer1_mask, answer2, answer2_segment_ids, answer2_mask], targets]
