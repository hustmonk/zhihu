import torch, numpy, random

def to_idx_torch(tokens1, tokens2, CLS, SEP):
    special_size = 3
    length = special_size + len(tokens1) + len(tokens2)
    if length > 400:
        tokens2length = 400 - special_size - len(tokens1)
        tokens2 = tokens2[:tokens2length]
    ids = CLS + tokens1 + SEP + tokens2 + SEP
    segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)
    return torch.LongTensor(ids), torch.LongTensor(segment_ids)

def vectorize(ex, tokenizer, training):
    questionid, scenario, passage, question, answer1, answer2, label = ex
    MASK = tokenizer.convert_tokens_to_ids(["[MASK]"])
    word_dropout = 0.1
    def worddropout(context):
        if training and word_dropout > 0:
            for j in range(len(context)):
                if numpy.random.rand() < word_dropout:
                    context[j] = MASK[0]
        return context

    answer1 = worddropout(tokenizer.convert_tokens_to_ids(answer1))
    passage = worddropout(tokenizer.convert_tokens_to_ids(passage))
    question = worddropout(tokenizer.convert_tokens_to_ids(question))
    answer2 = worddropout(tokenizer.convert_tokens_to_ids(answer2))

    CLS = tokenizer.convert_tokens_to_ids(["[CLS]"])
    SEP = tokenizer.convert_tokens_to_ids(["[SEP]"])
    answer1_ids, answer1_segment_ids = to_idx_torch(answer1 + SEP + question, passage, CLS, SEP)
    answer2_ids, answer2_segment_ids = to_idx_torch(answer2 + SEP + question, passage, CLS, SEP)
    qanswer1_ids, qanswer1_segment_ids = to_idx_torch(answer1, question + SEP + answer2, CLS, SEP)
    qanswer2_ids, qanswer2_segment_ids = to_idx_torch(answer2, question + SEP + answer1, CLS, SEP)
    label = torch.LongTensor([label])

    return [questionid, [answer1_ids, answer1_segment_ids, answer2_ids, answer2_segment_ids, qanswer1_ids, qanswer1_segment_ids, qanswer2_ids, qanswer2_segment_ids], label]

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
    answer1, answer1_segment_ids, answer2, answer2_segment_ids, qanswer1, qanswer1_segment_ids, qanswer2, qanswer2_segment_ids = inputs

    targets = torch.cat([ex[2] for ex in batch])

    answer1, answer1_segment_ids, answer1_mask = tomask(answer1, answer1_segment_ids)
    answer2, answer2_segment_ids, answer2_mask = tomask(answer2, answer2_segment_ids)

    qanswer1, qanswer1_segment_ids, qanswer1_mask = tomask(qanswer1, qanswer1_segment_ids)
    qanswer2, qanswer2_segment_ids, qanswer2_mask = tomask(qanswer2, qanswer2_segment_ids)

    return [ids, [answer1, answer1_segment_ids, answer1_mask, answer2, answer2_segment_ids, answer2_mask,
                  qanswer1, qanswer1_segment_ids, qanswer1_mask, qanswer2, qanswer2_segment_ids, qanswer2_mask], targets]
