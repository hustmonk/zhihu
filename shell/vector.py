import torch, numpy
def vectorize(ex, model):
    questionid, scenario, passage, questiontext, answer1, answer2, label = ex
    word_dict = model.word_dict
    passage = torch.LongTensor([word_dict[w] for w in passage.split(" ")])
    question = torch.LongTensor([word_dict[w] for w in questiontext.split(" ")])
    answer1 = torch.LongTensor([word_dict[w] for w in answer1.split(" ")])
    answer2 = torch.LongTensor([word_dict[w] for w in answer2.split(" ")])

    if numpy.random.random() > 0.5:
        questioninfo = torch.cat([question, answer1, answer2], -1)
    else:
        questioninfo = torch.cat([question, answer2, answer1], -1)

    qanswer1 = torch.cat([question, answer1], -1)
    qanswer2 = torch.cat([question, answer2], -1)

    label = torch.LongTensor(1).fill_(label)

    return [questionid, [passage, question, questioninfo, answer1, answer2, qanswer1, qanswer2], label]

def tomask(texts):
    # Batch questions
    max_length = max([q.size(0) for q in texts])
    x = torch.LongTensor(len(texts), max_length).zero_()
    x_mask = torch.ByteTensor(len(texts), max_length).fill_(1)
    for i, q in enumerate(texts):
        x[i, :q.size(0)].copy_(q)
        x_mask[i, :q.size(0)].fill_(0)
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
#                  answer1, answer1_mask, answer2, answer2_mask,
                  qanswer1, qanswer1_mask, qanswer2, qanswer2_mask], targets]
