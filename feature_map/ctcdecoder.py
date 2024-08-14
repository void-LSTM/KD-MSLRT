
import numpy as np
import math
import collections
import torch
import  time
NEG_INF = -float("inf")
def ctc_beam_search_decode(probs, beam_size=5, blank=0):
    """
    :param probs: The output probabilities (e.g. post-softmax) for each
    time step. Should be an array of shape (time x output dim).
    :param beam:
    :param blank:
    :return:
    """
    # T表示时间，S表示词表大小
    T, S = probs.shape

    # 求概率的对数
    probs = np.log(probs)

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    # 每次总是保留beam_size条路径
    beam = [(tuple(), ((0.0, NEG_INF), tuple()))]

    for t in range(T):  # Loop over time
        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()

        for s in range(S):  # Loop over vocab
            # print(s)
            p = probs[t, s]  # t时刻，符号为s的概率

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, ((p_b, p_nb), prefix_p) in beam:  # Loop over beam
                # p_b表示前缀最后一个是blank的概率，p_nb是前缀最后一个非blank的概率
                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.

                if s == blank:
                    # 增加的字母是blank
                    # 先取出对应prefix的两个概率，然后更后缀为blank的概率n_p_b
                    (n_p_b, n_p_nb), _ = next_beam[prefix]  # -inf, -inf
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)  # 更新后缀为blank的概率
                    next_beam[prefix] = ((n_p_b, n_p_nb), prefix_p)  # s=blank， prefix不更新，因为blank要去掉的。
                    # print(next_beam[prefix])
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)  # 更新 prefix, 它是一个tuple
                n_prefix_p = prefix_p + (p,)
                # 先取出对应 n_prefix 的两个概率, 这个是更新了blank概率之后的 new 概率
                (n_p_b, n_p_nb), _ = next_beam[n_prefix]  # -inf, -inf

                if s != end_t:
                    # 如果s不和上一个不重复，则更新非空格的概率
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # 如果s和上一个重复，也要更新非空格的概率
                    # We don't include the previous probability of not ending
                    # in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    n_p_nb = logsumexp(n_p_nb, p_b + p)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:
                    (n_p_b, n_p_nb), n_prefix_p = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    # 如果是s=end_t，则prefix不更新
                    next_beam[prefix] = ((n_p_b, n_p_nb), n_prefix_p)
                else:
                    # *NB* this would be a good place to include an LM score.
                    next_beam[n_prefix] = ((n_p_b, n_p_nb), n_prefix_p)
        # print(t, next_beam.keys())
        # Sort and trim the beam before moving on to the
        # next time-step.
        # 根据概率进行排序，每次保留概率最高的beam_size条路径
        beam = sorted(next_beam.items(),
                      key=lambda x: logsumexp(*x[1][0]),
                      reverse=True)
        beam = beam[:beam_size]

    # best = beam[0]
    # return best[0], -logsumexp(*best[1][0]), best[1][1]

    pred_lens = [len(beam[i][0]) for i in range(beam_size)]
    max_len = max(pred_lens)
    pred_seq, scores, pred_pobs = np.zeros((beam_size, max_len), dtype=np.int32), \
                                  [], np.zeros((beam_size, max_len))
    for bs in range(beam_size):
        pred_seq[bs][:pred_lens[bs]] = beam[bs][0]
        scores.append(-logsumexp(*beam[bs][1][0]))
        pred_pobs[bs][:pred_lens[bs]] = np.exp(beam[bs][1][1])
    return pred_seq, scores, pred_pobs


# 因为代码中为了避免数据下溢，都采用的是对数概率，所以看起来比较繁琐
def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))  # 概率相加再取log，为避免数值下溢
    return a_max + lsp


# 创建一个新的beam
def make_new_beam():
    fn = lambda: ((NEG_INF, NEG_INF), tuple())
    return collections.defaultdict(fn)

# if __name__ == "__main__":
    

#     np.random.seed(3)

#     seq_len = 50
#     output_dim = 20

#     probs = np.random.rand(seq_len, output_dim)
#     # probs = np.random.rand(time, output_dim)
#     # probs = np.random.rand(time, output_dim)
#     probs = probs / np.sum(probs, axis=1, keepdims=True)

#     start_time = time.time()
#     labels, score, labels_p = ctc_beam_search_decode(probs, beam_size=5, blank=0)
#     print("labels:", labels[0], len(labels[0]))
#     print("labels_p: ", labels_p[0], len(labels_p[0]))
#     print("Score {:.3f}".format(score[0]))
#     print("First method time: ", time.time() - start_time)

    # dec_logits = torch.FloatTensor(probs).unsqueeze(0)
    # len_video = torch.LongTensor([seq_len])
    # decoder_vocab = [chr(x) for x in range(20000, 20000 + output_dim)]

    # second_time = time.time()
    # decoder = ctcdecode.CTCBeamDecoder(decoder_vocab, beam_width=5, blank_id=0, num_processes=10)

    # pred_seq, scores, _, out_seq_len = decoder.decode(dec_logits, len_video)

    # # pred_seq: [batch, beam, length]
    # # out_seq_len: [batch, beam]
    # print(pred_seq[0, 0, :][:out_seq_len[0, 0]])
    # print(out_seq_len[0, 0])
    # print("Score {:.3f}".format(scores[0, 0]))
    # print("Second method time: ", time.time() - second_time)