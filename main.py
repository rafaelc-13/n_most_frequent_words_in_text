import re
import heapq
from collections import defaultdict
import random

def top_n_frequent_words_streaming(source, n):
    """
    Streaming solution for top-n frequent words from an iterable of text chunks.
    """
    freq = defaultdict(int)

    # 1) Tokenize the chunks and immediately count in freq
    for chunk in source:
        for word in re.findall(r'\w+', chunk.lower()):
            freq[word] += 1

    # 2) Min-heap selection - specialized tree‑based structure
    heap = []
    for word, count in freq.items():
        if len(heap) < n:
            heapq.heappush(heap, (count, word))
        else:
            if count > heap[0][0] or (count == heap[0][0] and word < heap[0][1]):
                heapq.heapreplace(heap, (count, word))

    # 3) Sort for final ordering
    return [word for count, word in sorted(heap, key=lambda x: (-x[0], x[1]))]


# ---------- TEST WITH 600-WORD STRING ----------

# Create a vocabulary and skew the distribution
common_words = ["data", "growth", "insight"]
other_words = [f"word{i}" for i in range(1, 51)]
vocab = common_words * 20 + other_words  # common words have higher weight

# Generate 600 words
random.seed(42)  # reproducibility
words_list = [random.choice(vocab) for _ in range(600)]
long_text = " ".join(words_list)

# Run streaming top-N (feeding one chunk here: the whole string)
result = top_n_frequent_words_streaming([long_text], 5)
