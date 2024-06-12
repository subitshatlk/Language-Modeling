import torch
import pickle
from scripts import utils
import argparse
import math
from collections import defaultdict
from scripts.utils import get_files, convert_files2idx, convert_line2idx 
torch.manual_seed(42)

def load_vocabulary(file_path):
    with open(file_path, 'rb') as f:
        vocabulary = pickle.load(f)
        print("Vocab loaded")
    return vocabulary

def train_and_compute_perplexity(train_data, test_data,train_probabilities, trigrams, fourgrams, unigrams,vocabulary):
  for line in train_data:
      chars = line
      for i in range(0,len(chars)):
          if i >= 2:
              trigrams[(chars[i-2], chars[i-1], chars[i])] += 1
          if i >= 3:
              fourgrams[(chars[i-3], chars[i-2], chars[i-1], chars[i])] += 1

  vocab_length = len(vocabulary)
 
  total_perplexity = 0
  for idx, line in enumerate(test_data):
      chars = line
      n = len(chars) 
      probs_sum = 0
      tri_grams = 0
      four_grams = 0
      calc_loss = 0

      for i in range(3,len(chars)):
          trigram = (chars[i-3], chars[i-2], chars[i-1])
          fourgram = (chars[i-3], chars[i-2], chars[i-1], chars[i]) 
          if trigram in trigrams:
            tri_grams = trigrams[trigram]

          if fourgram in fourgrams:
            four_grams = fourgrams[fourgram]
          probability = (four_grams+1)/(tri_grams+vocab_length)
          probs_sum += (math.log2(probability)) 

      calc_loss = -(probs_sum/n) 
      total_perplexity +=  2**calc_loss

  return total_perplexity/len(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default="/scratch/general/vast/u1471339/mp3")
    parser.add_argument('--model_dir', type=str, default="models")
    parser.add_argument('--input_dir', type=str, default="/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp3/mp3_release")
    parser.add_argument('--vocab_file', type=str, default="vocab.pkl")
    parser.add_argument('--k', type=int, default="4")
    
    args, _ = parser.parse_known_args()

    train_files = utils.get_files(f'{args.input_dir}/data/train')
    dev_files = utils.get_files(f'{args.input_dir}/data/dev')
    test_files = utils.get_files(f'{args.input_dir}/data/test')

    vocabulary = load_vocabulary("/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp3/mp3_release/data/vocab.pkl")
    print('Vocabulary Loaded')

    train_data = utils.convert_files2idx(train_files,vocabulary)
    test_data = utils.convert_files2idx(test_files,vocabulary)
    dev_data = utils.convert_files2idx(dev_files,vocabulary)

    train_probabilities = defaultdict(float)
    trigrams = defaultdict(int)
    fourgrams = defaultdict(int)
    unigrams = defaultdict(int)

    train_and_compute_perplexity(train_data, test_data, train_probabilities, trigrams, fourgrams, unigrams,vocabulary)
