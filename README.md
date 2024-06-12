# Language-Modeling

## Summary of Work Done
The project demonstrated an introduction to language modeling, defining it as the task of building models that can predict the likelihood of sequences in a given language. We tackled this by implementing a character-based model, focusing on predicting a character given a sequence of preceding characters. Two primary models were developed:

N-gram Model: We constructed a simple 4-gram model using relative frequency counts with Laplace smoothing to handle unseen n-grams. This model calculates the probability of a character based on the previous three characters, adjusting for frequency and vocabulary size to avoid assigning zero probability to unseen n-grams in the test set.

LSTM Network Model: We advanced to a more sophisticated approach using an LSTM (Long Short-Term Memory) network. This model involved building an LSTM-based recurrent neural network that processes input sequences into character embeddings, which are then used to predict the next character in the sequence. The LSTM model was designed to handle longer dependencies and contexts more effectively than the simpler n-gram model, using layers of memory cells that retain information over longer text spans.

## Key Configurations
Character Embedding Dimension: Set at 50.
LSTM Hidden Dimension: 200.
Vocabulary Size: 386 unique characters.
Sequence Length for Training: Fixed to 500 characters.
Training Epochs: Maximum of 5 epochs were allowed with learning rates of 0.0001, 0.00001, and 0.000001.
LSTM Layers: Models with one and two layers were tested to compare performance impacts.

## Evaluation Metrics
The primary evaluation metric used was perplexity, which measures how well the model predicts a sample. A lower perplexity score indicates a better predictive model as it implies less confusion. Both models were assessed based on their perplexity scores on a designated test set.

## Observations and Outcomes
Performance: The LSTM model generally outperformed the n-gram model in terms of perplexity, showcasing its ability to capture longer-term dependencies within the data.
Complexity and Training: Although the LSTM model was computationally more intensive, requiring significant training time and resources, it provided a more nuanced understanding of character sequences.
