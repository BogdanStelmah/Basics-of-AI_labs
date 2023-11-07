import numpy as np
import random

from rnn import RNN
from data import train_data, test_data

# Створити словник
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)

# Призначити індекс кожному слову
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
print(word_to_idx['good'])
print(idx_to_word[0])

def createInputs(text):
  '''
  Повертає масив унітарних векторів які представляють слова у введеному рядку тексту
  - текст є рядком string
  - Унітарний вектор має форму (vocab_size, 1)
  '''

  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs

def softmax(xs):
  # Застосування функції Softmax для вхідного масиву
  return np.exp(xs) / sum(np.exp(xs))

# Ініціалізація нашої рекурентної нейронної мережі RNN
rnn = RNN(vocab_size, 2)

inputs = createInputs('i am very good')
out, h = rnn.forward(inputs)
probs = softmax(out)
print(probs) # [[0.50000095], [0.49999905]]

def processData(data, backprop=True):
  '''
  Повернення втрат RNN і точності для даних
  - дані подані як словник, що відображує текст як True або False.
  - backprop визначає, чи потрібно використовувати звороднє розподілення
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = createInputs(x)
    target = int(y)

    # Пряме розподілення
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Обчислення втрат / точності
    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    if backprop:
      # Создание dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Зворотне розподілення
      rnn.backprop(d_L_d_y)

  return loss / len(data), num_correct / len(data)

# Цикл тренування
for epoch in range(1000):
  train_loss, train_acc = processData(train_data)

  if epoch % 100 == 99:
    print('--- Epoch %d' % (epoch + 1))
    print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

    test_loss, test_acc = processData(test_data, backprop=False)
    print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))
