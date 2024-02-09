# Implementando GPT
Fala pessoal, esse é um git de estudo mostrando o passo a passo da implementação de um GPT, como o ChatGPT da OpenAI.

Esse git foi inspirado principalmente no vídeo do Andrej Karpathy ( Let's build GPT: from scratch no Youtube).

Toda a implementação será baseado no paper Original da arquitetura Transformers ( Attention is all you Need ).
![transformers](https://github.com/gugaio/transformer/assets/17186525/3b2d9f43-b09e-4410-be6a-94ec7882d8b5)

Parece meio complicada a imagen mas é simples de entender passo a passo. Vou explicando cada parte da imagem ao longo do tutorial.

No final da implementação vou mostrar o código original de vários GPT como o ChatGPT da OpenAI e Bart do Google.

A implementação será feita em 5 fases, pra ajudar a entender a evolução de cada ponto da arquitetura.
* Gerador simples
* Implementando Self Attention
* Implementando Multi Head Attention
* Implementando o Decoder

# Passo 1 - Gerador simples

No primeiro passo, vamos implementar o nosso gerador v0, sem nada de transformer ainda. 

Basicamente ele vai receber um texto e escrever a próxima palavra, como qualquer GPT.
A primeira tarefa será carregar um grande texto para o treinamento do GPT.

![texto](https://github.com/gugaio/transformer/assets/17186525/a2477def-a20f-4323-80d9-5ca18c2f7ffe)

## Texto para treinamento

Isso é bem simples, basta um grande texto. Aqui disponibilizei um texto de exemplo no github e vou baixar com wget e carregar normalmente em python
```
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
input = "input.txt"
with open(input, 'r', encoding='utf8') as f:
  file_content = f.read()
```
Entretanto você deve saber que toda I.A funciona com números, os chamados tokens, e não texto. Por isso nosso próximo passo é implementar um Tokenizer que basicamente converte qualquer texto numa lista de tokens id, onde cada ID representa uma palavra.

![tokenizer](https://github.com/gugaio/transformer/assets/17186525/b1497ae5-55ce-4a9b-a9e4-378f50d712eb)

## Implementando nosso Tokenizer

