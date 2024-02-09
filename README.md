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

