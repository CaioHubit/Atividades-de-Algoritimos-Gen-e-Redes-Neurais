
</head>
<body>
  <h1>Redes Neurais em Python</h1>
<h>
  <p>Este projeto tem como objetivo explorar o uso de redes neurais na linguagem de programação Python, com foco no aprendizado de máquina. Como parte do meu bacharelado em ciências, estou atualmente estudando redes neurais e suas aplicações em diversos campos.</p>
<h>
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:932/1*KE7D8geBF6vg1cRyzTCZRQ.png" alt="Legenda" width="300">
</p>

  <h2>Por que usamos Redes Neurais?</h2>
<h>
  <p>Redes neurais são modelos computacionais inspirados pelo funcionamento do cérebro humano. Elas são capazes de aprender a partir de dados, identificar padrões complexos e tomar decisões ou fazer previsões com base nesses padrões. As redes neurais são especialmente adequadas para resolver problemas que são difíceis de serem abordados por meio de programação tradicional, como reconhecimento de imagens, processamento de linguagem natural, recomendação de produtos, entre outros.</p>

  <h2>Por que treinamos Redes Neurais?</h2>

  <p>As redes neurais são treinadas para aprender a partir de um conjunto de dados conhecidos como conjunto de treinamento. Durante o processo de treinamento, a rede neural ajusta seus parâmetros internos, de modo a minimizar a diferença entre as saídas previstas e as saídas desejadas. O objetivo é fazer com que a rede neural seja capaz de generalizar o aprendizado para dados não vistos anteriormente, permitindo que ela faça previsões precisas em novas situações.</p>

  <h2>Aplicações das Redes Neurais</h2>
<h>
  <p>As redes neurais têm uma ampla gama de aplicações em diversas áreas, incluindo:</p>

  <ul>
    <li>Reconhecimento de padrões e classificação de dados;</li>
    <li>Processamento de linguagem natural e tradução automática;</li>
    <li>Reconhecimento de fala e assistentes virtuais;</li>
    <li>Visão computacional e detecção de objetos em imagens;</li>
    <li>Recomendação de produtos e personalização;</li>
    <li>Predição de séries temporais e previsões financeiras;</li>
    <li>Medicina e diagnóstico médico;</li>
    <li>E muitas outras.</li>
  </ul>

<h2>Anatomia das Redes Neurais</h2>
<h>
<b>Neurônios de entrada (input neurons):</b>

São os neurônios que recebem os valores de entrada da rede neural. Cada neurônio de entrada representa uma característica específica dos dados de entrada. Por exemplo, se estivermos trabalhando com uma rede neural para classificar imagens de gatos e cachorros, cada neurônio de entrada pode representar um pixel da imagem.

<b>Neurônios ocultos (hidden neurons):</b>

Os neurônios ocultos são aqueles que se encontram entre as camadas de entrada e de saída. Eles processam as informações recebidas dos neurônios de entrada ou dos neurônios da camada anterior e transmitem essas informações para os neurônios da camada seguinte. Esses neurônios são responsáveis por extrair características e padrões mais complexos dos dados de entrada.

<b>Neurônios de saída (output neurons):</b>

São os neurônios que fornecem a saída final da rede neural. Cada neurônio de saída representa uma classe ou valor de interesse, dependendo do tipo de tarefa que a rede neural está realizando. Por exemplo, em uma rede neural de classificação binária, podemos ter um neurônio de saída representando a classe "positivo" e outro representando a classe "negativo". Se a rede neural estiver realizando uma regressão, o neurônio de saída pode representar um valor numérico.

<h3>Regra da Cadeia </h3>
<h>
 <p>A regra da cadeia é dada por:</p>

<p align="center">
  <img src="https://machinelearningmastery.com/wp-content/uploads/2021/07/chain_rule_5.png" alt="Legenda" width="300">
</p>
<p>

Ela permite calcular a derivada de uma função composta a partir das derivadas das funções que a compõem.

A regra da cadeia afirma que, se tivermos duas funções diferentes, u(x) e v(x), onde v(x) é uma função de u(x), então a derivada da função composta f(x) = v(u(x)) em relação a x pode ser encontrada através da seguinte fórmula:

f'(x) = v'(u(x)) * u'(x)

Em termos mais simples, para derivar uma função composta, primeiro, calculamos a derivada da função externa (v'(u(x))) e, em seguida, multiplicamos pela derivada da função interna (u'(x)).

Para utilizar a regra da cadeia, você precisa identificar a função composta em questão e suas funções componentes. Em seguida, calcule as derivadas das funções componentes separadamente e, finalmente, aplique a fórmula da regra da cadeia para obter a derivada da função composta.

Vamos considerar um exemplo para ilustrar melhor. Se tivermos a função f(x) = (2x + 3)³, podemos identificar a função composta f(x) = g(h(x)), onde g(x) = x³ e h(x) = 2x + 3.

Agora, vamos calcular as derivadas das funções componentes:
g'(x) = 3x² (derivada de x³)
h'(x) = 2 (derivada de 2x + 3)

Aplicando a regra da cadeia, temos:
f'(x) = g'(h(x)) * h'(x)
= 3(2x + 3)² * 2

Simplificando, obtemos:
f'(x) = 6(2x + 3)²

Portanto, a derivada da função f(x) = (2x + 3)³ em relação a x é igual a 6(2x + 3)².

A regra da cadeia é uma ferramenta poderosa para calcular derivadas de funções compostas e é amplamente utilizada em cálculo diferencial e integral.</p>
<h3> Como calcular o gradiente local em Redes Neurais</h3>

Aqui está um exemplo passo a passo de como usar uma rede neural para realizar essa tarefa:

1. Preparação dos dados:

Temos um conjunto de dados rotulados com pontos (x1, x2) e suas respectivas classes, A ou B.
Dividimos os dados em um conjunto de treinamento e um conjunto de teste.
Definição da arquitetura da rede neural:

Escolhemos uma arquitetura adequada para o problema. Por exemplo, uma rede neural com uma camada oculta contendo 5 neurônios pode ser suficiente para esse caso.

2. Inicialização dos pesos:

Inicializamos os pesos da rede neural com valores aleatórios pequenos.

3. Propagação direta (forward pass):

Alimentamos um ponto de dados de treinamento (x1, x2) na rede neural.
Cada neurônio da camada oculta calcula sua saída multiplicando os valores de entrada pelos pesos correspondentes, passando-os por uma função de ativação, como a função sigmoidal.
As saídas da camada oculta são passadas para a camada de saída, onde a classificação é realizada usando outra função de ativação adequada, como a função sigmoidal para uma classificação binária.
Obtemos a previsão da rede neural para o ponto de dados fornecido.

4. Cálculo do erro:

Comparamos a previsão da rede neural com a classe verdadeira do ponto de dados e calculamos o erro usando uma função de perda, como o erro quadrático médio ou a entropia cruzada.

5. Retropropagação (backpropagation):

Calculamos o gradiente do erro em relação aos pesos da rede neural usando o algoritmo de retropropagação. Esse cálculo envolve a propagação do gradiente de volta pela rede, ajustando os pesos em cada camada de acordo com o gradiente calculado.

. Atualização dos pesos:

Usamos um algoritmo de otimização, como o gradiente descendente estocástico (SGD), para atualizar os pesos da rede neural com base no gradiente calculado. Isso ajuda a minimizar o erro e melhorar o desempenho da rede.

7. Repetição:

Repetimos os passos 4 a 7 para cada ponto de dados de treinamento, ajustando gradualmente os pesos da rede neural para melhorar seu desempenho.
Realizamos várias épocas (iterações completas pelo conjunto de treinamento) até que a rede neural atinja um desempenho satisfatório.

8. Avaliação do desempenho:

Usamos o conjunto de teste para avaliar o desempenho da rede neural após o treinamento.
Alimentamos os pontos de teste na rede neural e comparamos suas previsões com as classes verdadeiras.
Calculamos métricas de desempenho, como a precisão (accuracy), para avaliar quão bem a rede neural está classificando os pontos de teste.
  <h2>Uso do PyTorch e Python puro</h2>

  <p>No desenvolvimento deste projeto, utilizei principalmente a biblioteca PyTorch, uma poderosa biblioteca de aprendizado de máquina em Python. No entanto, também dediquei parte do projeto à implementação em Python puro, a fim de aprofundar minha compreensão sobre os princípios fundamentais das redes neurais.</p>

  <p>Se você estiver interessado em aprender mais sobre redes neurais em Python, recomendo explorar as possibilidades oferecidas pelo PyTorch, assim como estudar o funcionamento interno das redes neurais utilizando apenas Python puro.</p>

  <h2>Referências</h2>

  <p>Aqui estão algumas referências úteis para aprender mais sobre redes neurais em Python:</p>

  <ul>
    <li> </li>
  </ul>

  <p>Este README é parte de um projeto acadêmico e está em constante desenvolvimento. Fique à vontade para entrar em contato comigo para mais informações ou discussões sobre redes neurais em Python.</p>
</body>
</html>

