# Prompt Policy 


**Autor:** Miguel Angel Cabrera Victoria

**Insititucion :** Insitituto Tecnologico y de Estudios Superiores De Monterrey

**Email:** A01782982@tec.com.mx | miguelangelcabreravictoria@gmail.com

**Fecha :** 29 de Mayo del 2025

## Indice 

1. [Descripción del Proyecto](#descripción-del-proyecto)  
2. [Arquitectura General](#arquitectura-general)  
3. [Fundamentos Teóricos](#fundamentos-teóricos)  
   3.1 [REINFORCE y Policy Gradient](#reinforce-y-policy-gradient)  
   3.2 [Modelos de Lenguaje (LLM)](#modelos-de-lenguaje-llm)  
4. [Metodologia](#metodologia)
5. [Entrenamiento y Resultados](#entrenamiento-y-resultados)  
6. [Ejecución del Proyecto](#ejecución-del-proyecto)  


## Descripción del Proyecto

Este proyecto explora el uso de técnicas de **Reinforcement Learning**, especificamente **Policy Gradient** dentro de un entorno simple de **Multi-Armed Bandit**, utilizando **modelos de lenguaje (LLM)** como función de politica.

La idea central es usar un **nanoGPT** para elegir entre múltiples opciones, teniendo dos bandits que entrega recompensas al azar. Cada bandit presenta dos acciones posibles, y las recompensas se asignan de forma estocastica (aleatoria), buscando simular un entorno clásico de RL con recompensas. En lugar de predecir una salida correcta como funciones tradicionales de **NLP**, el nanoGPT deber tomar decisiones cohenrentes según el prompt que recibe. La salida es una acción elegida entre un conjunto de palabras restrigido, y el modelo recibe una recompensa dependiendo de la elección.

El entrenamiento se hace usando el algoritmo **REINFORCE**, donde se calcula el gradiente de la politica y actualiza en función del reward observado.

El objetivo es analizar cómo se comporta la política al **cruzar los contextos** entre los dos bandits, observando si el modelo desarrolla preferencias generalizables o especificas a los prompts que se le pasen.


## Arquitectura General

![Arquitectura](images/arquitectura.png)

**Componentes:**


- `nanoGPTModel`: modelo base de lenguaje
- `Agent`: clase que conecta el modelo con la política
- `allowed_words`: vocabulario restringido para decidir acciones
- `prompt`: entrada textual
- `logits`: salida del modelo para sampling
- `policy_loss`: función de pérdida basada en REINFORCE
## Fundamentos Teóricos
### REINFORCE y Policy Gradient

**REINFORCE** es un algoritmo basado en **Policy Gradient** que se usa en **Reinforcement Learning** para optimizar una política directamente; actualizando y mejorando dicha política en funcíon de las acciones realizadas durante la etapa de entrenamiento, para ajustar las probabilidades de las acciones realizadas en cada estado según las recompensas acumuladas, es decir reforzar las acciones que obtienen mayores recompesas y disminuir la probabilidad de las que no aportan valor.

La política esta parametrizada por una red neuronal, en este caso por el nanoGPT. El objetivo es maximimizar la recompensa esperada
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} R(\tau) \right]
$$

- $\theta$ son los parámetros de la política.

- $\tau$ es una trayectoria (secuencia de estados, acciones y recompensas).

- $R(\tau)$ es la recompensa total obtenida en la trayectoria.

Para actualizar los parámetros $\theta$ usamos la siguiente expresión:
$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_\theta(a_t | s_t) \cdot R_t \right]
$$

- Estado ($s_t$)  Prompt de entrada

- Acción ($a_t$)  Palabra elegida del vocabulario

- Política ($\pi_\theta$) Prediccion de distribiciones de palabras por el nanoGPT

- Recompensa ($R_t$) Valor de recompesa


> En práctica, ya que si se minimiza el loss se maximiza el reward
    
        loss = -log_prob * reward

- Si la accion es buena, loss es negativo, así que al minimizar loss se aumenta la probabilidad de la acciíon

- Si la acción mala, loss es positiva, así que al minimizar, se disminuye la probabilidad de esa acción.

    
### Modelos de Lenguaje (LLM)

Generative Pre-trained Transformer es un modelo de inteligencia artificial generativa, su funcion principal es poder generar texto coherente, contextual y natural significativo a partir de un prompt que se le pase.

Su funcionamiento se base en una arquitectura de redes neuronales llamada **Transformer**, la cual le permite procesar y entender secuencias de texto mediante mecanismos de atencion, por lo que es capaz de capturar relaciones complejas entre palabras y frases, manteniendo el sentido del contexto.


<p align="center">
  <img src="images/arquitecturaGPT.png" alt="Arquitectura-GPT"/>
</p>

> Configuracion nanoGPT (./config/llm.yml)

    Tamaño Embedding: 192
    Numero de Cabeceras: 8
    Numero de Capas: 8
    Dropout: 0.3
    Tasa de aprendizajes (a): 0.0004
   

> Configuracion Estandard GPT2

    Tamaño Embedding: 768
    Numero de Cabeceras: 12
    Number de Capas: 12

## Metodología 

### Fase 1: Desarrollo y Configución de la Arquitectura GPT

#### 1.1 Diseño de Arquitectura Base

Se comenzó implementando una arquitectura base inspirada en GPT-2, pero reducida y adaptada para fines experimentales, enfocada específicamente en el problema de multi-armed bandit. La idea central era tratar cada paso de interacción con el entorno como parte de una secuencia que puede ser modelada con un Transformer autoregresivo. Para la tokenización se utilizó Tiktoken de OpenAI, ya que cuenta con un vocabulario de 50,257 tokens, lo cual permite codificar eficazmente las secuencias de acción y recompensa. El modelo fue entrenado utilizando una estrategia tipo sliding window, propia de los GPT, donde el input era una ventana de contexto formada por pares de acción-recompensa anteriores y el objetivo era predecir la siguiente mejor acción. Esta formulación convierte el problema en una especie de autocompletado inteligente, pero en lugar de predecir palabras, se predicen decisiones óptimas. 

#### 1.2 Diseño del Dataset para Multi-Armed Bandit

Se desarrollo un script especializado para generar un dataset balancedo que permita al modelo aprender las 4 opciones de muestreo requeridas, para ello se creaon preguntas directas de comparación, de preferencias y con contexto de decisión forzada.

#### 1.3 Configuraciones apropiadas al problema

Se exploraron distintas configuraciones para mejorar el desempeño. En particular, se probaron variantes del bloque de antención multi-cabeza, comenzando con 2 y escalando hasta 8 cabezas, con el objetivo de capturar relaciones dentro de las secuencias. Asimismo, se experimento con redes feedforward de entre 2 y 6 capas, ajustando el número de neurones por capa para encontrar un equilibrio de generalizacion, guardandolo en sistema robusto como checkpoints.


### Fase 2 Validacion y Evaluacion Inicial

Para la validacion de las distribuciones de probabilidad,se hizo una extracción de los checkpoints ya entrenados con el fin de poder evaluar el modelo con un prompts de prueba con el fin de poder realizar un analisis de las distribuciones de probabilidad y revisar que la noramlizacion sera correcta. 

$$
P(bean) + P(chickpea) \approx 0.5,P(lentil)+P(rice)
\approx0.5
\\
\sum P(options) \approx 1.0
$$

### Fase 3 Implementacion Agente + Policy Gradient

### 3.1 Implementacion del agente y 

Se procedio con la implementacíon del agente de Reinforcement Learning, se encapsulo la intecciíon entre el nanoGPT y el entorno de bandit. El agente utiliza un tokenizer para procesar los prompts y generar distribuciones de probabilidad de un conjunto limitado de palabras permitidas. A partir de los logits generados por el modelo, se aplica un Softmax para luego seleccionar una accion mediante muestreo estocástico.

### 3.2 Implementacion algoritmo REINFORCE

Se aplico el algoritmo REINFORCE utilizando una función de perdida que dado al multiplicarlo por el logaritmo de la probabilidad de la acción elegida por la recompensa obtenida, permite actualizar la politica directamente.

Se entrenó el modelo durante 50 épocas, alternando entre prompts diseñado especifiacamente para cada bandit. La optimización se realizo usando el optimizador AdamW, con el fin de ajistar los pesos internos del modelo nanoGPT para aumentar la probabilidad de seleccionar las palabras correactas según la recompensa obtenida.


### Limitaciones

- Entrenamiento de políticas con Feedback Humano (RLHF)
- Necesidad de fine-tuning especifico
- DataSet Extenso (5 GB)
- Dependencia del Prompt

## Entrenamiento y Resultados

### Resultados de Validacion 

![validacion](images/validacion.png)

### Entrenamiento nanoGPT

Se oberva que aun con un numero razonable de cabezeras y capas el modelo tiende a sobreajustar rápidamanete, a pesar de ello, el modelo mostró una aprendizaje estable y una capacidad razonable para generalizar.

![nanoGPT](images/llm/losses.png)

### Entrenamiento Policy Gradient

![policy-loss](images/rl/training_loss.png)


## Ejecución del Proyecto

> Estructuctura de Archivos

    |- config
        >llm.yml
    |- data
        >context.txt
    |- models
        |- llm
            |- attention
                >MultiHeadAttention.py
            |- dataloader
                >dataloader.py
                >dataset.py
            |- model
                >feedForward.py
                >gpt_mode.py
        |- rl
            > agent.py
    |- scripts
       >make_training_data.py
       >train_LLM.sh
    |- training
        |- llm
            > constant_trainining.py
            >evaluation.py
            >plotting.py
            >train_loop.py
        |- rl
            >training.py
    |- utils
        >config_loader.py



### Requisitos

    Python 3
    CUDA version 12.2+
    PyTorch
    Tiktoken

### Instalacion

    git clone git@github.com:MiguelCabreraVictoria/research.git


## Licencia



