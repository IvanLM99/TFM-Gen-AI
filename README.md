# TFM - Gen-AI



- projeccio amb VAE damunt GAN per veure relacio o quina tranformacio lineal es pot fer
- zifar 10 
- seccio metodolgia mes teorica (vae, gan, gan inversion)

=====================================================================================================================
=====================================================================================================================

Índice de la memoria:


1: Introducción -> lo que se va a hacer, breve 

1.1. Motivación y contexto
- La importancia de los modelos generativos (VAEs, GANs) en el deep learning.
- El papel del espacio latente en la representación de los datos aprendidos por estos modelos.
  
1.2. Planteamiento del problema
- ¿Son los espacios latentes de diferentes modelos (VAE vs. GAN) similares si se entrenan con el mismo dataset?
- La hipótesis del paper inicial (italianos): la posibilidad de una transformación lineal simple entre espacios de los modelos.
  
1.3. Objetivos del TFM
- Objetivo principal: Validar la hipótesis de la transformación lineal entre una VAE y una GAN (primero con MNIST y luego ----)
- Otros objetivos para alcanzar el principal:
- Implementar y entrenar un modelo VAE y un modelo GAN con MNIST.
- Utilizar GAN inversion para obtener los valores latentes de imágenes del dataset MNIST.
- Calcular una transformación lineal que alinee ambos espacios latentes.
- Evaluar la calidad de la proyección.

1.4. Alcance y limitaciones
- El estudio se centra inicialmente en modelos simples (StyleGAN, SVAE etc etc) y en pocos datasets (MNIST, ..., ...)
- Únicamente se calcula una transformación lineal.
  
1.5. Estructura completa de la memoria
- Breve descripción/introducción de los capítulos que se van a comentar a continuación.

  
2: Fundamentos teóricos y State of the Art de modelos generativos

2.1. Modelos generativos

2.2. VAE
- Arquitectura
- Loss function
- 
  
2.3. GAN
- Arquitectura
- Proceso de entrenamiento.

problemas, limitaciones?
  
2.4. Espacio Latente
- 
- 
  
2.5. GAN Inversion
- Qué es, para qué sirve, cómo
- 
- 

    
3: Metodología y diseño experimental -> lo que se ha hecho

3.1. Proceso experimental
- Diagrama de flujo :
- (Imagen Original) -> VAE Encoder -> zvae
- zvae -> transformación Lineal T-> zgan
- zgan -> GAN -> (Imagen Proyectada).
  
3.2. Conjunto de datos y procesamiento
- Descripción del dataset MNIST. 
- Normalización de datos.
- Otros datasets ???
  
3.3. Diseño e implementación de los modelos
- VAE: arquitectura detallada (capas, dimensiones, ...), hiperparámetros de entrenamiento.
- GAN: arquitectura detallada del Generador y Discriminador, hiperparámetros.
- Entorno de desarrollo: PyTorch, librerías principales, linux, GPU, 
  
3.4. Implementación de la proyección del Espacio Latente
- Paso 1: Generación del conjunto de pares (zvae,zgan) mediante inversión.
- Paso 2: Cálculo de la matriz de transformación TT mediante Mínimos Cuadrados.
  
3.5. Métricas de evaluación
- Métricas cuantitativas: error en el espacio eatente (L-MSE, COS similarity, R-MSE, M-MSE).
- Evaluación cualitativa: coherencia semántica en las imágenes generadas.

  
4: Análisis de resultados (primero MNIST, luego --)

4.1. Resultados entrenamiento de los modelos 
- Resultados del entrenamiento de la VAE (loss plot, calidad de reconstrucción).
- Resultados del entrenamiento de la GAN (loss plot, calidad de imágenes generadas).
- Rendimiento del proceso de GAN inversion.
  
4.2. Visualización de los espacios latentes =
  
4.3. Resultados de proyección lineal
- Resultados visuales (comparativa de imágenes: original, reconstruida VAE, invertida GAN, proyectada GAN).
- Análisis de las métricas cuantitativas obtenidas.
-
  
5: Discusión y propuesta de trabajo futuro

5.1. Interpretación de los resultados obtenidos
- 
- 
  




