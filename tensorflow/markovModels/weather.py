import tensorflow as tf
import logging
import tensorflow_probability as tfp 




#Inicializar el modulo de probability que vamos a usar
tfd = tfp.distributions

#Los dias frios estan codificados como 0 y los calurosos como 1
#El primer dia de nuestra secuencia bayesiana tiene un 80% de probabilidades de ser un dia frio
#Un dia frio tiene un 30% de probabilidades de seguirle un dia caliente
#Un dia caliente tiene un 20% de probabilidades de seguirle un dia frio
#Un dia frio se distribuye de forma normal con una media de 0 grados y una desviacion estandar de 5
#Un dia caliente se distribuye de forma normal con una media de 15 grados y una desviacion estandar de 10

initialDistribution = tfd.Categorical(probs = [0.8,0.2])
transitionDistribution = tfd.Categorical(probs = [[0.7,0.3],
                                                    [0.2,0.8]])

observationDistribution = tfd.Normal(loc = [0. , 15.], scale = [5.,10.])  #loc = media. #scale = stdv

#Crear el modelo
model = tfd.HiddenMarkovModel(initial_distribution = initialDistribution,
transition_distribution = transitionDistribution,
observation_distribution = observationDistribution,
num_steps = 200)

mean = model.mean()
mean = mean.numpy()

print("Temperatura media predicha para cada día:")
for i, temp in enumerate(mean):
    print(f"Día {i+1}: {temp:.2f} grados")
