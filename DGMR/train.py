"""Implementation of train

  Args:
    1: directory of TFR files
    2: year of data that should be used (right now only works for single year)
    3: if "eager": turns eager execution on for debugging purposes

    example: $HOME/TFR 2009
    example: $HOME/TFR 2009 eager """

import pathlib
import tensorflow as tf
import generator
import discriminator
import reading_data
import sys
from datetime import datetime
from PIL import Image
import numpy as np

print("------Import successful------")

###  Training parameters ####
base_directory =  pathlib.Path(sys.argv[1])
year = sys.argv[2]
tf.config.run_functions_eagerly(False)
if  len(sys.argv) > 2:
  if sys.argv[3] == "eager":
    tf.config.run_functions_eagerly(True)
    print("Running with eager execution")
else: print("Running with graph execution")

num_samples_per_input = 2 # default 6
epochs = 1
batch_size = 1
print_image = True

############
print("{} Data path\n{} year of data used".format(base_directory, year))
print("{} Epochs \n{} Samples per Input\n"
        "{} Batch Size".format( epochs, num_samples_per_input, batch_size))
print("----------------------------------")

def loss_hinge_disc(score_generated, score_real):
  """Discriminator hinge loss."""
  l1 = tf.nn.relu(1. - score_real)
  loss = tf.reduce_mean(l1)
  l2 = tf.nn.relu(1. + score_generated)
  loss += tf.reduce_mean(l2)
  return loss

def loss_hinge_gen(score_generated):
  """Generator hinge loss."""
  loss = -tf.reduce_mean(score_generated)
  return loss

def grid_cell_regularizer(generated_samples, batch_targets):
  """Grid cell regularizer.

  Args:
    generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
    batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

  Returns:
    loss: A tensor of shape [batch_size].
  """
  gen_mean = tf.reduce_mean(generated_samples, axis=0)
  # TODO check why clip at 24? maybe not relaistic to have higher cells
  weights = tf.clip_by_value(batch_targets, 0.0, 24.0)
  loss = tf.reduce_mean(tf.math.abs(gen_mean - batch_targets) * weights)
  return loss


class DGMR():

  def __init__(self,
               generator_obj = generator.Generator(lead_time=90, time_delta=5),
                discriminator_obj = discriminator.Discriminator(),
                generator_optimizer = tf.keras.optimizers.Adam(1e-4),
                discriminator_optimizer = tf.keras.optimizers.Adam(1e-4),
                epochs = 3):

    self._generator = generator_obj
    self._discriminator = discriminator_obj
    self._gen_op = generator_optimizer
    self._disc_op = discriminator_optimizer
    self._epochs = epochs

  def run(self, dataset):
    for epoch in range(self._epochs):
      print("Epoch:", epoch)
      for step, frames in enumerate(dataset):
        tf.print(step, "step")
        print("Step:", step)
        self.train_step(frames)


  @tf.function
  def train_step(self, frames):
    print("Dimension of frames", frames.shape)
    frames = tf.expand_dims(frames, -1)
    batch_inputs, batch_targets = tf.split(frames, [4, 18], axis=1)
    # TODO now it is trained twice on the same data batch, is that correct?
    for _ in range(2):
      # calculate samples and targets for discriminator steps
      # Concatenate the real and generated samples along the batch dimension
      batch_predictions = self._generator(batch_inputs)
      gen_sequence = tf.concat([batch_inputs, batch_predictions], axis=1)
      real_sequence = tf.concat([batch_inputs, batch_targets], axis=1)
      concat_inputs = tf.concat([real_sequence, gen_sequence], axis=0)
      with tf.GradientTape() as disc_tape:
        concat_outputs = self._discriminator(concat_inputs)
        # And split back to scores for real and generated samples
        score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
        disc_loss = loss_hinge_disc(score_generated, score_real)
        tf.summary.scalar('disc loss', data=disc_loss)
        print("DISC loss:", disc_loss)
        tf.print("disc_loss", disc_loss)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.trainable_variables)
      self._disc_op.apply_gradients(zip(gradients_of_discriminator, self._discriminator.trainable_variables))

    # make generator loss
    with tf.GradientTape() as gen_tape:
      gen_samples = [
        self._generator(batch_inputs) for _ in range(num_samples_per_input)]
      grid_cell_reg = grid_cell_regularizer(tf.stack(gen_samples, axis=0),
                                            batch_targets)
      gen_sequences = [tf.concat([batch_inputs, x], axis=1) for x in gen_samples]
      # this doesn't make sense, shouldn't we sum over the prediction of the discriminator?
      # so call disc from every sample in gen_samples and then call loss_hing_gen of outputs
      gen_disc_loss = loss_hinge_gen(tf.concat(gen_sequences, axis=0))
      ###
      gen_loss = gen_disc_loss + 20.0 * grid_cell_reg
      print("GEN_loss", gen_loss)
      tf.print("gen_loss", gen_loss)
    gradients_of_generator = gen_tape.gradient(gen_loss,  self._generator.trainable_variables)
    self._gen_op.apply_gradients(zip(gradients_of_generator,  self._generator.trainable_variables))


dataset = reading_data.read_TFR(base_directory, year=year, batch_size=batch_size)
model = DGMR(epochs=epochs)
model.run(dataset=dataset)

 # TODO add eraly stopping


