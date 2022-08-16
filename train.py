"""
Training loop

"""
import tensorflow as tf
import numpy as np
import generator
import discriminator
from datetime import datetime
import load_data
import pickle
import os

###  Training parameters ####
debugging_set = True # only loads data for January 2018 instead of whole set
num_samples_per_input = 1 # default 6
epochs = 2
batch_size = 4
############

# load small debugging datase
if debugging_set:
  data_path = "./dataset_Jan"
  dataset = tf.data.experimental.load(data_path, compression='GZIP')

else:
  # directory of radar files
  base_directory = '/Users/frederikesmac/important stuff/Uni/MA/Data/data/RAD_NL25_RAC_5min/'
  dataset = load_data.create_dataset(base_directory)

dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)


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
  #check why clip at 24? maybe not relaistic to have higher clips
  weights = tf.clip_by_value(batch_targets, 0.0, 24.0)
  loss = tf.reduce_mean(tf.math.abs(gen_mean - batch_targets) * weights)
  return loss



generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator_obj = generator.Generator(lead_time=90, time_delta=5)
discriminator_obj = discriminator.Discriminator()

def train(epochs, dataset):
  for epoch in range(epochs):
    print("Epoch:", epoch )
    for step, frames in enumerate(dataset):
      print("step:", step)
      train_step(frames)

    # TODO set up checkpoint
    #if (epoch + 1) % 5 == 0:
     # ckpt_save_path = ckpt_manager.save()
      #print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
 #                                                         ckpt_save_path))


@tf.function
def train_step(frames):
  frames = tf.expand_dims(frames, -1)
  batch_inputs, batch_targets =  tf.split(frames,[4,18], axis = 1)
  # force to specify batch_size otherwise graph doesn't build
  #batch_inputs = tf.reshape(batch_inputs,[batch_size,4,256,256,1])
  #batch_targets = tf.reshape(batch_targets,[batch_size,18,256,256,1])
  print("inputs and targets:")
  print(batch_inputs)
  print(batch_targets)
  # calculate samples and targets for discriminator steps
  batch_predictions = generator_obj(batch_inputs)
  gen_sequence = tf.concat([batch_inputs, batch_predictions], axis=1)
  real_sequence = tf.concat([batch_inputs, batch_targets], axis=1)
  concat_inputs = tf.concat([real_sequence, gen_sequence], axis=0)
  # Concatenate the real and generated samples along the batch dimension
  # TODO now it is trained twice on the same data batch, is that correct?
  for i in range(2):
    with tf.GradientTape() as disc_tape:
      concat_outputs = discriminator_obj(concat_inputs)
      # And split back to scores for real and generated samples
      score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
      disc_loss = loss_hinge_disc(score_generated, score_real)
      print("DISC loss:", disc_loss)
      tf.print("disc_loss", disc_loss)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_obj.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_obj.trainable_variables))

    # make generator loss
  with tf.GradientTape() as gen_tape:
    gen_samples = [
      generator_obj(batch_inputs) for _ in range(num_samples_per_input)]
    grid_cell_reg = grid_cell_regularizer(tf.stack(gen_samples, axis=0),
                                          batch_targets)
    gen_sequences = [tf.concat([batch_inputs, x], axis=1) for x in gen_samples]
    gen_disc_loss = loss_hinge_gen(tf.concat(gen_sequences, axis=0))
    gen_loss = gen_disc_loss + 20.0 * grid_cell_reg
    print("GEN_loss", gen_loss)
    tf.print("gen_loss", gen_loss)
  gradients_of_generator = gen_tape.gradient(gen_loss, generator_obj.trainable_variables)
  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_obj.trainable_variables))

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator=generator_obj,
                           discriminator=discriminator_obj,
                           generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/func/%s" % stamp
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True)
#tf.profiler.experimental.start(logdir)
# Call only one tf.function when tracing.
train(epochs, dataset)

with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)
  print("exported writer")
