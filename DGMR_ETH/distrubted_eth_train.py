"""Implementation of train

  Args:
    1: directory of TFR files
    2: directory for tensorboard logs
    3: Optional: if "eager": turns eager execution on for debugging purposes

    example: $HOME/TFR  $HOME/TFR/logs
    example: $HOME/TFR  $HOME/TFR/logs eager """

import pathlib
import tensorflow as tf
import generator
import discriminator
import reading_ETH_data
import sys
from datetime import datetime
import numpy as np


#os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

print("------Import successful------")
print("------Scratch------")

###  Training parameters ####
base_directory =  pathlib.Path(sys.argv[1])
log_dir = sys.argv[2]
tf.config.run_functions_eagerly(False)
if  len(sys.argv) > 3:
  if sys.argv[3] == "eager":
    tf.config.run_functions_eagerly(True)
    mirrored_strategy = tf.distribute.get_strategy()
    print("Running with eager execution")
    print("Running without distribution")

else:
  mirrored_strategy = tf.distribute.MirroredStrategy()
  print("Running with graph execution")

num_samples_per_input = 2 # default 6
epochs = 1
BATCH_SIZE = 4
steps_per_epoch = 200
year = None

############
print("{} Data path of data used".format(base_directory))
print("{} Epochs \n{} Samples per Input\n"
        "{} Batch Size".format( epochs, num_samples_per_input, BATCH_SIZE))
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

print("----------------------------------")



def loss_hinge_disc_dist(score_generated, score_real):
  """Discriminator hinge loss."""
  l1 = tf.nn.relu(1. - score_real)
  # divided by BATCH_SIZE * 2, as every loss score of two values: temporal and spatial
  loss = tf.reduce_sum(l1) * (1. / (BATCH_SIZE*2))
  l2 = tf.nn.relu(1. + score_generated)
  loss +=  tf.reduce_sum(l2) * (1. / (BATCH_SIZE*2))
  return loss


def loss_hinge_gen_dist(score_generated):
  """Generator hinge loss."""
  loss = -tf.reduce_sum(score_generated) *(1. / (BATCH_SIZE*2*num_samples_per_input))
  return loss

def grid_cell_regularizer_dist(generated_samples, batch_targets):
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
  loss = tf.reduce_mean(tf.math.abs(gen_mean - batch_targets) * weights,axis=list(range(1, len(batch_targets.shape))) )
  loss = tf.reduce_sum(loss) * (1/BATCH_SIZE)
  return loss



class DGMR():

  def __init__(self,
               generator_obj = generator.Generator(lead_time=90, time_delta=5, strategy = mirrored_strategy),
                discriminator_obj = discriminator.Discriminator(),
                generator_optimizer = tf.keras.optimizers.Adam(1e-4),
                discriminator_optimizer = tf.keras.optimizers.Adam(1e-4),
                epochs = 3):
    self._generator = generator_obj
    self._discriminator = discriminator_obj
    self._gen_op = generator_optimizer
    self._disc_op = discriminator_optimizer
    self._epochs = epochs
    print("in scope")
    print(mirrored_strategy.extended.variable_created_in_scope(self._generator._sampler._latent_stack._mini_atten_block._gamma))

  @tf.function
  def distributed_train_step(self, dataset_inputs):
    print("in dis train")
    per_replica_losses = mirrored_strategy.run(self.train_step, args=(dataset_inputs,))
    #return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
     #                      axis=None)

# put tf function here?
  def run(self, train_dataset, test_dataset ):

    test_iterator = iter(test_dataset)
    idx = 0

    for epoch in range(self._epochs):
      dist_iterator = iter(train_dataset)
      print("Epoch:", epoch)
      tf.print("Epoch:", epoch)

      for step in tf.range(steps_per_epoch):
        tf.print(step, "step")
        print("Step:", step)
        data = dist_iterator.get_next_as_optional()
        if not data.has_value():
          break
        self.distributed_train_step(data.get_value())
        if step %50 == 0:
          self.eval(next(test_iterator), idx)
          idx+=1
        if step == 500:
          break


  def train_step(self, frames):
    frames = tf.expand_dims(frames, -1)
    radar_frames = frames[0]
    print("Dimension of frames", radar_frames.shape)

    eth_frames = frames[1]
    radar_batch_inputs, batch_targets = tf.split(radar_frames, [4, 18], axis=1)
    eth_batch_inputs, _ = tf.split(eth_frames, [4, 18], axis=1)
    # TODO now it is trained twice on the same data batch, is that correct?
    for _ in range(1):
      # calculate samples and targets for discriminator steps
      # Concatenate the real and generated samples along the batch dimension
      batch_predictions = self._generator(radar_batch_inputs, eth_batch_inputs)
      gen_sequence = tf.concat([radar_batch_inputs, batch_predictions], axis=1)
      real_sequence = tf.concat([radar_batch_inputs, batch_targets], axis=1)
      concat_inputs = tf.concat([real_sequence, gen_sequence], axis=0)
      for _ in range(2):
        with tf.GradientTape() as disc_tape:
          concat_outputs = self._discriminator(concat_inputs)
          # And split back to scores for real and generated samples
          score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
          disc_loss_dist = loss_hinge_disc_dist(score_generated, score_real)
          with writer.as_default():
            tf.summary.scalar('disc loss', data=disc_loss_dist, step =0)
          print("DISC loss:", disc_loss_dist)
          tf.print("disc_loss", disc_loss_dist)
        gradients_of_discriminator = disc_tape.gradient(disc_loss_dist, self._discriminator.trainable_variables)
        self._disc_op.apply_gradients(zip(gradients_of_discriminator, self._discriminator.trainable_variables))

    # make generator loss
    with tf.GradientTape() as gen_tape:
      gen_samples = [
        self._generator(radar_batch_inputs, eth_batch_inputs) for _ in range(num_samples_per_input)]
      grid_cell_reg_dist = grid_cell_regularizer_dist(tf.stack(gen_samples, axis=0),
                                                      batch_targets)
      gen_sequences = [tf.concat([radar_batch_inputs, x], axis=1) for x in gen_samples]# from here on numpys as tf tensors
      # Excpect error in pseudocode:
      #  gen_disc_loss = loss_hinge_gen(tf.concat(gen_sequences, axis=0))
      # changed to call discriminator on gen_sequence and caluculate loss on this output
      disc_output = [self._discriminator(x) for x in gen_sequences]
      gen_disc_loss_dist = loss_hinge_gen_dist(tf.concat(disc_output, axis=0))
      gen_loss = gen_disc_loss_dist + 20.0 * grid_cell_reg_dist
      print("GEN_loss", gen_loss)
      tf.print("gen_loss", gen_loss)
      #with writer.as_default():
       # tf.summary.scalar('Gen loss', data=gen_loss, step = 0)
    gradients_of_generator = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
    self._gen_op.apply_gradients(zip(gradients_of_generator, self._generator.trainable_variables))

  def eval(self, frames, idx):
      print("eval index", idx)
      frames = tf.expand_dims(frames, -1)
      radar_frames = frames[0]
      eth_frames = frames[1]
      radar_inputs, targets = tf.split(radar_frames, [4, 18], axis=1)
      eth_inputs, _ = tf.split(eth_frames, [4, 18], axis=1)
      inputs, targets = tf.split(frames, [4, 18], axis=1)
      predictions=  self._generator(radar_inputs, eth_inputs)
      gen_sequence = tf.concat([radar_inputs, predictions], axis=1)
      real_sequence = tf.concat([radar_inputs, targets], axis=1)
      with writer.as_default():
        eth = np.reshape(eth_inputs, (-1, 256, 256, 1))
        tf.summary.image("ETH", eth, max_outputs=50, step=idx)
        target = np.reshape(real_sequence, (-1, 256, 256, 1))
        tf.summary.image("batch_target", target, max_outputs=50, step=idx)
        generated = np.reshape(gen_sequence, (-1, 256, 256, 1))
        tf.summary.image("generated", generated, max_outputs=50, step=idx)



# return disc_loss_dist
dataset = reading_ETH_data.read_TFR(base_directory, batch_size=BATCH_SIZE)
dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

test_set =  reading_ETH_data.read_TFR_test(base_directory, batch_size=1, window_shift=40)




stamp = datetime.now().strftime("%m%d-%H%M")
logdir = log_dir + "/func/%s" % stamp + "B" + str(BATCH_SIZE)
writer = tf.summary.create_file_writer(logdir)
#tf.profiler.experimental.start(logdir)
#tf.summary.trace_on(graph=True)
#tf.summary.trace_on(graph=False)

with mirrored_strategy.scope():
  model = DGMR(epochs=epochs)
  model.run(train_dataset=dataset, test_dataset = test_set)

#with writer.as_default():
#  tf.summary.trace_export(
#    name="my_func_trace",
#    step=0,
#    profiler_outdir=logdir)

#tf.profiler.experimental.stop(save=True)

