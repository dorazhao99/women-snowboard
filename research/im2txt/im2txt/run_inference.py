# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import glob
import sys
import json

import tensorflow as tf

sys.path.append('./im2txt/')
sys.path.append('.')
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("dump_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("beam_size", "3", "Beam size for beam search.")
tf.flags.DEFINE_boolean("use_nn", "False", "Whether or not to use nn.")
tf.flags.DEFINE_string("pickle_file", "", "Name of file to save data to.")
tf.flags.DEFINE_string("train_data_dir", "", "Directory with training images.")
tf.flags.DEFINE_string("caption_path", "", "Filepath with captions.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(), FLAGS.checkpoint_path)
  g.finalize()
  print('Test')	
  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = glob.glob(FLAGS.input_files)
  print(filenames)	
  # for file_pattern in FLAGS.input_files.split(","):
  #   filenames.extend(tf.gfile.Glob(file_pattern))
  # tf.logging.info("Running caption generation on %d files matching %s",
  #                 len(filenames), FLAGS.input_files)
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    beam_size = int(FLAGS.beam_size)
    generator = caption_generator.CaptionGenerator(model, vocab, beam_size=beam_size)
    caption_dicts = [] 
    for i, filename in enumerate(filenames):
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      if FLAGS.use_nn:
        captions = generator.consensus_NN(sess, image, FLAGS.caption_path, FLAGS.train_data_dir, FLAGS.pickle_file)
      else:
        captions = generator.beam_search(sess, image)
#      image_id = int(filename.split('_')[-1].split('.')[0])
      image_id = int(filename.split('/')[-1].split('.')[0])
      if FLAGS.use_nn:
        sentence = captions
      else:
        sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
        if sentence[-1] == '.':
          sentence = sentence[:-1]
        sentence = " ".join(sentence)
        sentence += '.'
      caption_dict = {'caption': sentence, 'image_id': image_id, 'filename': filename }
      caption_dicts.append(caption_dict)
      if i % 10 == 0:
          sys.stdout.write('\n%d/%d: (img %d) %s' %(i, len(filenames), image_id, sentence))
   
    with open(FLAGS.dump_file, 'w') as outfile:
      json.dump(caption_dicts, outfile)
#      print("Captions for image %s:" % os.path.basename(filename))
#      for i, caption in enumerate(captions):
#        # Ignore begin and end words.
#        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
#        sentence = " ".join(sentence)
#        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


if __name__ == "__main__":
  tf.app.run()
