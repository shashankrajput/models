# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""SSDFeatureExtractor for InceptionV2 features."""
import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from nets import inception_v2

slim = tf.contrib.slim


class SSDInceptionV2FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
    """SSD Feature Extractor using InceptionV2 features."""

    def __init__(self,
                 is_training,
                 depth_multiplier,
                 min_depth,
                 pad_to_multiple,
                 conv_hyperparams,
                 batch_norm_trainable=True,
                 reuse_weights=None):
        """InceptionV2 Feature Extractor for SSD Models.

        Args:
          is_training: whether the network is in training mode.
          depth_multiplier: float depth multiplier for feature extractor.
          min_depth: minimum feature extractor depth.
          pad_to_multiple: the nearest multiple to zero pad the input height and
            width dimensions to.
          conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
          batch_norm_trainable: Whether to update batch norm parameters during
            training or not. When training with a small batch size
            (e.g. 1), it is desirable to disable batch norm update and use
            pretrained batch norm params.
          reuse_weights: Whether to reuse variables. Default is None.
        """
        super(SSDInceptionV2FeatureExtractor, self).__init__(
            is_training, depth_multiplier, min_depth, pad_to_multiple,
            conv_hyperparams, batch_norm_trainable, reuse_weights)

    def preprocess(self, resized_inputs):
        """SSD preprocessing.

        Maps pixel values to the range [-1, 1].

        Args:
          resized_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.

        Returns:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.
        """

        # size = 299
        # with tf.name_scope('resize') as scope:
        #     max_dim = tf.squeeze(tf.maximum(tf.shape(resized_inputs)[1], tf.shape(resized_inputs)[2]))
        #     padded_size = tf.tile(max_dim, tf.constant([2]))
        #     padded_images = tf.image.resize_image_with_crop_or_pad(resized_inputs, max_dim, max_dim)
        #     resized_inputs = tf.image.resize_images(padded_images, [size, size])

        return (2.0 / 255.0) * resized_inputs - 1.0

    def extract_features(self, preprocessed_inputs):
        """Extract features from preprocessed inputs.

        Args:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.

        Returns:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]
        """
        preprocessed_inputs.get_shape().assert_has_rank(4)
        shape_assert = tf.Assert(
            tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                           tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
            ['image size must at least be 33 in both height and width.'])

        feature_map_layout = {
            'from_layer': ['Mixed_4c', 'Mixed_5c', '', '', '', ''],
            'layer_depth': [-1, -1, 512, 256, 256, 128],
        }

        with tf.control_dependencies([shape_assert]):
            with slim.arg_scope(self._conv_hyperparams):
                with tf.variable_scope('InceptionV2',
                                       reuse=self._reuse_weights) as scope:
                    zoomed_inputs = self.si_cnn(preprocessed_inputs)

                    _, image_features = inception_v2.inception_v2_base(
                        ops.pad_to_multiple(zoomed_inputs, self._pad_to_multiple),
                        final_endpoint='Mixed_5c',
                        min_depth=self._min_depth,
                        depth_multiplier=self._depth_multiplier,
                        scope=scope)
                    feature_maps = feature_map_generators.multi_resolution_feature_maps(
                        feature_map_layout=feature_map_layout,
                        depth_multiplier=self._depth_multiplier,
                        min_depth=self._min_depth,
                        insert_1x1_conv=True,
                        image_features=image_features)

        return feature_maps.values()

    def si_cnn(self, preprocessed_inputs):
        batch_size = preprocessed_inputs.get_shape()[0]
        image_size = preprocessed_inputs.get_shape()[1]
        scale = tf.ones([batch_size])

        single_bb = tf.multiply(tf.constant([0, 0, 1, 1]), (image_size - 1))
        single_bb_expanded = tf.expand_dims(single_bb, axis=0)
        square_bb = tf.tile(single_bb_expanded, [batch_size, 1])

        single_bl_bb = tf.multiply(tf.constant([0, 0, 1, 1]), (image_size - 1))
        single_bl_bb_expanded = tf.expand_dims(single_bl_bb, axis=0)
        blackout_bb = tf.cast(tf.tile(single_bl_bb_expanded, [batch_size, 1]), tf.float32)

        num_channels = 3
        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, 3, num_channels], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(preprocessed_inputs, kernel, [1, 4, 4, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[num_channels], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)

        # zooming1
        with tf.name_scope('zoom1') as scope:
            fmap_weights = tf.Variable(tf.truncated_normal([num_channels], dtype=tf.float32,
                                                           stddev=1e-1), name='fmap_weights')
            summed_images = tf.reduce_sum(tf.multiply(conv1, fmap_weights), axis=-1)
            summed_rows = tf.reduce_sum(summed_images, axis=1)
            summed_cols = tf.reduce_sum(summed_images, axis=2)
            curr_size = tf.shape(summed_rows)[-1]
            linspace = tf.cast(tf.range(0, summed_rows.get_shape().as_list()[-1], 1), tf.float32)

            x_means = tf.reduce_sum(tf.multiply(summed_rows, linspace)) / tf.reduce_sum(summed_rows)
            y_means = tf.reduce_sum(tf.multiply(summed_cols, linspace)) / tf.reduce_sum(summed_cols)
            x_vars = tf.reduce_sum(tf.multiply(tf.square(linspace - x_means), summed_rows)) / tf.reduce_sum(summed_rows)
            y_vars = tf.reduce_sum(tf.multiply(tf.square(linspace - y_means), summed_cols)) / tf.reduce_sum(summed_cols)
            x_sds = tf.sqrt(x_vars)
            y_sds = tf.sqrt(y_vars)
            factor = 5
            # If proposed bounding box cuts into blackout region, reduce its size
            x_shift = (x_means + factor * x_sds) - tf.minimum(x_means + factor * x_sds, blackout_bb[:, 3])
            x_means -= x_shift / 2
            x_sds -= x_shift / (2 * factor)
            x_shift = tf.maximum(x_means - factor * x_sds, blackout_bb[:, 1]) - (x_means - factor * x_sds)
            x_means += x_shift / 2
            x_sds -= x_shift / (2 * factor)

            y_shift = (y_means + factor * y_sds) - tf.minimum(y_means + factor * y_sds, blackout_bb[:, 3])
            y_means -= y_shift / 2
            y_sds -= y_shift / (2 * factor)
            y_shift = tf.maximum(y_means - factor * y_sds, blackout_bb[:, 1]) - (y_means - factor * y_sds)
            y_means += y_shift / 2
            y_sds -= y_shift / (2 * factor)

            # Slicing
            max_sds = tf.maximum(x_sds, y_sds)
            y1s = y_means - factor * max_sds
            y2s = y_means + factor * max_sds
            x1s = x_means - factor * max_sds
            x2s = x_means + factor * max_sds

            y1s = tf.expand_dims(y1s, axis=1)
            y2s = tf.expand_dims(y2s, axis=1)
            x1s = tf.expand_dims(x1s, axis=1)
            x2s = tf.expand_dims(x2s, axis=1)

            square_bounding_boxes = tf.concat([y1s, x1s, y2s, x2s], axis=1)
            square_bounding_boxes = square_bounding_boxes + tf.cast(curr_size, tf.float32) / 2 * factor
            # padding so that bounding boxes remain within the limits

            padding_temp = curr_size / 2 * factor
            pad_tensor = [[0, 0], [padding_temp, padding_temp], [padding_temp, padding_temp], [0, 0]]

            padded_images = tf.pad(conv1, pad_tensor, "CONSTANT")

            # Crop images to bounding boxes, and zoom to current size
            cropped_images = tf.image.crop_and_resize(padded_images, square_bounding_boxes, tf.range(batch_size),
                                                      [curr_size, curr_size])

            # Blackout part of image corresponding to original bounding boxes (rather than the current square bounding boxes)
            # For the zoomed image, new center is just the exact center. Padding needed in original scale is (factor*x_sds).
            # Padding needed in new scale is (factor*x_sds) * [(curr_size/2) / (factor*max_sds)]

            curr_size_float_by_2 = tf.cast(curr_size, tf.float32) / 2

            # x_means = curr_size_float_by_2
            # y_means = curr_size_float_by_2

            # Actually like the following line but factor will cancel out
            # x2s = x_means + factor * x_sds * (curr_size / 2) / (factor * max_sds)

            x2s = (1.0 + x_sds / max_sds) * curr_size_float_by_2
            x1s = (1.0 - x_sds / max_sds) * curr_size_float_by_2
            y2s = (1.0 + y_sds / max_sds) * curr_size_float_by_2
            y1s = (1.0 - y_sds / max_sds) * curr_size_float_by_2

            blackout_bb = tf.stack([y1s, x1s, y2s, x2s], axis=1)

            x_masks = tf.cast(tf.logical_xor(tf.sequence_mask(x2s, curr_size), tf.sequence_mask(x1s, curr_size)),
                              tf.float32)
            y_masks = tf.cast(tf.logical_xor(tf.sequence_mask(y2s, curr_size), tf.sequence_mask(y1s, curr_size)),
                              tf.float32)

            y_masks = tf.expand_dims(y_masks, axis=1)
            x_masks = tf.expand_dims(x_masks, axis=2)

            masks = tf.matmul(x_masks, y_masks)
            blackout_images = tf.multiply(cropped_images, tf.expand_dims(masks, axis=-1))

            new_scale = scale * curr_size_float_by_2 / (factor * max_sds)

            #new_square_bb = tf.zeros([batch_size, 4])
            #new_square_bb[:, 1]

            new_square_bb_1 = tf.cast(square_bb[:, 1], tf.float32) + square_bounding_boxes[:, 1] / scale
            new_square_bb_0 = tf.cast(square_bb[:, 0], tf.float32) + square_bounding_boxes[:, 0] / scale
            new_square_bb_3 = tf.cast(square_bb[:, 1], tf.float32) + square_bounding_boxes[:, 3] / scale
            new_square_bb_2 = tf.cast(square_bb[:, 0], tf.float32) + square_bounding_boxes[:, 2] / scale

            new_square_bb = tf.stack([new_square_bb_0,new_square_bb_1,new_square_bb_2,new_square_bb_3], axis=1)

            scale = new_scale
            square_bb = new_square_bb


            blackout_images_resized = tf.image.resize_images(blackout_images, [image_size, image_size])

            blackout_images_processed = self.preprocess(blackout_images_resized)

            # Find better way to do the following logic
            blackout_images_resized_sliced = tf.slice(blackout_images_resized,[0, 0, 0, 0],[-1,-1,-1,num_channels])

            print(blackout_images_resized_sliced)
            print(preprocessed_inputs)


        return blackout_images_resized_sliced
