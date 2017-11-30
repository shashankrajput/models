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
"""SSD Meta-architecture definition.

General tensorflow implementation of convolutional Multibox/SSD detection
models.
"""
from abc import abstractmethod

import re
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import box_predictor as bpredictor
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import shape_utils
from object_detection.utils import visualization_utils

slim = tf.contrib.slim


class SSDFeatureExtractor(object):
    """SSD Feature Extractor definition."""

    def __init__(self,
                 is_training,
                 depth_multiplier,
                 min_depth,
                 pad_to_multiple,
                 conv_hyperparams,
                 batch_norm_trainable=True,
                 reuse_weights=None):
        """Constructor.

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
          reuse_weights: whether to reuse variables. Default is None.
        """
        self._is_training = is_training
        self._depth_multiplier = depth_multiplier
        self._min_depth = min_depth
        self._pad_to_multiple = pad_to_multiple
        self._conv_hyperparams = conv_hyperparams
        self._batch_norm_trainable = batch_norm_trainable
        self._reuse_weights = reuse_weights

    @abstractmethod
    def preprocess(self, resized_inputs):
        """Preprocesses images for feature extraction (minus image resizing).

        Args:
          resized_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.

        Returns:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.
        """
        pass

    @abstractmethod
    def extract_features(self, preprocessed_inputs):
        """Extracts features from preprocessed inputs.

        This function is responsible for extracting feature maps from preprocessed
        images.

        Args:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.

        Returns:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]
        """
        pass


class SSDMetaArch(model.DetectionModel):
    """SSD Meta-architecture definition."""

    def __init__(self,
                 is_training,
                 anchor_generator,
                 box_predictor,
                 box_coder,
                 feature_extractor,
                 matcher,
                 region_similarity_calculator,
                 image_resizer_fn,
                 non_max_suppression_fn,
                 score_conversion_fn,
                 classification_loss,
                 localization_loss,
                 classification_loss_weight,
                 localization_loss_weight,
                 normalize_loss_by_num_matches,
                 hard_example_miner,
                 add_summaries=True):
        """SSDMetaArch Constructor.

        TODO: group NMS parameters + score converter into a class and loss
        parameters into a class and write config protos for postprocessing
        and losses.

        Args:
          is_training: A boolean indicating whether the training version of the
            computation graph should be constructed.
          anchor_generator: an anchor_generator.AnchorGenerator object.
          box_predictor: a box_predictor.BoxPredictor object.
          box_coder: a box_coder.BoxCoder object.
          feature_extractor: a SSDFeatureExtractor object.
          matcher: a matcher.Matcher object.
          region_similarity_calculator: a
            region_similarity_calculator.RegionSimilarityCalculator object.
          image_resizer_fn: a callable for image resizing.  This callable always
            takes a rank-3 image tensor (corresponding to a single image) and
            returns a rank-3 image tensor, possibly with new spatial dimensions.
            See builders/image_resizer_builder.py.
          non_max_suppression_fn: batch_multiclass_non_max_suppression
            callable that takes `boxes`, `scores` and optional `clip_window`
            inputs (with all other inputs already set) and returns a dictionary
            hold tensors with keys: `detection_boxes`, `detection_scores`,
            `detection_classes` and `num_detections`. See `post_processing.
            batch_multiclass_non_max_suppression` for the type and shape of these
            tensors.
          score_conversion_fn: callable elementwise nonlinearity (that takes tensors
            as inputs and returns tensors).  This is usually used to convert logits
            to probabilities.
          classification_loss: an object_detection.core.losses.Loss object.
          localization_loss: a object_detection.core.losses.Loss object.
          classification_loss_weight: float
          localization_loss_weight: float
          normalize_loss_by_num_matches: boolean
          hard_example_miner: a losses.HardExampleMiner object (can be None)
          add_summaries: boolean (default: True) controlling whether summary ops
            should be added to tensorflow graph.
        """
        super(SSDMetaArch, self).__init__(num_classes=box_predictor.num_classes)
        self._is_training = is_training

        # Needed for fine-tuning from classification checkpoints whose
        # variables do not have the feature extractor scope.
        self._extract_features_scope = 'FeatureExtractor'

        self._anchor_generator = anchor_generator
        self._box_predictor = box_predictor

        self._box_coder = box_coder
        self._feature_extractor = feature_extractor
        self._matcher = matcher
        self._region_similarity_calculator = region_similarity_calculator

        # TODO: handle agnostic mode and positive/negative class weights
        unmatched_cls_target = None
        unmatched_cls_target = tf.constant([1] + self.num_classes * [0], tf.float32)
        self._target_assigner = target_assigner.TargetAssigner(
            self._region_similarity_calculator,
            self._matcher,
            self._box_coder,
            positive_class_weight=1.0,
            negative_class_weight=1.0,
            unmatched_cls_target=unmatched_cls_target)

        self._classification_loss = classification_loss
        self._localization_loss = localization_loss
        self._classification_loss_weight = classification_loss_weight
        self._localization_loss_weight = localization_loss_weight
        self._normalize_loss_by_num_matches = normalize_loss_by_num_matches
        self._hard_example_miner = hard_example_miner

        self._image_resizer_fn = image_resizer_fn
        self._non_max_suppression_fn = non_max_suppression_fn
        self._score_conversion_fn = score_conversion_fn

        self._anchors = None
        self._add_summaries = add_summaries

    @property
    def anchors(self):

        return None

    def preprocess(self, inputs):
        """Feature-extractor specific preprocessing.

        See base class.

        Args:
          inputs: a [batch, height_in, width_in, channels] float tensor representing
            a batch of images with values between 0 and 255.0.

        Returns:
          preprocessed_inputs: a [batch, height_out, width_out, channels] float
            tensor representing a batch of images.
        Raises:
          ValueError: if inputs tensor does not have type tf.float32
        """
        if inputs.dtype is not tf.float32:
            raise ValueError('`preprocess` expects a tf.float32 tensor')
        with tf.name_scope('Preprocessor'):
            # TODO: revisit whether to always use batch size as the number of parallel
            # iterations vs allow for dynamic batching.
            resized_inputs = tf.map_fn(self._image_resizer_fn,
                                       elems=inputs,
                                       dtype=tf.float32)
            return self._feature_extractor.preprocess(resized_inputs)

    def predict(self, preprocessed_inputs):
        """Predicts unpostprocessed tensors from input tensor.

        This function takes an input batch of images and runs it through the forward
        pass of the network to yield unpostprocessesed predictions.

        A side effect of calling the predict method is that self._anchors is
        populated with a box_list.BoxList of anchors.  These anchors must be
        constructed before the postprocess or loss functions can be called.

        Args:
          preprocessed_inputs: a [batch, height, width, channels] image tensor.

        Returns:
          prediction_dict: a dictionary holding "raw" prediction tensors:
            1) box_encodings: 4-D float tensor of shape [batch_size, num_anchors,
              box_code_dimension] containing predicted boxes.
            2) class_predictions_with_background: 3-D float tensor of shape
              [batch_size, num_anchors, num_classes+1] containing class predictions
              (logits) for each of the anchors.  Note that this tensor *includes*
              background class predictions (at class index 0).
            3) feature_maps: a list of tensors where the ith tensor has shape
              [batch, height_i, width_i, depth_i].
            4) anchors: 2-D float tensor of shape [num_anchors, 4] containing
              the generated anchors in normalized coordinates.
        """
        with tf.variable_scope(None, self._extract_features_scope,
                               [preprocessed_inputs]):
            feature_maps, bounding_boxes = self.si_cnn(
                preprocessed_inputs)

        return bounding_boxes

    def si_cnn(self, preprocessed_inputs):
        batch_size = preprocessed_inputs.get_shape()[0]
        image_size = preprocessed_inputs.get_shape()[1]
        orig_image_num_channels = preprocessed_inputs.get_shape()[3]

        scale = tf.ones([batch_size])

        single_bb = tf.multiply(tf.constant([0, 0, 1, 1]), (image_size - 1))

        single_bb_expanded = tf.expand_dims(single_bb, axis=0)

        square_bb = tf.tile(single_bb_expanded, [batch_size, 1])

        single_bl_bb = tf.multiply(tf.constant([0, 0, 1, 1]), (image_size - 1))
        single_bl_bb_expanded = tf.expand_dims(single_bl_bb, axis=0)
        blackout_bb = tf.cast(tf.tile(single_bl_bb_expanded, [batch_size, 1]), tf.float32)

        num_channels = 4
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

            # below is the original statement with conv1 as the input
            # we are changing conv1 to preprocessed_inputs, so that we apply bounding boxes on the
            # original image so that original image is passed to the next layer.
            # padded_images = tf.pad(conv1, pad_tensor, "CONSTANT")

            conv1 = tf.Print(conv1, [tf.shape(conv1)],
                             message="@@@@@@@@@@@@@@@@@@@@@@@@@@conv1",
                             summarize=100)
            padded_images = tf.pad(preprocessed_inputs, pad_tensor, "CONSTANT")

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

            # new_square_bb = tf.zeros([batch_size, 4])
            # new_square_bb[:, 1]

            new_square_bb_1 = tf.cast(square_bb[:, 1], tf.float32) + square_bounding_boxes[:, 1] / scale
            new_square_bb_0 = tf.cast(square_bb[:, 0], tf.float32) + square_bounding_boxes[:, 0] / scale
            new_square_bb_3 = tf.cast(square_bb[:, 1], tf.float32) + square_bounding_boxes[:, 3] / scale
            new_square_bb_2 = tf.cast(square_bb[:, 0], tf.float32) + square_bounding_boxes[:, 2] / scale

            new_square_bb = tf.stack([new_square_bb_0, new_square_bb_1, new_square_bb_2, new_square_bb_3], axis=1)

            scale = new_scale
            square_bb = new_square_bb

            blackout_images_resized = tf.image.resize_images(blackout_images, [image_size, image_size])

            # preprocessing to normalize pixel values
            blackout_images_processed = self.preprocess(blackout_images_resized)

            # for fixing the shape of tensor as expected in next step in the pipeline
            # Find better way to do the following logic
            blackout_images_resized_sliced = tf.slice(blackout_images_processed, [0, 0, 0, 0],
                                                      [-1, -1, -1, orig_image_num_channels])

        return blackout_images_resized_sliced, square_bb  # TODO: Is blackout_bb the correct bounding box in the original image (no, calculate rectangular bb using

        # both blackout_Bb and square_bb

    def postprocess(self, bounding_boxes):
        """Converts prediction tensors to final detections.

        This function converts raw predictions tensors to final detection results by
        slicing off the background class, decoding box predictions and applying
        non max suppression and clipping to the image window.

        See base class for output format conventions.  Note also that by default,
        scores are to be interpreted as logits, but if a score_conversion_fn is
        used, then scores are remapped (and may thus have a different
        interpretation).

        Args:
          prediction_dict: a dictionary holding prediction tensors with
            1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
              box_code_dimension] containing predicted boxes.
            2) class_predictions_with_background: 3-D float tensor of shape
              [batch_size, num_anchors, num_classes+1] containing class predictions
              (logits) for each of the anchors.  Note that this tensor *includes*
              background class predictions.

        Returns:
          detections: a dictionary containing the following fields
            detection_boxes: [batch, max_detections, 4]
            detection_scores: [batch, max_detections]
            detection_classes: [batch, max_detections]
            detection_keypoints: [batch, max_detections, num_keypoints, 2] (if
              encoded in the prediction_dict 'box_encodings')
            num_detections: [batch]
        Raises:
          ValueError: if prediction_dict does not contain `box_encodings` or
            `class_predictions_with_background` fields.
        """

        with tf.name_scope('Postprocessor'):
            detection_dict = {'detection_boxes': bounding_boxes,
                              'detection_scores': tf.zeros([tf.shape(bounding_boxes)[0], 1]),
                              'detection_classes': tf.zeros([tf.shape(bounding_boxes)[0], 1]),
                              'num_detections': tf.to_float(tf.ones([tf.shape(bounding_boxes)[0]]))}
            return detection_dict

    def loss(self, bounding_boxes, scope=None):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Calling this function requires that groundtruth tensors have been
        provided via the provide_groundtruth function.

        Args:
          prediction_dict: a dictionary holding prediction tensors with
            1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
              box_code_dimension] containing predicted boxes.
            2) class_predictions_with_background: 3-D float tensor of shape
              [batch_size, num_anchors, num_classes+1] containing class predictions
              (logits) for each of the anchors. Note that this tensor *includes*
              background class predictions.
          scope: Optional scope name.

        Returns:
          a dictionary mapping loss keys (`localization_loss` and
            `classification_loss`) to scalar tensors representing corresponding loss
            values.
        """

        with tf.name_scope(scope, 'Loss', [bounding_boxes]):

            gt_box_list = self.groundtruth_lists(fields.BoxListFields.boxes)

            # if self._add_summaries:
            #     self._summarize_input(
            #         self.groundtruth_lists(fields.BoxListFields.boxes), match_list)


            temp_list = []
            for box in gt_box_list:
                default_box = tf.constant([[0,0,100,100]], dtype=tf.float32)
                box_concated = tf.concat([box, default_box], axis=0)
                box_concated = tf.Print(box_concated, [box_concated[0]], message="#########################GroundTruth", summarize=100)
                temp_list.append(tf.squeeze(tf.slice(box_concated, [0, 0], [1, -1], name="kasoorwar")))

            bounding_boxes = tf.Print(bounding_boxes,[bounding_boxes[0]], message="$$$$$$$$$$$$$$$$$$$$$$$$$$$$$OURS:",summarize=100)

            box_list_first_tensor = tf.convert_to_tensor(temp_list)

            box_list_first = box_list.BoxList(box_list_first_tensor)

            bounding_boxes_boxlist = box_list.BoxList(bounding_boxes)

            location_losses = 1 - box_list_ops.iou(
                bounding_boxes_boxlist,
                box_list_first)

            localization_loss = tf.reduce_sum(location_losses)
            classification_loss = tf.zeros_like(localization_loss)

            # Optionally normalize by number of positive matches
            normalizer = tf.constant(1.0, dtype=tf.float32)

            with tf.name_scope('localization_loss'):
                localization_loss = ((self._localization_loss_weight / normalizer) *
                                     localization_loss)

            loss_dict = {
                'localization_loss': localization_loss,
                'classification_loss': localization_loss
            }
        return loss_dict

    def _summarize_anchor_classification_loss(self, class_ids, cls_losses):
        positive_indices = tf.where(tf.greater(class_ids, 0))
        positive_anchor_cls_loss = tf.squeeze(
            tf.gather(cls_losses, positive_indices), axis=1)
        visualization_utils.add_cdf_image_summary(positive_anchor_cls_loss,
                                                  'PositiveAnchorLossCDF')
        negative_indices = tf.where(tf.equal(class_ids, 0))
        negative_anchor_cls_loss = tf.squeeze(
            tf.gather(cls_losses, negative_indices), axis=1)
        visualization_utils.add_cdf_image_summary(negative_anchor_cls_loss,
                                                  'NegativeAnchorLossCDF')

    def _assign_targets(self, groundtruth_boxes_list, groundtruth_classes_list,
                        groundtruth_keypoints_list=None):
        """Assign groundtruth targets.

        Adds a background class to each one-hot encoding of groundtruth classes
        and uses target assigner to obtain regression and classification targets.

        Args:
          groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
            containing coordinates of the groundtruth boxes.
              Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]
              format and assumed to be normalized and clipped
              relative to the image window with y_min <= y_max and x_min <= x_max.
          groundtruth_classes_list: a list of 2-D one-hot (or k-hot) tensors of
            shape [num_boxes, num_classes] containing the class targets with the 0th
            index assumed to map to the first non-background class.
          groundtruth_keypoints_list: (optional) a list of 3-D tensors of shape
            [num_boxes, num_keypoints, 2]

        Returns:
          batch_cls_targets: a tensor with shape [batch_size, num_anchors,
            num_classes],
          batch_cls_weights: a tensor with shape [batch_size, num_anchors],
          batch_reg_targets: a tensor with shape [batch_size, num_anchors,
            box_code_dimension]
          batch_reg_weights: a tensor with shape [batch_size, num_anchors],
          match_list: a list of matcher.Match objects encoding the match between
            anchors and groundtruth boxes for each image of the batch,
            with rows of the Match objects corresponding to groundtruth boxes
            and columns corresponding to anchors.
        """
        groundtruth_boxlists = [
            box_list.BoxList(boxes) for boxes in groundtruth_boxes_list
        ]
        groundtruth_classes_with_background_list = [
            tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT')
            for one_hot_encoding in groundtruth_classes_list
        ]
        if groundtruth_keypoints_list is not None:
            for boxlist, keypoints in zip(
                    groundtruth_boxlists, groundtruth_keypoints_list):
                boxlist.add_field(fields.BoxListFields.keypoints, keypoints)
        return target_assigner.batch_assign_targets(
            self._target_assigner, self.anchors, groundtruth_boxlists,
            groundtruth_classes_with_background_list)

    def _summarize_input(self, groundtruth_boxes_list, match_list):
        """Creates tensorflow summaries for the input boxes and anchors.

        This function creates four summaries corresponding to the average
        number (over images in a batch) of (1) groundtruth boxes, (2) anchors
        marked as positive, (3) anchors marked as negative, and (4) anchors marked
        as ignored.

        Args:
          groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
            containing corners of the groundtruth boxes.
          match_list: a list of matcher.Match objects encoding the match between
            anchors and groundtruth boxes for each image of the batch,
            with rows of the Match objects corresponding to groundtruth boxes
            and columns corresponding to anchors.
        """
        num_boxes_per_image = tf.stack(
            [tf.shape(x)[0] for x in groundtruth_boxes_list])
        pos_anchors_per_image = tf.stack(
            [match.num_matched_columns() for match in match_list])
        neg_anchors_per_image = tf.stack(
            [match.num_unmatched_columns() for match in match_list])
        ignored_anchors_per_image = tf.stack(
            [match.num_ignored_columns() for match in match_list])
        tf.summary.scalar('Input/AvgNumGroundtruthBoxesPerImage',
                          tf.reduce_mean(tf.to_float(num_boxes_per_image)))
        tf.summary.scalar('Input/AvgNumPositiveAnchorsPerImage',
                          tf.reduce_mean(tf.to_float(pos_anchors_per_image)))
        tf.summary.scalar('Input/AvgNumNegativeAnchorsPerImage',
                          tf.reduce_mean(tf.to_float(neg_anchors_per_image)))
        tf.summary.scalar('Input/AvgNumIgnoredAnchorsPerImage',
                          tf.reduce_mean(tf.to_float(ignored_anchors_per_image)))

    def _apply_hard_mining(self, location_losses, cls_losses, prediction_dict,
                           match_list):
        """Applies hard mining to anchorwise losses.

        Args:
          location_losses: Float tensor of shape [batch_size, num_anchors]
            representing anchorwise location losses.
          cls_losses: Float tensor of shape [batch_size, num_anchors]
            representing anchorwise classification losses.
          prediction_dict: p a dictionary holding prediction tensors with
            1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
              box_code_dimension] containing predicted boxes.
            2) class_predictions_with_background: 3-D float tensor of shape
              [batch_size, num_anchors, num_classes+1] containing class predictions
              (logits) for each of the anchors.  Note that this tensor *includes*
              background class predictions.
          match_list: a list of matcher.Match objects encoding the match between
            anchors and groundtruth boxes for each image of the batch,
            with rows of the Match objects corresponding to groundtruth boxes
            and columns corresponding to anchors.

        Returns:
          mined_location_loss: a float scalar with sum of localization losses from
            selected hard examples.
          mined_cls_loss: a float scalar with sum of classification losses from
            selected hard examples.
        """
        class_predictions = tf.slice(
            prediction_dict['class_predictions_with_background'], [0, 0,
                                                                   1], [-1, -1, -1])

        decoded_boxes, _ = self._batch_decode(prediction_dict['box_encodings'])
        decoded_box_tensors_list = tf.unstack(decoded_boxes)
        class_prediction_list = tf.unstack(class_predictions)
        decoded_boxlist_list = []
        for box_location, box_score in zip(decoded_box_tensors_list,
                                           class_prediction_list):
            decoded_boxlist = box_list.BoxList(box_location)
            decoded_boxlist.add_field('scores', box_score)
            decoded_boxlist_list.append(decoded_boxlist)
        return self._hard_example_miner(
            location_losses=location_losses,
            cls_losses=cls_losses,
            decoded_boxlist_list=decoded_boxlist_list,
            match_list=match_list)

    def _batch_decode(self, box_encodings):
        """Decodes a batch of box encodings with respect to the anchors.

        Args:
          box_encodings: A float32 tensor of shape
            [batch_size, num_anchors, box_code_size] containing box encodings.

        Returns:
          decoded_boxes: A float32 tensor of shape
            [batch_size, num_anchors, 4] containing the decoded boxes.
          decoded_keypoints: A float32 tensor of shape
            [batch_size, num_anchors, num_keypoints, 2] containing the decoded
            keypoints if present in the input `box_encodings`, None otherwise.
        """
        combined_shape = shape_utils.combined_static_and_dynamic_shape(
            box_encodings)
        batch_size = combined_shape[0]
        tiled_anchor_boxes = tf.tile(
            tf.expand_dims(self.anchors.get(), 0), [batch_size, 1, 1])
        tiled_anchors_boxlist = box_list.BoxList(
            tf.reshape(tiled_anchor_boxes, [-1, 4]))
        decoded_boxes = self._box_coder.decode(
            tf.reshape(box_encodings, [-1, self._box_coder.code_size]),
            tiled_anchors_boxlist)
        decoded_keypoints = None
        if decoded_boxes.has_field(fields.BoxListFields.keypoints):
            decoded_keypoints = decoded_boxes.get_field(
                fields.BoxListFields.keypoints)
            num_keypoints = decoded_keypoints.get_shape()[1]
            decoded_keypoints = tf.reshape(
                decoded_keypoints,
                tf.stack([combined_shape[0], combined_shape[1], num_keypoints, 2]))
        decoded_boxes = tf.reshape(decoded_boxes.get(), tf.stack(
            [combined_shape[0], combined_shape[1], 4]))
        return decoded_boxes, decoded_keypoints

    def restore_map(self, from_detection_checkpoint=True):
        """Returns a map of variables to load from a foreign checkpoint.

        See parent class for details.

        Args:
          from_detection_checkpoint: whether to restore from a full detection
            checkpoint (with compatible variable names) or to restore from a
            classification checkpoint for initialization prior to training.

        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        """
        variables_to_restore = {}
        for variable in tf.global_variables():
            if variable.op.name.startswith(self._extract_features_scope):
                var_name = variable.op.name
                if not from_detection_checkpoint:
                    var_name = (re.split('^' + self._extract_features_scope + '/',
                                         var_name)[-1])
                variables_to_restore[var_name] = variable
        return variables_to_restore
