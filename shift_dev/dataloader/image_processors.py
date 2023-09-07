from typing import Dict, Optional, Union, List

import PIL
import numpy as np
from torch import TensorType
from transformers import SegformerImageProcessor, BatchFeature
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import ImageInput, ChannelDimension, to_numpy_array, infer_channel_dimension_format, \
    PILImageResampling, make_list_of_images, valid_images


class SegformerMultitaskImageProcessor(SegformerImageProcessor):

    def _preprocess_depth(
        self,
        depth: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # All transformations expect numpy arrays.
        depth = to_numpy_array(depth)
        if depth.ndim == 2:
            added_channel_dim = True
            depth = depth[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(depth, num_channels=1)
        depth = self._preprocess(
            image=depth,
            do_reduce_labels=False,
            do_resize=do_resize,
            size=size,
            input_data_format=input_data_format,
            do_rescale=False,
            do_normalize=False,
        )
        # Remove extra channel dimension if added for processing
        if added_channel_dim:
            depth = depth.squeeze(0)
        if data_format is not None:
            depth = to_channel_dimension_format(depth, data_format, input_channel_dim=input_data_format)
        return depth

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        depth_masks: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            segmentation_maps (`ImageInput`, *optional*):
                Segmentation map to preprocess.
            depth_masks ('ImageInput', *optional*):
                Depth masks to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after `resize` is applied.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
                is used for background, and background itself is not included in all classes of a dataset (e.g.
                ADE20k). The background label will be replaced by 255.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels
        resample = resample if resample is not None else self.resample
        size = size if size is not None else self.size
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)
        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)

        if depth_masks is not None:
            depth_masks = make_list_of_images(depth_masks, expected_ndims=2)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if segmentation_maps is not None and not valid_images(segmentation_maps):
            raise ValueError(
                "Invalid segmentation map type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        images = [
            self._preprocess_image(
                image=img,
                do_resize=do_resize,
                resample=resample,
                size=size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for img in images
        ]

        data = {"pixel_values": images}

        if segmentation_maps is not None:
            segmentation_maps = [
                self._preprocess_mask(
                    segmentation_map=segmentation_map,
                    do_reduce_labels=do_reduce_labels,
                    do_resize=do_resize,
                    size=size,
                    input_data_format=input_data_format,
                )
                for segmentation_map in segmentation_maps
            ]
            data["labels"] = segmentation_maps

        if depth_masks is not None:
            depth_masks = [
                self._preprocess_depth(
                    depth=depth_mask,
                    do_resize=do_resize,
                    size=size,
                    input_data_format=input_data_format,
                )
                for depth_mask in depth_masks
            ]
            data["depth_labels"] = depth_masks

        return BatchFeature(data=data, tensor_type=return_tensors)
