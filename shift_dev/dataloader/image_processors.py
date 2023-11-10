import warnings
from typing import Dict, Optional, Union, List, Any, Tuple

import PIL
import numpy as np
from copy import deepcopy
from torch import TensorType, Tensor
from transformers import SegformerImageProcessor, BatchFeature
from transformers.image_processing_utils import get_size_dict
from transformers.image_transforms import to_channel_dimension_format, corners_to_center_format, resize
from transformers.image_utils import ImageInput, ChannelDimension, to_numpy_array, infer_channel_dimension_format, \
    PILImageResampling, make_list_of_images, valid_images
from transformers.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def normalize_boxes2d(boxes2d: Tensor, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
    boxes = corners_to_center_format(boxes2d)
    boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
    return boxes


class MultitaskImageProcessor(SegformerImageProcessor):

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        do_normalize_boxes2d: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: bool = False,
        class_id_remap: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> None:
        if "reduce_labels" in kwargs:
            warnings.warn(
                "The `reduce_labels` parameter is deprecated and will be removed in a future version. Please use "
                "`do_reduce_labels` instead.",
                FutureWarning,
            )
            do_reduce_labels = kwargs.pop("reduce_labels")

        super().__init__(**kwargs)
        size = size if size is not None else {"height": 512, "width": 512}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_normalize_boxes2d = do_normalize_boxes2d
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_reduce_labels = do_reduce_labels
        self.class_id_remap = class_id_remap

    def resize_annotation(
            self,
            annotation: Dict[str, Any],
            orig_size: Tuple[int, int],
            target_size: Tuple[int, int],
            threshold: float = 0.5,
            resample: PILImageResampling = PILImageResampling.NEAREST,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Resizes an annotation to a target size.

        Args:
            annotation (`Dict[str, Any]`):
                The annotation dictionary.
            orig_size (`Tuple[int, int]`):
                The original size of the input image.
            target_size (`Tuple[int, int]`):
                The target size of the image, as returned by the preprocessing `resize` step.
            threshold (`float`, *optional*, defaults to 0.5):
                The threshold used to binarize the segmentation masks.
            resample (`PILImageResampling`, defaults to `PILImageResampling.NEAREST`):
                The resampling filter to use when resizing the masks.
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
        ratio_height, ratio_width = ratios
        annotation["input_hw"] = target_size

        for key, value in annotation.items():
            if key == "boxes2d":
                boxes = value
                scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height],
                                                  dtype=np.float32)
                annotation["boxes2d"] = scaled_boxes
            elif key == "intrinsics":
                intr_scale = np.array([[ratio_width, 1, ratio_width], [1, ratio_height, ratio_height], [1, 1, 1]])
                annotation["intrinsics"] = annotation["intrinsics"] * intr_scale
            elif key == "masks":
                masks = value[:, None]
                # masks = to_numpy_array(masks)
                if len(masks) > 0:
                    masks = make_list_of_images(masks)
                    masks = [
                        self._preprocess_mask(
                            segmentation_mask=mask,
                            do_reduce_labels=False,
                            do_resize=True,
                            size={"height": target_size[0], "width": target_size[1]},
                            input_data_format=input_data_format,
                        )
                        for mask in masks
                    ]
                    masks = [mask.astype(np.float32)[0] > threshold for mask in masks]
                # masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
                # masks = masks.astype(np.float32)
                # masks = masks[:, 0] > threshold
                annotation["masks"] = masks
            else:
                pass

        return annotation

    def _preprocess_mask(
        self,
        segmentation_mask: ImageInput,
        do_reduce_labels: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        is_instance: bool = False,
    ) -> np.ndarray:
        """Preprocesses a single mask."""
        segmentation_mask = to_numpy_array(segmentation_mask)
        # Add channel dimension if missing - needed for certain transformations
        if segmentation_mask.ndim == 2:
            added_channel_dim = True
            segmentation_mask = segmentation_mask[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_mask, num_channels=1)
        # Remap IDs if remap dict was passed
        if (
                hasattr(self, "class_id_remap")
                and self.class_id_remap is not None
                and len(self.class_id_remap) > 0
                and not is_instance
        ):
            mappings = np.zeros(shape=max(self.class_id_remap.keys()) + 1, dtype=np.int64)
            for class_id in np.unique(segmentation_mask):
                mappings[class_id] = self.class_id_remap.get(class_id, class_id)
            segmentation_mask = mappings[segmentation_mask]
        # reduce zero label if needed
        segmentation_mask = self._preprocess(
            image=segmentation_mask,
            do_reduce_labels=do_reduce_labels,
            do_resize=do_resize,
            resample=PILImageResampling.NEAREST,
            size=size,
            do_rescale=False,
            do_normalize=False,
            input_data_format=input_data_format,
        )
        # Remove extra channel dimension if added for processing
        if added_channel_dim:
            segmentation_mask = segmentation_mask.squeeze(0)
        segmentation_mask = segmentation_mask.astype(np.int64)
        return segmentation_mask

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
        pillow_scale = depth.max()
        depth /= pillow_scale
        depth = self._preprocess(
            image=depth,
            do_reduce_labels=False,
            do_resize=do_resize,
            size=size,
            input_data_format=input_data_format,
            do_rescale=False,
            do_normalize=False,
        )
        depth *= pillow_scale
        # Remove extra channel dimension if added for processing
        if added_channel_dim:
            depth = depth.squeeze(0)
        if data_format is not None:
            depth = to_channel_dimension_format(depth, data_format, input_channel_dim=input_data_format)
        return depth

    def _preprocess_boxes2d(
        self,
        boxes2d: Tensor,
        do_normalize: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        if do_normalize:
            boxes2d = normalize_boxes2d(boxes2d=boxes2d, image_size=image_size)

        return boxes2d

    def preprocess(
        self,
        data_view_dict: dict,
        # images: ImageInput,
        # segmentation_masks: Optional[ImageInput] = None,
        # depth_maps: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        do_normalize_boxes2d: Optional[bool] = None,
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
            data_view_dict (`Dict[str, Any]`):
                Dictionary containing images and annotations produced by SHIFTDataset
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
        do_normalize_boxes2d = do_normalize_boxes2d if do_normalize_boxes2d is not None else self.do_normalize_boxes2d
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels
        resample = resample if resample is not None else self.resample
        size = size if size is not None else self.size
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        data = deepcopy(data_view_dict)
        images = make_list_of_images(data.pop("images"))
        segmentation_masks = None
        if data.get("segmentation_masks", None) is not None:
            segmentation_masks = make_list_of_images(data.pop("segmentation_masks"), expected_ndims=2)

        depth_maps = None
        if data.get("depth_maps", None) is not None:
            depth_maps = make_list_of_images(data.pop("depth_maps"), expected_ndims=2)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if segmentation_masks is not None and not valid_images(segmentation_masks):
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

        output = {"pixel_values": images}

        if segmentation_masks is not None:
            segmentation_masks = [
                self._preprocess_mask(
                    segmentation_mask=segmentation_mask,
                    do_reduce_labels=do_reduce_labels,
                    do_resize=do_resize,
                    size=size,
                    input_data_format=input_data_format,
                )
                for segmentation_mask in segmentation_masks
            ]
            output["segmentation_masks"] = segmentation_masks

        if depth_maps is not None:
            depth_maps = [
                self._preprocess_depth(
                    depth=depth_mask,
                    do_resize=do_resize,
                    size=size,
                    input_data_format=input_data_format,
                )
                for depth_mask in depth_maps
            ]
            output["depth_maps"] = depth_maps

        if do_resize:
            data = self.resize_annotation(
                annotation=data,
                orig_size=data["original_hw"],
                target_size=(size["height"], size["width"]),
                input_data_format=input_data_format,
            )

        if data.get("boxes2d", None) is not None:
            data["boxes2d"] = self._preprocess_boxes2d(
                boxes2d=data.pop("boxes2d"),
                do_normalize=do_normalize_boxes2d,
                image_size=data["input_hw"]
            )

        output.update(data)

        return BatchFeature(data=output, tensor_type=return_tensors)
