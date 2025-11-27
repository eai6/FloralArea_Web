#!/usr/bin/env python3
"""
Unified Object Analyzer - Measure area, dimensions, and count objects

Builds on proven FloralAreaAnalyzer (R²=0.97) with unified API for:
- Area measurement (flowers, leaves, etc.)
- Width/height measurement (tree diameter, etc.) 
- Object counting (any objects)

Author: Claude & Team
Date: November 2025
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import existing proven components
from floralarea.cv.sam3_huggingface import SAM3HuggingFaceSegmenter as SAM3Improved
from floralarea.cv.reference_quality_control import ReferenceQualityControl
from floralarea.core.image_analyzer import FloralAreaAnalyzer as LegacyFloralAreaAnalyzer


class ImageAnalyzer:
    """
    ImageAnalyzer analyzer for object measurement and counting.
    
    Supports multiple measurement modes:
    - 'area': Measure object area in cm²
    - 'width': Measure object width in cm
    - 'height': Measure object height in cm  
    - 'dimensions': Measure both width and height
    - 'count': Count objects
    
    Example:
        >>> analyzer = ImageAnalyzer()
        >>> 
        >>> # Measure flower area
        >>> result = analyzer.measure('flower.jpg', mode='area', 
        ...                           object_prompt='flower',
        ...                           reference_prompt='reference card')
        >>> print(f"Area: {result['area_cm2']:.1f} cm²")
        >>> 
        >>> # Measure tree diameter
        >>> result = analyzer.measure('tree.jpg', mode='width',
        ...                           object_prompt='tree trunk',
        ...                           reference_prompt='reference tag',
        ...                           reference_dimension_cm=7.6)
        >>> print(f"Diameter: {result['width_cm']:.1f} cm")
        >>> 
        >>> # Count apples
        >>> result = analyzer.measure('orchard.jpg', mode='count',
        ...                           object_prompt='apple')
        >>> print(f"Count: {result['count']}")
    """
    
    def __init__(self, 
                 sam3_model_path: Optional[str] = None,
                 enable_dead_flower_filter: bool = False,
                 dead_flower_saturation_threshold: float = 0.15,
                 verbose: bool = True):
        """
        Initialize unified object analyzer.
        
        Args:
            sam3_model_path: Path to SAM3 model (None = auto-detect)
            enable_dead_flower_filter: Enable dead flower filtering for area mode
            dead_flower_saturation_threshold: Saturation threshold for dead flowers
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.enable_dead_flower_filter = enable_dead_flower_filter
        self.dead_flower_saturation_threshold = dead_flower_saturation_threshold
        
        # Initialize SAM3 with proven optimizations
        if self.verbose:
            print("🔧 Initializing Unified Object Analyzer...")
            
        self.sam = SAM3Improved(model_path=sam3_model_path)
        self._legacy_area_analyzer: Optional[LegacyFloralAreaAnalyzer] = None
        
        if self.verbose:
            print(f"   ✅ SAM3 loaded on device: {self.sam.device}")
            print(f"   ✅ Ready for: area, width, height, dimensions, count")
    
    # ================================================================
    # Main Entry Point
    # ================================================================
    
    def measure(self,
                image_path: str,
                mode: str = 'area',
                object_prompt: str = 'flower',
                reference_prompt: Optional[str] = 'brown square cardboard',
                reference_dimension_cm: float = 7.6,
                reference_area_cm2: Optional[float] = None,
                object_threshold: float = 0.15,
                reference_threshold: float = 0.5,
                mask_threshold: float = 0.7,
                **kwargs) -> Dict[str, Any]:
        """
        Unified measurement interface.
        
        Args:
            image_path: Path to image
            mode: Measurement mode: 'area', 'width', 'height', 'dimensions', 'count'
            object_prompt: Text prompt for object to measure
            reference_prompt: Text prompt for reference object (None for count mode)
            reference_dimension_cm: Reference dimension in cm (side length for square)
            reference_area_cm2: Reference area in cm² (if known, for area mode)
            object_threshold: Confidence threshold for object detection (0.15 optimal from flowers)
            reference_threshold: Starting threshold for reference detection (0.5)
            mask_threshold: Mask confidence threshold (0.7 optimal from flowers)
            **kwargs: Mode-specific parameters
            
        Returns:
            Dictionary with measurement results (format depends on mode)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Measuring: {Path(image_path).name}")
            print(f"Mode: {mode.upper()}")
            print(f"{'='*70}")
        
        # Route to appropriate measurement method
        if mode == 'area':
            return self._measure_area(
                image_path=image_path,
                object_prompt=object_prompt,
                reference_prompt=reference_prompt,
                reference_area_cm2=reference_area_cm2 or (reference_dimension_cm ** 2),
                object_threshold=object_threshold,
                reference_threshold=reference_threshold,
                mask_threshold=mask_threshold,
                **kwargs
            )
        
        elif mode == 'width':
            return self._measure_width(
                image_path=image_path,
                object_prompt=object_prompt,
                reference_prompt=reference_prompt,
                reference_dimension_cm=reference_dimension_cm,
                object_threshold=object_threshold,
                reference_threshold=reference_threshold,
                mask_threshold=mask_threshold,
                **kwargs
            )
        
        elif mode == 'height':
            return self._measure_height(
                image_path=image_path,
                object_prompt=object_prompt,
                reference_prompt=reference_prompt,
                reference_dimension_cm=reference_dimension_cm,
                object_threshold=object_threshold,
                reference_threshold=reference_threshold,
                mask_threshold=mask_threshold,
                **kwargs
            )
        
        elif mode == 'dimensions':
            return self._measure_dimensions(
                image_path=image_path,
                object_prompt=object_prompt,
                reference_prompt=reference_prompt,
                reference_dimension_cm=reference_dimension_cm,
                object_threshold=object_threshold,
                reference_threshold=reference_threshold,
                mask_threshold=mask_threshold,
                **kwargs
            )
        
        elif mode == 'count':
            return self._measure_count(
                image_path=image_path,
                object_prompt=object_prompt,
                reference_prompt=reference_prompt,  # Optional for count
                reference_dimension_cm=reference_dimension_cm,
                object_threshold=object_threshold,
                reference_threshold=reference_threshold,
                mask_threshold=mask_threshold,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'area', 'width', 'height', 'dimensions', or 'count'.")
    
    # ================================================================
    # Mode Implementations
    # ================================================================
    
    def _measure_area(self, 
                     image_path: str,
                     object_prompt: str,
                     reference_prompt: str,
                     reference_area_cm2: float,
                     object_threshold: float,
                     reference_threshold: float,
                     mask_threshold: float,
                     **kwargs) -> Dict[str, Any]:
        """
        Measure total area of objects (proven method from flowers, R²=0.97).
        
        This is the production-ready method from FloralAreaAnalyzer.
        """
        if self.verbose:
            print("📊 Area measurement mode (proven: R²=0.97)")
        
        # Lazily materialize a lightweight proxy that reuses this instance's SAM
        if self._legacy_area_analyzer is None:
            legacy = LegacyFloralAreaAnalyzer.__new__(LegacyFloralAreaAnalyzer)
            legacy.camera_model = 'default'
            legacy.output_dir = None
            legacy.verbose = self.verbose
            legacy.enable_dead_flower_filter = self.enable_dead_flower_filter
            legacy.dead_flower_saturation_threshold = self.dead_flower_saturation_threshold
            legacy.sam = self.sam
            legacy._segmentation_mode_logged = False
            self._legacy_area_analyzer = legacy
        else:
            legacy = self._legacy_area_analyzer
            legacy.verbose = self.verbose
            legacy.enable_dead_flower_filter = self.enable_dead_flower_filter
            legacy.dead_flower_saturation_threshold = self.dead_flower_saturation_threshold
            legacy.sam = self.sam
        
        # Always refresh reference QC with current reference area
        legacy.ref_qc = ReferenceQualityControl(known_area_cm2=reference_area_cm2)
        
        legacy_kwargs = {
            'image_path': image_path,
            'flower_prompt': object_prompt,
            'reference_prompt': reference_prompt,
            'reference_area_cm2': reference_area_cm2,
            'flower_threshold': object_threshold,
            'reference_threshold': reference_threshold,
            'mask_threshold': mask_threshold,
        }
        
        passthrough_keys = (
            'flower_distance_m',
            'overlap_iou_threshold',
            'camera_model',
            'output_dir',
            'save_visualizations'
        )
        for key in passthrough_keys:
            if key in kwargs:
                legacy_kwargs[key] = kwargs[key]
        
        result = legacy.measure(**legacy_kwargs)
        result.setdefault('mode', 'area')
        return result
    
    def _measure_width(self,
                      image_path: str,
                      object_prompt: str,
                      reference_prompt: Optional[str],
                      reference_dimension_cm: float,
                      object_threshold: float,
                      reference_threshold: float,
                      mask_threshold: float,
                      method: str = 'oriented_bbox',
                      **kwargs) -> Dict[str, Any]:
        """
        Measure object width in cm (e.g., tree diameter).
        
        Args:
            method: 'bbox', 'max_span', or 'oriented_bbox' (recommended)
        """
        if self.verbose:
            print("📏 Width measurement mode (NEW)")
            print(f"   Method: {method}")
            
        # Step 1: Detect object
        if self.verbose:
            print(f"\n📸 Detecting object: '{object_prompt}'...")
            print(f"   Threshold: {object_threshold}")
            print(f"   Mask threshold: {mask_threshold}")
        
        object_detections = self.sam.segment_by_text(
            image_path,
            object_prompt,
            threshold=object_threshold,
            mask_threshold=mask_threshold
        )
        
        if not object_detections:
            return {
                'error': 'NO_OBJECTS_DETECTED',
                'mode': 'width',
                'width_cm': 0.0,
                'confidence': 'NONE'
            }
        
        if self.verbose:
            print(f"   ✅ Detected {len(object_detections)} object(s)")
        
        # Step 2: Calculate width in pixels for each object
        widths_pixels = []
        for det in object_detections:
            mask = det['mask']
            width_px = self._calculate_width_pixels(mask, method=method)
            widths_pixels.append(width_px)
        
        total_width_pixels = np.mean(widths_pixels)  # Or max, depending on use case
        
        # Step 3: Detect reference (if provided)
        if reference_prompt:
            reference_result = self._detect_reference_1d(
                image_path,
                reference_prompt,
                reference_dimension_cm,
                reference_threshold,
                dimension_type='width'
            )
            
            if reference_result['error']:
                return {
                    'error': 'NO_REFERENCE_DETECTED',
                    'mode': 'width',
                    'width_pixels': total_width_pixels,
                    'confidence': 'LOW'
                }
            
            scale_factor = reference_result['scale_factor_cm_per_pixel']
            width_cm = total_width_pixels * scale_factor
            
            if self.verbose:
                print(f"\n✅ Width measurement complete!")
                print(f"   Width: {width_cm:.2f} cm ({total_width_pixels:.1f} pixels)")
                print(f"   Scale: {scale_factor:.4f} cm/pixel")
            
            return {
                'mode': 'width',
                'width_cm': width_cm,
                'width_pixels': total_width_pixels,
                'num_objects': len(object_detections),
                'method': method,
                'reference_dimension_cm': reference_dimension_cm,
                'reference_dimension_pixels': reference_result['reference_dimension_pixels'],
                'scale_factor_cm_per_pixel': scale_factor,
                'confidence': 'HIGH' if reference_result['confidence'] == 'HIGH' else 'MEDIUM',
                'image_path': image_path
            }
        
        else:
            # No reference - return pixels only
            return {
                'mode': 'width',
                'width_pixels': total_width_pixels,
                'num_objects': len(object_detections),
                'method': method,
                'confidence': 'PIXELS_ONLY',
                'image_path': image_path,
                'note': 'No reference provided - width in pixels only'
            }
    
    def _measure_height(self,
                       image_path: str,
                       object_prompt: str,
                       reference_prompt: Optional[str],
                       reference_dimension_cm: float,
                       object_threshold: float,
                       reference_threshold: float,
                       mask_threshold: float,
                       method: str = 'oriented_bbox',
                       **kwargs) -> Dict[str, Any]:
        """
        Measure object height in cm.
        
        Very similar to width measurement, but measures vertical dimension.
        """
        if self.verbose:
            print("📏 Height measurement mode (NEW)")
            print(f"   Method: {method}")
        
        if self.verbose:
            print(f"\n📸 Detecting object: '{object_prompt}'...")
            print(f"   Threshold: {object_threshold}")
            print(f"   Mask threshold: {mask_threshold}")
        
        object_detections = self.sam.segment_by_text(
            image_path,
            object_prompt,
            threshold=object_threshold,
            mask_threshold=mask_threshold
        )
        
        if not object_detections:
            return {
                'error': 'NO_OBJECTS_DETECTED',
                'mode': 'height',
                'height_cm': 0.0,
                'confidence': 'NONE'
            }
        
        if self.verbose:
            print(f"   ✅ Detected {len(object_detections)} object(s)")
        
        heights_pixels = []
        for det in object_detections:
            mask = det['mask']
            height_px = self._calculate_height_pixels(mask, method=method)
            heights_pixels.append(height_px)
        
        total_height_pixels = np.mean(heights_pixels)
        
        if reference_prompt:
            reference_result = self._detect_reference_1d(
                image_path,
                reference_prompt,
                reference_dimension_cm,
                reference_threshold,
                dimension_type='height'
            )
            
            if reference_result['error']:
                return {
                    'error': 'NO_REFERENCE_DETECTED',
                    'mode': 'height',
                    'height_pixels': total_height_pixels,
                    'confidence': 'LOW'
                }
            
            scale_factor = reference_result['scale_factor_cm_per_pixel']
            height_cm = total_height_pixels * scale_factor
            
            if self.verbose:
                print(f"\n✅ Height measurement complete!")
                print(f"   Height: {height_cm:.2f} cm ({total_height_pixels:.1f} pixels)")
                print(f"   Scale: {scale_factor:.4f} cm/pixel")
            
            return {
                'mode': 'height',
                'height_cm': height_cm,
                'height_pixels': total_height_pixels,
                'num_objects': len(object_detections),
                'method': method,
                'reference_dimension_cm': reference_dimension_cm,
                'reference_dimension_pixels': reference_result['reference_dimension_pixels'],
                'scale_factor_cm_per_pixel': scale_factor,
                'confidence': 'HIGH' if reference_result['confidence'] == 'HIGH' else 'MEDIUM',
                'image_path': image_path
            }
        
        return {
            'mode': 'height',
            'height_pixels': total_height_pixels,
            'num_objects': len(object_detections),
            'method': method,
            'confidence': 'PIXELS_ONLY',
            'image_path': image_path,
            'note': 'No reference provided - height in pixels only'
        }
    
    def _measure_dimensions(self,
                           image_path: str,
                           object_prompt: str,
                           reference_prompt: Optional[str],
                           reference_dimension_cm: float,
                           object_threshold: float,
                           reference_threshold: float,
                           mask_threshold: float,
                           **kwargs) -> Dict[str, Any]:
        """
        Measure full dimensions (width, height, area, aspect ratio, etc.).
        """
        if self.verbose:
            print("📐 Full dimensions mode (NEW)")
        
        method = kwargs.get('method', 'oriented_bbox')
        
        if self.verbose:
            print(f"\n📸 Detecting object: '{object_prompt}' using method '{method}'")
            print(f"   Threshold: {object_threshold} | Mask threshold: {mask_threshold}")
        
        object_detections = self.sam.segment_by_text(
            image_path,
            object_prompt,
            threshold=object_threshold,
            mask_threshold=mask_threshold
        )
        
        if not object_detections:
            return {
                'error': 'NO_OBJECTS_DETECTED',
                'mode': 'dimensions',
                'objects': [],
                'confidence': 'NONE'
            }
        
        objects: List[Dict[str, Any]] = []
        for idx, det in enumerate(object_detections):
            mask = det['mask']
            width_px = self._calculate_width_pixels(mask, method=method)
            height_px = self._calculate_height_pixels(mask, method=method)
            area_px = float(np.count_nonzero(mask))
            aspect_ratio = (width_px / height_px) if height_px else None
            
            objects.append({
                'id': idx,
                'width_pixels': width_px,
                'height_pixels': height_px,
                'area_pixels': area_px,
                'aspect_ratio': aspect_ratio,
                'bbox': det.get('bbox'),
                'confidence': det.get('score')
            })
        
        primary_object = max(objects, key=lambda obj: obj['area_pixels'])
        total_area_pixels = sum(obj['area_pixels'] for obj in objects)
        avg_aspect_ratio = np.mean(
            [obj['aspect_ratio'] for obj in objects if obj['aspect_ratio'] is not None]
        ) if any(obj['aspect_ratio'] is not None for obj in objects) else None
        
        result: Dict[str, Any] = {
            'mode': 'dimensions',
            'method': method,
            'num_objects': len(objects),
            'objects': objects,
            'primary_object_id': primary_object['id'],
            'width_pixels': primary_object['width_pixels'],
            'height_pixels': primary_object['height_pixels'],
            'area_pixels': primary_object['area_pixels'],
            'aspect_ratio': primary_object['aspect_ratio'],
            'avg_aspect_ratio': avg_aspect_ratio,
            'total_area_pixels': total_area_pixels,
            'image_path': image_path
        }
        
        if reference_prompt:
            reference_result = self._detect_reference_1d(
                image_path,
                reference_prompt,
                reference_dimension_cm,
                reference_threshold,
                dimension_type='width'
            )
            
            if reference_result['error']:
                result.update({
                    'error': reference_result['error'],
                    'confidence': 'LOW'
                })
                return result
            
            scale_factor = reference_result['scale_factor_cm_per_pixel']
            scale_factor_sq = scale_factor ** 2
            
            for obj in objects:
                obj['width_cm'] = obj['width_pixels'] * scale_factor
                obj['height_cm'] = obj['height_pixels'] * scale_factor
                obj['area_cm2'] = obj['area_pixels'] * scale_factor_sq
            
            result.update({
                'width_cm': result['width_pixels'] * scale_factor,
                'height_cm': result['height_pixels'] * scale_factor,
                'area_cm2': result['area_pixels'] * scale_factor_sq,
                'scale_factor_cm_per_pixel': scale_factor,
                'reference_dimension_cm': reference_dimension_cm,
                'reference_dimension_pixels': reference_result['reference_dimension_pixels'],
                'confidence': 'HIGH' if reference_result['confidence'] == 'HIGH' else 'MEDIUM'
            })
        
        else:
            result.update({
                'confidence': 'PIXELS_ONLY',
                'note': 'No reference provided - dimensions in pixels only'
            })
        
        if self.verbose:
            print("\n✅ Dimension analysis complete!")
            if 'width_cm' in result:
                print(f"   Width: {result['width_cm']:.2f} cm | Height: {result['height_cm']:.2f} cm")
                print(f"   Area: {result['area_cm2']:.2f} cm²")
            else:
                print(f"   Width: {result['width_pixels']:.1f} px | Height: {result['height_pixels']:.1f} px")
        
        return result
    
    def _measure_count(self,
                      image_path: str,
                      object_prompt: str,
                      reference_prompt: Optional[str],
                      reference_dimension_cm: float,
                      object_threshold: float,
                      reference_threshold: float,
                      mask_threshold: float,
                      min_area_pixels: Optional[int] = None,
                      max_area_pixels: Optional[int] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Count objects in image.
        
        Args:
            min_area_pixels: Minimum object area (filter small detections)
            max_area_pixels: Maximum object area (filter large detections)
        """
        if self.verbose:
            print("🔢 Object counting mode (NEW)")
        
        # Detect all objects
        detections = self.sam.segment_by_text(
            image_path,
            object_prompt,
            threshold=object_threshold,
            mask_threshold=mask_threshold
        )
        
        if not detections:
            return {
                'mode': 'count',
                'count': 0,
                'objects': [],
                'confidence': 'NONE'
            }
        
        # Optional: Filter by size
        if min_area_pixels or max_area_pixels:
            filtered = []
            for det in detections:
                area = np.sum(det['mask'])
                if min_area_pixels and area < min_area_pixels:
                    continue
                if max_area_pixels and area > max_area_pixels:
                    continue
                filtered.append(det)
            detections = filtered
        
        # Extract object info
        objects = []
        for i, det in enumerate(detections):
            objects.append({
                'id': i,
                'area_pixels': int(np.sum(det['mask'])),
                'bbox': det.get('bbox', None),
                'confidence': det.get('score', None)
            })
        
        if self.verbose:
            print(f"\n✅ Count: {len(objects)} objects")
        
        return {
            'mode': 'count',
            'count': len(objects),
            'objects': objects,
            'total_area_pixels': sum(obj['area_pixels'] for obj in objects),
            'confidence': 'HIGH',
            'image_path': image_path
        }
    
    # ================================================================
    # Helper Methods
    # ================================================================
    
    def _calculate_width_pixels(self, mask: np.ndarray, method: str = 'oriented_bbox') -> float:
        """
        Calculate object width in pixels from binary mask.
        
        Args:
            mask: Binary mask (H, W) with 1 for object, 0 for background
            method: 'bbox', 'max_span', or 'oriented_bbox'
            
        Returns:
            Width in pixels
        """
        if method == 'bbox':
            # Simple bounding box width
            x_coords = np.where(np.any(mask, axis=0))[0]
            if len(x_coords) == 0:
                return 0.0
            width = x_coords[-1] - x_coords[0]
            return float(width)
        
        elif method == 'max_span':
            # Maximum horizontal span
            width = np.max(np.sum(mask, axis=0))
            return float(width)
        
        elif method == 'oriented_bbox':
            # Oriented bounding box (rotation-invariant, most accurate)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return 0.0
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contours[0])
            # rect = ((center_x, center_y), (width, height), angle)
            
            # Return smaller dimension (actual width, not length)
            width = min(rect[1])
            return float(width)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_reference_1d(self,
                            image_path: str,
                            reference_prompt: str,
                            reference_dimension_cm: float,
                            reference_threshold: float,
                            dimension_type: str = 'width') -> Dict[str, Any]:
        """
        Detect reference object and calculate 1D scale factor.
        
        Args:
            dimension_type: 'width' or 'height'
            
        Returns:
            Dictionary with scale factor and reference info
        """
        if self.verbose:
            print(f"\n📏 Detecting reference: '{reference_prompt}'...")
        
        # Adaptive threshold search (proven from flowers)
        adaptive_thresholds = [reference_threshold]
        if reference_threshold > 0.1:
            current_threshold = reference_threshold
            while current_threshold > 0.1:
                current_threshold -= 0.05
                current_threshold = max(0.1, current_threshold)
                adaptive_thresholds.append(current_threshold)
        
        ref_detections = []
        for attempt, thresh in enumerate(adaptive_thresholds):
            ref_detections = self.sam.segment_by_text(
                image_path,
                reference_prompt,
                threshold=thresh,
                mask_threshold=0.3  # Loose for reference
            )
            
            if ref_detections:
                if attempt > 0 and self.verbose:
                    print(f"   🔍 Reference found with adaptive threshold {thresh:.2f}")
                break
            elif attempt == 0 and len(adaptive_thresholds) > 1 and self.verbose:
                print(f"   🔍 No reference at {thresh:.2f}, trying lower thresholds...")
        
        if not ref_detections:
            return {
                'error': 'NO_REFERENCE_DETECTED',
                'confidence': 'NONE'
            }
        
        # Use reference quality control from flowers (proven!)
        best_reference = select_best_reference(ref_detections)
        
        if best_reference is None:
            return {
                'error': 'REFERENCE_QUALITY_CHECK_FAILED',
                'confidence': 'LOW'
            }
        
        # Calculate dimension in pixels
        ref_mask = best_reference['mask']
        
        if dimension_type == 'width':
            dimension_pixels = self._calculate_width_pixels(ref_mask, method='oriented_bbox')
        else:  # height
            dimension_pixels = self._calculate_height_pixels(ref_mask, method='oriented_bbox')
        
        # Calculate scale factor (cm/pixel)
        scale_factor = reference_dimension_cm / dimension_pixels
        
        if self.verbose:
            print(f"   ✅ Reference: {reference_dimension_cm:.1f} cm = {dimension_pixels:.1f} pixels")
            print(f"   ✅ Scale: {scale_factor:.4f} cm/pixel")
        
        return {
            'error': None,
            'reference_dimension_cm': reference_dimension_cm,
            'reference_dimension_pixels': dimension_pixels,
            'scale_factor_cm_per_pixel': scale_factor,
            'confidence': 'HIGH'
        }
    
    def _calculate_height_pixels(self, mask: np.ndarray, method: str = 'oriented_bbox') -> float:
        """Calculate object height in pixels (similar to width)."""
        if method == 'bbox':
            y_coords = np.where(np.any(mask, axis=1))[0]
            if len(y_coords) == 0:
                return 0.0
            height = y_coords[-1] - y_coords[0]
            return float(height)
        
        elif method == 'max_span':
            height = np.max(np.sum(mask, axis=1))
            return float(height)
        
        elif method == 'oriented_bbox':
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return 0.0
            
            rect = cv2.minAreaRect(contours[0])
            # Return larger dimension (actual height, not width)
            height = max(rect[1])
            return float(height)
        
        else:
            raise ValueError(f"Unknown method: {method}")


# ================================================================
# Backward Compatibility Wrapper
# ================================================================

class FloralAreaAnalyzerV3(UnifiedObjectAnalyzer):
    """
    Backward compatible version using UnifiedObjectAnalyzer.
    
    Example:
        >>> analyzer = FloralAreaAnalyzerV3()
        >>> result = analyzer.measure('flower.jpg')  # Works like V2!
    """
    
    def measure(self, 
                image_path: str,
                flower_prompt: str = 'flower',
                reference_prompt: str = 'reference card',
                flower_threshold: float = 0.15,
                mask_threshold: float = 0.7,
                **kwargs) -> Dict[str, Any]:
        """
        Measure floral area (backward compatible with V2).
        """
        return super().measure(
            image_path=image_path,
            mode='area',
            object_prompt=flower_prompt,
            reference_prompt=reference_prompt,
            object_threshold=flower_threshold,
            mask_threshold=mask_threshold,
            **kwargs
        )


if __name__ == '__main__':
    print("🎯 Unified Object Analyzer")
    print("=" * 70)
    print("\nExample usage:")
    print("""
    from unified_object_analyzer import UnifiedObjectAnalyzer
    
    analyzer = UnifiedObjectAnalyzer()
    
    # Measure area
    result = analyzer.measure('flower.jpg', mode='area', 
                              object_prompt='flower')
    
    # Measure width
    result = analyzer.measure('tree.jpg', mode='width',
                              object_prompt='tree trunk',
                              reference_dimension_cm=7.6)
    
    # Count objects
    result = analyzer.measure('orchard.jpg', mode='count',
                              object_prompt='apple')
    """)
    print("\n" + "=" * 70)
    print("Ready for implementation! 🚀")
