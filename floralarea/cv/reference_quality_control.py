# """
# Reference Object Quality Control

# Ensures accurate reference-based calibration by:
# 1. Selecting the most square-like reference when multiple detected
# 2. Handling partial occlusion by using longest side
# 3. Validating reference quality

# For a 7.6cm × 7.6cm square reference (58 cm²):
# - Ideal aspect ratio: 1.0
# - Occluded reference: Use longest side to reconstruct full square
# """

# import numpy as np
# from typing import List, Dict, Optional, Tuple


# class ReferenceQualityControl:
#     """
#     Quality control for reference object detection
    
#     Handles:
#     - Multiple reference detections (choose best)
#     - Partial occlusion (reconstruct full square)
#     - Quality validation (is it square enough?)
#     """
    
#     def __init__(self, 
#                  known_area_cm2: float = 58.0,
#                  min_aspect_ratio: float = 0.7,
#                  max_aspect_ratio: float = 1.43):
#         """
#         Initialize quality control
        
#         Args:
#             known_area_cm2: Known area of the square reference object
#             min_aspect_ratio: Minimum acceptable aspect ratio (width/height)
#             max_aspect_ratio: Maximum acceptable aspect ratio
#                              Default: 0.7-1.43 allows up to 30% occlusion
#         """
#         self.known_area_cm2 = known_area_cm2
#         self.known_side_cm = np.sqrt(known_area_cm2)  # 7.6 cm for 58 cm²
#         self.min_aspect_ratio = min_aspect_ratio
#         self.max_aspect_ratio = max_aspect_ratio
    
#     def calculate_aspect_ratio(self, bbox: List[float]) -> float:
#         """
#         Calculate aspect ratio from bounding box
        
#         Args:
#             bbox: [x1, y1, x2, y2] bounding box coordinates
            
#         Returns:
#             Aspect ratio (width/height)
#         """
#         x1, y1, x2, y2 = bbox
#         width = abs(x2 - x1)
#         height = abs(y2 - y1)
        
#         if height == 0:
#             return float('inf')
        
#         return width / height
    
#     def squareness_score(self, aspect_ratio: float) -> float:
#         """
#         Calculate how "square" a detection is
        
#         Args:
#             aspect_ratio: Width/height ratio
            
#         Returns:
#             Score from 0.0 (not square) to 1.0 (perfect square)
#             1.0 = aspect ratio of 1.0
#             0.0 = very elongated (far from 1.0)
#         """
#         # Distance from ideal aspect ratio of 1.0
#         deviation = abs(aspect_ratio - 1.0)
        
#         # Convert to score (exponential decay)
#         # deviation=0.0 → score=1.0
#         # deviation=0.3 → score=0.74
#         # deviation=0.5 → score=0.61
#         score = np.exp(-deviation * 1.5)
        
#         return float(score)
    
#     def select_best_reference(self, 
#                              detections: List[Dict],
#                              verbose: bool = True) -> Optional[Dict]:
#         """
#         Select the best reference object from multiple detections
        
#         Strategy:
#         1. Filter by aspect ratio (must be square-ish)
#         2. Among valid ones, choose most square
#         3. Prefer larger detections (avoid small false positives)
        
#         Args:
#             detections: List of detection dictionaries with 'bbox', 'area', 'score'
#             verbose: Print selection reasoning
            
#         Returns:
#             Best reference detection, or None if no valid reference
#         """
#         if not detections:
#             return None
        
#         if len(detections) == 1:
#             # Only one detection - validate it
#             det = detections[0]
#             aspect_ratio = self.calculate_aspect_ratio(det['bbox'])
#             squareness = self.squareness_score(aspect_ratio)
            
#             if verbose:
#                 print(f"  Single reference detected:")
#                 print(f"    Aspect ratio: {aspect_ratio:.3f}")
#                 print(f"    Squareness: {squareness:.3f}")
            
#             # Check if acceptable
#             if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
#                 det['aspect_ratio'] = aspect_ratio
#                 det['squareness'] = squareness
#                 return det
#             else:
#                 if verbose:
#                     print(f"    ⚠️  Not square enough! (acceptable: {self.min_aspect_ratio:.2f}-{self.max_aspect_ratio:.2f})")
#                 return None
        
#         # Multiple detections - select best
#         if verbose:
#             print(f"  {len(detections)} reference detections - selecting best:")
        
#         scored_detections = []
        
#         for i, det in enumerate(detections):
#             aspect_ratio = self.calculate_aspect_ratio(det['bbox'])
#             squareness = self.squareness_score(aspect_ratio)
            
#             # Check if valid
#             is_valid = self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
            
#             # Combined score: squareness (70%) + size (30%)
#             # Prefer square + larger detections
#             max_area = max(d['area'] for d in detections)
#             size_score = det['area'] / max_area if max_area > 0 else 0
#             combined_score = (0.7 * squareness) + (0.3 * size_score)
            
#             scored_detections.append({
#                 'detection': det,
#                 'aspect_ratio': aspect_ratio,
#                 'squareness': squareness,
#                 'size_score': size_score,
#                 'combined_score': combined_score,
#                 'is_valid': is_valid
#             })
            
#             if verbose:
#                 status = "✓" if is_valid else "✗"
#                 print(f"    {status} Detection {i+1}: AR={aspect_ratio:.3f}, Square={squareness:.3f}, "
#                       f"Size={size_score:.3f}, Score={combined_score:.3f}")
        
#         # Filter to valid detections
#         valid_detections = [s for s in scored_detections if s['is_valid']]
        
#         if not valid_detections:
#             if verbose:
#                 print(f"    ⚠️  No valid square-like references found!")
#             return None
        
#         # Select best by combined score
#         best = max(valid_detections, key=lambda x: x['combined_score'])
#         best_det = best['detection']
#         best_det['aspect_ratio'] = best['aspect_ratio']
#         best_det['squareness'] = best['squareness']
        
#         if verbose:
#             print(f"    ✓ Selected best reference (score={best['combined_score']:.3f})")
        
#         return best_det
    
#     def handle_occlusion(self,
#                         detection: Dict,
#                         verbose: bool = True) -> Dict:
#         """
#         Handle partial occlusion of reference object
        
#         Strategy:
#         - If aspect ratio != 1.0, reference may be partially occluded
#         - Use longest side to estimate full square area
#         - This prevents underestimating reference area (which would overestimate flower area)
        
#         Args:
#             detection: Detection dictionary with 'bbox' and 'area'
#             verbose: Print occlusion handling
            
#         Returns:
#             Updated detection with corrected area
#         """
#         bbox = detection['bbox']
#         x1, y1, x2, y2 = bbox
#         width = abs(x2 - x1)
#         height = abs(y2 - y1)
        
#         aspect_ratio = detection.get('aspect_ratio', width / height if height > 0 else 1.0)
#         original_area = detection['area']
        
#         # Determine if occlusion handling is needed
#         if 0.95 <= aspect_ratio <= 1.05:
#             # Nearly square - no correction needed
#             if verbose:
#                 print(f"  Reference is square (AR={aspect_ratio:.3f}) - no occlusion correction")
#             detection['occlusion_corrected'] = False
#             detection['corrected_area_pixels'] = original_area
#             return detection
        
#         # Likely occluded - use longest side to reconstruct square
#         longest_side = max(width, height)
#         reconstructed_area = longest_side * longest_side
        
#         correction_factor = reconstructed_area / original_area if original_area > 0 else 1.0
        
#         if verbose:
#             print(f"  Reference appears occluded (AR={aspect_ratio:.3f}):")
#             print(f"    Original area: {original_area:,} pixels")
#             print(f"    Longest side: {longest_side:.1f} pixels")
#             print(f"    Reconstructed square: {reconstructed_area:,} pixels")
#             print(f"    Correction factor: {correction_factor:.2f}x")
        
#         detection['occlusion_corrected'] = True
#         detection['corrected_area_pixels'] = reconstructed_area
#         detection['original_area_pixels'] = original_area
#         detection['correction_factor'] = correction_factor
        
#         return detection
    
#     def process_reference_detections(self,
#                                     detections: List[Dict],
#                                     verbose: bool = True) -> Optional[Dict]:
#         """
#         Complete reference processing pipeline
        
#         1. Select best reference (if multiple)
#         2. Handle occlusion (if needed)
#         3. Return validated, corrected reference
        
#         Args:
#             detections: List of reference detections
#             verbose: Print processing details
            
#         Returns:
#             Best reference with corrected area, or None if no valid reference
#         """
#         if verbose:
#             print("\n📏 Processing reference object...")
        
#         # Step 1: Select best reference
#         best_ref = self.select_best_reference(detections, verbose=verbose)
        
#         if best_ref is None:
#             if verbose:
#                 print("  ❌ No valid reference object found!")
#             return None
        
#         # Step 2: Handle occlusion
#         corrected_ref = self.handle_occlusion(best_ref, verbose=verbose)
        
#         if verbose:
#             print(f"  ✓ Reference validated and corrected")
        
#         return corrected_ref
    
#     def calculate_calibrated_area(self,
#                                  flower_pixels: int,
#                                  reference_detection: Dict) -> float:
#         """
#         Calculate calibrated floral area using corrected reference
        
#         Args:
#             flower_pixels: Total flower area in pixels
#             reference_detection: Processed reference detection with corrected area
            
#         Returns:
#             Calibrated floral area in cm²
#         """
#         # Use corrected reference area if occlusion was handled
#         ref_pixels = reference_detection.get('corrected_area_pixels', 
#                                             reference_detection['area'])
        
#         # Calculate calibrated area
#         area_cm2 = (flower_pixels / ref_pixels) * self.known_area_cm2
        
#         return area_cm2


# # ============================================================================
# # INTEGRATION EXAMPLES
# # ============================================================================

# def example_basic_usage():
#     """Basic usage example"""
#     from sam3_improved import SAM3Improved
    
#     # Initialize
#     sam = SAM3Improved()
#     qc = ReferenceQualityControl(known_area_cm2=58.0)
    
#     # Detect references
#     ref_detections = sam.segment_by_text(
#         "image.jpg",
#         "brown square cardboard",
#         threshold=0.5,
#         mask_threshold=0.7
#     )
    
#     # Process reference with quality control
#     best_ref = qc.process_reference_detections(ref_detections, verbose=True)
    
#     if best_ref:
#         print(f"Reference area (corrected): {best_ref['corrected_area_pixels']:,} pixels")
#         print(f"Squareness score: {best_ref['squareness']:.3f}")
#     else:
#         print("No valid reference found!")


# def example_with_flower_measurement():
#     """Complete measurement example with quality control"""
#     from sam3_improved import SAM3Improved
    
#     sam = SAM3Improved()
#     qc = ReferenceQualityControl(known_area_cm2=58.0)
    
#     # Combined segmentation
#     results = sam.segment_combined(
#         "image.jpg",
#         prompts=["flower", "brown square cardboard"],
#         mask_threshold=0.7
#     )
    
#     flowers = results["flower"]
#     ref_detections = results["brown square cardboard"]
    
#     # Quality control on reference
#     best_ref = qc.process_reference_detections(ref_detections, verbose=True)
    
#     if not best_ref:
#         raise ValueError("No valid reference object detected!")
    
#     # Filter dead flowers
#     alive_flowers, _ = sam.filter_dead_flowers("image.jpg", flowers)
    
#     # Calculate area
#     flower_pixels = sum(f['area'] for f in alive_flowers)
#     area_cm2 = qc.calculate_calibrated_area(flower_pixels, best_ref)
    
#     print(f"\nFinal results:")
#     print(f"  Floral area: {area_cm2:.2f} cm²")
#     print(f"  Reference quality: {best_ref['squareness']:.3f}")
#     print(f"  Occlusion corrected: {best_ref.get('occlusion_corrected', False)}")


# if __name__ == "__main__":
#     print("\n" + "="*70)
#     print("REFERENCE QUALITY CONTROL - USAGE EXAMPLES")
#     print("="*70)
    
#     print("\n1. Basic Usage:")
#     print("   qc = ReferenceQualityControl(known_area_cm2=58.0)")
#     print("   best_ref = qc.process_reference_detections(ref_detections)")
    
#     print("\n2. Quality Metrics:")
#     print("   - Aspect ratio: width/height (ideal = 1.0)")
#     print("   - Squareness: 0.0-1.0 (how square-like)")
#     print("   - Occlusion correction: Uses longest side to reconstruct")
    
#     print("\n3. Selection Strategy:")
#     print("   - If multiple references: Choose most square + largest")
#     print("   - Score = 70% squareness + 30% size")
#     print("   - Reject if aspect ratio outside 0.7-1.43")
    
#     print("\n4. Occlusion Handling:")
#     print("   - If AR ≠ 1.0 → Likely occluded")
#     print("   - Use longest side to estimate full square")
#     print("   - Prevents underestimating reference area")
    
#     print("\n" + "="*70)












"""
Reference Object Quality Control

Ensures accurate reference-based calibration by:
1. Selecting the most square-like reference when multiple detected
2. Handling partial occlusion by using longest side
3. Validating reference quality

For a 7.6cm × 7.6cm square reference (58 cm²):
- Ideal aspect ratio: 1.0
- Occluded reference: Use longest side to reconstruct full square
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class ReferenceQualityControl:
    """
    Quality control for reference object detection
    
    Handles:
    - Multiple reference detections (choose best)
    - Partial occlusion (reconstruct full square)
    - Quality validation (is it square enough?)
    """
    
    def __init__(self, 
                 known_area_cm2: float = 58.0,
                 min_aspect_ratio: float = 0.7,
                 max_aspect_ratio: float = 1.43):
        """
        Initialize quality control
        
        Args:
            known_area_cm2: Known area of the square reference object
            min_aspect_ratio: Minimum acceptable aspect ratio (width/height)
            max_aspect_ratio: Maximum acceptable aspect ratio
                             Default: 0.7-1.43 allows up to 30% occlusion
        """
        self.known_area_cm2 = known_area_cm2
        self.known_side_cm = np.sqrt(known_area_cm2)  # 7.6 cm for 58 cm²
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
    
    def calculate_aspect_ratio(self, bbox: List[float]) -> float:
        """
        Calculate aspect ratio from bounding box
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            Aspect ratio (width/height)
        """
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        if height == 0:
            return float('inf')
        
        return width / height
    
    def squareness_score(self, aspect_ratio: float) -> float:
        """
        Calculate how "square" a detection is
        
        Args:
            aspect_ratio: Width/height ratio
            
        Returns:
            Score from 0.0 (not square) to 1.0 (perfect square)
            1.0 = aspect ratio of 1.0
            0.0 = very elongated (far from 1.0)
        """
        # Distance from ideal aspect ratio of 1.0
        deviation = abs(aspect_ratio - 1.0)
        
        # Convert to score (exponential decay)
        # deviation=0.0 → score=1.0
        # deviation=0.3 → score=0.74
        # deviation=0.5 → score=0.61
        score = np.exp(-deviation * 1.5)
        
        return float(score)
    
    def select_best_reference(self, 
                             detections: List[Dict],
                             verbose: bool = True) -> Optional[Dict]:
        """
        Select the best reference object from multiple detections
        
        Strategy:
        1. Filter by aspect ratio (must be square-ish)
        2. Among valid ones, choose most square
        3. Prefer larger detections (avoid small false positives)
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'area', 'score'
            verbose: Print selection reasoning
            
        Returns:
            Best reference detection, or None if no valid reference
        """
        if not detections:
            return None
        
        if len(detections) == 1:
            # Only one detection - ALWAYS accept it!
            # Squareness is for COMPARISON, not REJECTION
            # Even a partially occluded reference is better than no reference
            det = detections[0]
            aspect_ratio = self.calculate_aspect_ratio(det['bbox'])
            squareness = self.squareness_score(aspect_ratio)
            
            if verbose:
                print(f"  Single reference detected:")
                print(f"    Aspect ratio: {aspect_ratio:.3f}")
                print(f"    Squareness: {squareness:.3f}")
            
            # ALWAYS accept single reference (no rejection based on squareness)
            det['aspect_ratio'] = aspect_ratio
            det['squareness'] = squareness
            
            if verbose:
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    print(f"    ℹ️  Reference appears heavily occluded (AR={aspect_ratio:.3f})")
                    print(f"    ✓ Accepting anyway (only reference available)")
                else:
                    print(f"    ✓ Reference is acceptable")
            
            return det
        
        # Multiple detections - select best using squareness for COMPARISON
        # Note: With multiple candidates, we CAN be selective and reject badly shaped ones
        if verbose:
            print(f"  {len(detections)} reference detections - selecting best:")
        
        scored_detections = []
        
        for i, det in enumerate(detections):
            aspect_ratio = self.calculate_aspect_ratio(det['bbox'])
            squareness = self.squareness_score(aspect_ratio)
            
            # Check if valid
            is_valid = self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
            
            # Combined score: squareness (70%) + size (30%)
            # Prefer square + larger detections
            max_area = max(d['area'] for d in detections)
            size_score = det['area'] / max_area if max_area > 0 else 0
            combined_score = (0.7 * squareness) + (0.3 * size_score)
            
            scored_detections.append({
                'detection': det,
                'aspect_ratio': aspect_ratio,
                'squareness': squareness,
                'size_score': size_score,
                'combined_score': combined_score,
                'is_valid': is_valid
            })
            
            if verbose:
                status = "✓" if is_valid else "✗"
                print(f"    {status} Detection {i+1}: AR={aspect_ratio:.3f}, Square={squareness:.3f}, "
                      f"Size={size_score:.3f}, Score={combined_score:.3f}")
        
        # Filter to valid detections (within acceptable aspect ratio range)
        valid_detections = [s for s in scored_detections if s['is_valid']]
        
        if valid_detections:
            # At least one valid reference - use the best valid one
            best = max(valid_detections, key=lambda x: x['combined_score'])
            if verbose:
                print(f"    ✓ Selected best valid reference (score={best['combined_score']:.3f})")
        else:
            # All references outside acceptable range - use best available anyway
            # Better to have an imperfect reference than no reference!
            best = max(scored_detections, key=lambda x: x['combined_score'])
            if verbose:
                print(f"    ⚠️  All references heavily occluded/distorted")
                print(f"    ✓ Accepting best available (score={best['combined_score']:.3f})")
        
        best_det = best['detection']
        best_det['aspect_ratio'] = best['aspect_ratio']
        best_det['squareness'] = best['squareness']
        
        return best_det
    
    def handle_occlusion(self,
                        detection: Dict,
                        verbose: bool = True) -> Dict:
        """
        Handle partial occlusion of reference object
        
        Strategy:
        - If aspect ratio != 1.0, reference may be partially occluded
        - Use longest side to estimate full square area
        - This prevents underestimating reference area (which would overestimate flower area)
        
        Args:
            detection: Detection dictionary with 'bbox' and 'area'
            verbose: Print occlusion handling
            
        Returns:
            Updated detection with corrected area
        """
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        aspect_ratio = detection.get('aspect_ratio', width / height if height > 0 else 1.0)
        original_area = detection['area']
        
        # Determine if occlusion handling is needed
        if 0.95 <= aspect_ratio <= 1.05:
            # Nearly square - no correction needed
            if verbose:
                print(f"  Reference is square (AR={aspect_ratio:.3f}) - no occlusion correction")
            detection['occlusion_corrected'] = False
            detection['corrected_area_pixels'] = original_area
            return detection
        
        # Likely occluded - use longest side to reconstruct square
        longest_side = max(width, height)
        reconstructed_area = longest_side * longest_side
        
        correction_factor = reconstructed_area / original_area if original_area > 0 else 1.0
        
        if verbose:
            print(f"  Reference appears occluded (AR={aspect_ratio:.3f}):")
            print(f"    Original area: {original_area:,} pixels")
            print(f"    Longest side: {longest_side:.1f} pixels")
            print(f"    Reconstructed square: {reconstructed_area:,} pixels")
            print(f"    Correction factor: {correction_factor:.2f}x")
        
        detection['occlusion_corrected'] = True
        detection['corrected_area_pixels'] = reconstructed_area
        detection['original_area_pixels'] = original_area
        detection['correction_factor'] = correction_factor
        
        return detection
    
    def process_reference_detections(self,
                                    detections: List[Dict],
                                    verbose: bool = True) -> Optional[Dict]:
        """
        Complete reference processing pipeline
        
        1. Select best reference (if multiple)
        2. Handle occlusion (if needed)
        3. Return validated, corrected reference
        
        Args:
            detections: List of reference detections
            verbose: Print processing details
            
        Returns:
            Best reference with corrected area, or None if no valid reference
        """
        if verbose:
            print("\n📏 Processing reference object...")
        
        # Step 1: Select best reference
        best_ref = self.select_best_reference(detections, verbose=verbose)
        
        if best_ref is None:
            if verbose:
                print("  ❌ No valid reference object found!")
            return None
        
        # Step 2: Handle occlusion
        corrected_ref = self.handle_occlusion(best_ref, verbose=verbose)
        
        if verbose:
            print(f"  ✓ Reference validated and corrected")
        
        return corrected_ref
    
    def calculate_calibrated_area(self,
                                 flower_pixels: int,
                                 reference_detection: Dict) -> float:
        """
        Calculate calibrated floral area using corrected reference
        
        Args:
            flower_pixels: Total flower area in pixels
            reference_detection: Processed reference detection with corrected area
            
        Returns:
            Calibrated floral area in cm²
        """
        # Use corrected reference area if occlusion was handled
        ref_pixels = reference_detection.get('corrected_area_pixels', 
                                            reference_detection['area'])
        
        # Calculate calibrated area
        area_cm2 = (flower_pixels / ref_pixels) * self.known_area_cm2
        
        return area_cm2


# ============================================================================
# INTEGRATION EXAMPLES
# ============================================================================

def example_basic_usage():
    """Basic usage example"""
    from sam3_improved import SAM3Improved
    
    # Initialize
    sam = SAM3Improved()
    qc = ReferenceQualityControl(known_area_cm2=58.0)
    
    # Detect references
    ref_detections = sam.segment_by_text(
        "image.jpg",
        "brown square cardboard",
        threshold=0.5,
        mask_threshold=0.7
    )
    
    # Process reference with quality control
    best_ref = qc.process_reference_detections(ref_detections, verbose=True)
    
    if best_ref:
        print(f"Reference area (corrected): {best_ref['corrected_area_pixels']:,} pixels")
        print(f"Squareness score: {best_ref['squareness']:.3f}")
    else:
        print("No valid reference found!")


def example_with_flower_measurement():
    """Complete measurement example with quality control"""
    from sam3_improved import SAM3Improved
    
    sam = SAM3Improved()
    qc = ReferenceQualityControl(known_area_cm2=58.0)
    
    # Combined segmentation
    results = sam.segment_combined(
        "image.jpg",
        prompts=["flower", "brown square cardboard"],
        mask_threshold=0.7
    )
    
    flowers = results["flower"]
    ref_detections = results["brown square cardboard"]
    
    # Quality control on reference
    best_ref = qc.process_reference_detections(ref_detections, verbose=True)
    
    if not best_ref:
        raise ValueError("No valid reference object detected!")
    
    # Filter dead flowers
    alive_flowers, _ = sam.filter_dead_flowers("image.jpg", flowers)
    
    # Calculate area
    flower_pixels = sum(f['area'] for f in alive_flowers)
    area_cm2 = qc.calculate_calibrated_area(flower_pixels, best_ref)
    
    print(f"\nFinal results:")
    print(f"  Floral area: {area_cm2:.2f} cm²")
    print(f"  Reference quality: {best_ref['squareness']:.3f}")
    print(f"  Occlusion corrected: {best_ref.get('occlusion_corrected', False)}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REFERENCE QUALITY CONTROL - USAGE EXAMPLES")
    print("="*70)
    
    print("\n1. Basic Usage:")
    print("   qc = ReferenceQualityControl(known_area_cm2=58.0)")
    print("   best_ref = qc.process_reference_detections(ref_detections)")
    
    print("\n2. Quality Metrics:")
    print("   - Aspect ratio: width/height (ideal = 1.0)")
    print("   - Squareness: 0.0-1.0 (how square-like)")
    print("   - Occlusion correction: Uses longest side to reconstruct")
    
    print("\n3. Selection Strategy:")
    print("   - If multiple references: Choose most square + largest")
    print("   - Score = 70% squareness + 30% size")
    print("   - Reject if aspect ratio outside 0.7-1.43")
    
    print("\n4. Occlusion Handling:")
    print("   - If AR ≠ 1.0 → Likely occluded")
    print("   - Use longest side to estimate full square")
    print("   - Prevents underestimating reference area")
    
    print("\n" + "="*70)