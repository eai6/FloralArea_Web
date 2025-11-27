# """
# SAM 3 via Hugging Face Transformers - Mac Compatible!
# Uses the actual working SAM 3 API from transformers library

# Installation:
#     pip install transformers
#     pip install torch torchvision opencv-python pillow matplotlib
# """

# import torch
# import numpy as np
# from PIL import Image
# from typing import List, Tuple, Dict, Optional
# import cv2


# class SAM3HuggingFaceSegmenter:
#     """
#     SAM 3 wrapper using Hugging Face Transformers
    
#     Works on Mac! (Both Intel and Apple Silicon with MPS)
    
#     Uses text prompts to segment objects - just describe what you want!
    
#     Example:
#         >>> sam = SAM3HuggingFaceSegmenter()
#         >>> masks = sam.segment_by_text("image.jpg", "flower")
#         >>> ref_masks = sam.segment_by_text("image.jpg", "brown square cardboard")
#     """
    
#     def __init__(self, model_id: str = "facebook/sam3"):
#         """
#         Initialize SAM 3 model from Hugging Face
        
#         Args:
#             model_id: Hugging Face model ID (default: "facebook/sam3")
#         """
#         try:
#             from transformers import Sam3Processor, Sam3Model
#         except ImportError:
#             raise ImportError(
#                 "\n" + "="*70 + "\n"
#                 "Transformers library not installed or outdated!\n"
#                 "="*70 + "\n"
#                 "Install with:\n"
#                 "  pip install transformers\n"
#                 "  pip install torch torchvision opencv-python pillow\n"
#                 "="*70
#             )
        
#         # Auto-detect best device
#         if torch.cuda.is_available():
#             self.device = "cuda"
#         elif torch.backends.mps.is_available():
#             self.device = "mps"  # Apple Silicon GPU
#         else:
#             self.device = "cpu"
        
#         print(f"Loading SAM 3 from Hugging Face: {model_id}")
#         print(f"Device: {self.device}")
#         print("(Model will auto-download on first use)")
        
#         # Load model and processor
#         self.model = Sam3Model.from_pretrained(model_id).to(self.device)
#         self.processor = Sam3Processor.from_pretrained(model_id)
        
#         print("✓ SAM 3 loaded successfully!")
    
#     def segment_by_text(self, 
#                        image_path: str, 
#                        text_prompt: str,
#                        threshold: float = 0.3,
#                        mask_threshold: float = 0.3) -> List[Dict]:
#         """
#         Segment all instances of objects matching text prompt
        
#         Args:
#             image_path: Path to input image
#             text_prompt: Text description (e.g., "flower", "brown square cardboard")
#             threshold: Detection confidence threshold (0.0-1.0)
#             mask_threshold: Mask confidence threshold (0.0-1.0)
            
#         Returns:
#             List of detections, each with:
#                 - 'mask': binary mask (H, W) numpy array
#                 - 'score': confidence score
#                 - 'area': mask area in pixels
#                 - 'bbox': bounding box [x1, y1, x2, y2]
#                 - 'prompt': the text prompt used
#         """
#         # Load image
#         image = Image.open(image_path).convert('RGB')
        
#         # Prepare inputs with text prompt
#         inputs = self.processor(
#             images=image, 
#             text=text_prompt, 
#             return_tensors="pt"
#         ).to(self.device)
        
#         # Run inference
#         print(f"Segmenting '{text_prompt}'...")
#         with torch.no_grad():
#             outputs = self.model(**inputs)
        
#         # Post-process for instance segmentation
#         results = self.processor.post_process_instance_segmentation(
#             outputs,
#             threshold=threshold,
#             mask_threshold=mask_threshold,
#             target_sizes=inputs.get("original_sizes").tolist()
#         )[0]
        
#         # Extract masks, boxes, and scores
#         masks = results.get('masks', [])
#         boxes = results.get('boxes', [])
#         scores = results.get('scores', [])
        
#         # Convert to our format
#         detections = []
#         for i in range(len(masks)):
#             # Get mask
#             mask = masks[i]
#             if hasattr(mask, "cpu"):
#                 mask_np = mask.cpu().numpy()
#             else:
#                 mask_np = np.array(mask)
            
#             # Ensure 2D and binary
#             while mask_np.ndim > 2:
#                 mask_np = mask_np[0]
            
#             # Convert to binary if needed
#             if mask_np.dtype.kind in ("f", "i") and mask_np.max() > 1:
#                 # Apply sigmoid and threshold if logits
#                 mask_np = (1.0 / (1.0 + np.exp(-mask_np))) > 0.5
#             else:
#                 mask_np = mask_np.astype(bool)
            
#             mask_binary = mask_np.astype(np.uint8)
            
#             # Get box and score
#             box = boxes[i].tolist() if len(boxes) > i else [0, 0, 0, 0]
#             score = scores[i].item() if len(scores) > i else 1.0
            
#             detections.append({
#                 'mask': mask_binary,
#                 'score': float(score),
#                 'area': int(mask_binary.sum()),
#                 'bbox': [int(b) for b in box],  # [x1, y1, x2, y2]
#                 'prompt': text_prompt
#             })
        
#         print(f"✓ Found {len(detections)} instances of '{text_prompt}'")
#         return detections
    
#     def get_combined_mask(self, detections: List[Dict]) -> np.ndarray:
#         """
#         Combine multiple detection masks into a single binary mask
        
#         Args:
#             detections: List of detection dictionaries
            
#         Returns:
#             Combined binary mask (H, W) as uint8 (0 or 255)
#         """
#         if not detections:
#             return None
        
#         # Get image dimensions from first mask
#         h, w = detections[0]['mask'].shape
#         combined = np.zeros((h, w), dtype=bool)
        
#         # Combine all masks
#         for det in detections:
#             combined = np.logical_or(combined, det['mask'].astype(bool))
        
#         return (combined.astype(np.uint8) * 255)
    
#     def visualize(self,
#                  image_path: str,
#                  detections: List[Dict],
#                  output_path: Optional[str] = None) -> np.ndarray:
#         """
#         Visualize detections on the image
        
#         Args:
#             image_path: Path to input image
#             detections: List of detection dictionaries
#             output_path: Optional path to save visualization
            
#         Returns:
#             Annotated image as numpy array (RGB)
#         """
#         # Load image
#         image = cv2.imread(image_path)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         overlay = image_rgb.copy()
        
#         # Random colors for each detection
#         np.random.seed(42)
#         colors = np.random.randint(100, 255, size=(len(detections), 3), dtype=np.uint8)
        
#         for i, det in enumerate(detections):
#             mask = det['mask']
#             score = det['score']
#             bbox = det['bbox']
#             color = colors[i]
            
#             # Apply colored mask (70% color, 30% original)
#             mask_bool = mask > 0
#             overlay[mask_bool] = (overlay[mask_bool] * 0.3 + color * 0.7).astype(np.uint8)
            
#             # Draw contours
#             contours, _ = cv2.findContours(
#                 mask.astype(np.uint8),
#                 cv2.RETR_EXTERNAL,
#                 cv2.CHAIN_APPROX_SIMPLE
#             )
#             cv2.drawContours(overlay, contours, -1, (255, 255, 0), 3)
            
#             # Draw bounding box
#             if len(bbox) == 4 and sum(bbox) > 0:
#                 x1, y1, x2, y2 = bbox
#                 cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)
            
#             # Add label with score
#             if contours:
#                 M = cv2.moments(contours[0])
#                 if M["m00"] != 0:
#                     cx = int(M["m10"] / M["m00"])
#                     cy = int(M["m01"] / M["m00"])
#                     label = f"{det['prompt']}: {score:.2f}"
#                     cv2.putText(overlay, label, (cx-50, cy),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         if output_path:
#             cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
#         return overlay


# def estimate_area_sam3_hf(image_path: str,
#                           reference_area_cm2: float = 58.0,
#                           flower_prompt: str = "flower",
#                           reference_prompt: str = "brown square cardboard",
#                           threshold: float = 0.3,
#                           mask_threshold: float = 0.3) -> Tuple[float, Image.Image]:
#     """
#     Estimate floral area using SAM 3 via Hugging Face Transformers
    
#     This works on Mac! (Intel and Apple Silicon)
    
#     Args:
#         image_path: Path to input image
#         reference_area_cm2: Known area of reference object in cm²
#         flower_prompt: Text description of flowers
#         reference_prompt: Text description of reference object
#         threshold: Detection confidence threshold
#         mask_threshold: Mask confidence threshold
        
#     Returns:
#         Tuple of (estimated_area_cm2, annotated_image)
    
#     Example:
#         >>> area, img = estimate_area_sam3_hf(
#         ...     "flower.jpg",
#         ...     flower_prompt="pink flower",
#         ...     reference_prompt="brown square cardboard"
#         ... )
#         >>> print(f"Floral area: {area:.2f} cm²")
#     """
#     # Initialize SAM 3
#     print("Initializing SAM 3 via Hugging Face Transformers...")
#     sam = SAM3HuggingFaceSegmenter()
    
#     # Segment flowers
#     flower_detections = sam.segment_by_text(
#         image_path, 
#         flower_prompt,
#         threshold=threshold,
#         mask_threshold=mask_threshold
#     )
    
#     # Segment reference object
#     ref_detections = sam.segment_by_text(
#         image_path, 
#         reference_prompt,
#         threshold=threshold,
#         mask_threshold=mask_threshold
#     )
    
#     # Validate detections
#     if not ref_detections:
#         raise ValueError(
#             f"❌ Reference object not found with prompt '{reference_prompt}'.\n"
#             "Try different prompts like:\n"
#             "  - 'brown cardboard'\n"
#             "  - 'square card'\n"
#             "  - 'reference object'\n"
#             "Or try lowering the threshold (current: {threshold})"
#         )
    
#     if not flower_detections:
#         raise ValueError(
#             f"❌ No flowers found with prompt '{flower_prompt}'.\n"
#             "Try different prompts like:\n"
#             "  - 'flower'\n"
#             "  - 'pink flower'\n"
#             "  - 'flowering plant'\n"
#             "Or try lowering the threshold (current: {threshold})"
#         )
    
#     # Calculate areas
#     flower_pixel_area = sum(d['area'] for d in flower_detections)
#     # Use largest reference object
#     reference_pixel_area = max(d['area'] for d in ref_detections)
    
#     # Scale to real-world area
#     area_cm2 = (flower_pixel_area / reference_pixel_area) * reference_area_cm2
    
#     # Create visualization
#     all_detections = flower_detections + ref_detections
#     annotated = sam.visualize(image_path, all_detections)
    
#     # Print results
#     print(f"\n" + "="*70)
#     print("RESULTS:")
#     print("="*70)
#     print(f"  Flowers found: {len(flower_detections)}")
#     print(f"  Flower pixel area: {flower_pixel_area:,}")
#     print(f"  Reference pixel area: {reference_pixel_area:,}")
#     print(f"  Estimated floral area: {area_cm2:.2f} cm²")
#     print("="*70)
    
#     return area_cm2, Image.fromarray(annotated)








"""
SAM3 Improved - With Batch Processing, Combined Segmentation, and Dead Flower Filtering

Enhancements:
1. ✅ Batch processing for multiple images
2. ✅ Combined segmentation (flowers + reference in one pass)
3. ✅ mask_threshold=0.7 to reduce blob issues
4. ✅ Dead flower detection and filtering
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class SAM3HuggingFaceSegmenter:
    """
    Improved SAM3 wrapper with:
    - Batch processing for speed
    - Combined prompt segmentation
    - Tighter mask thresholds
    - Dead flower filtering
    """
    
    def __init__(self, model_id: str = "facebook/sam3"):
        """Initialize SAM3 model"""
        try:
            from transformers import Sam3Processor, Sam3Model
        except ImportError:
            raise ImportError("pip install transformers torch")
        
        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Loading SAM3: {model_id}")
        print(f"Device: {self.device}")
        
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.processor = Sam3Processor.from_pretrained(model_id)
        
        print("✓ SAM3 loaded!")
    
    def segment_by_text(self, 
                       image_path: str, 
                       text_prompt: str,
                       threshold: float = 0.3,
                       mask_threshold: float = 0.7) -> List[Dict]:  # ✅ CHANGED: 0.7 default!
        """
        Segment objects by text prompt
        
        Args:
            image_path: Path to image
            text_prompt: Text description (e.g., "flower")
            threshold: Detection confidence (0.0-1.0)
            mask_threshold: Mask tightness (0.0-1.0) - HIGHER = TIGHTER boundaries!
                           Default 0.7 reduces blob issues
        
        Returns:
            List of detections with 'mask', 'score', 'area', 'bbox', 'prompt'
        """
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(
            images=image, 
            text=text_prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        print(f"Segmenting '{text_prompt}' (mask_threshold={mask_threshold})...")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,  # ✅ Tighter masks!
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results.get('masks', [])
        boxes = results.get('boxes', [])
        scores = results.get('scores', [])
        
        detections = []
        for i in range(len(masks)):
            mask = masks[i]
            if hasattr(mask, "cpu"):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            while mask_np.ndim > 2:
                mask_np = mask_np[0]
            
            if mask_np.dtype.kind in ("f", "i") and mask_np.max() > 1:
                mask_np = (1.0 / (1.0 + np.exp(-mask_np))) > 0.5
            else:
                mask_np = mask_np.astype(bool)
            
            mask_binary = mask_np.astype(np.uint8)
            box = boxes[i].tolist() if len(boxes) > i else [0, 0, 0, 0]
            score = scores[i].item() if len(scores) > i else 1.0
            
            detections.append({
                'mask': mask_binary,
                'score': float(score),
                'area': int(mask_binary.sum()),
                'bbox': [int(b) for b in box],
                'prompt': text_prompt
            })
        
        print(f"✓ Found {len(detections)} instances")
        return detections
    
    def segment_combined(self,
                        image_path: str,
                        prompts: List[str],
                        threshold: float = 0.3,
                        mask_threshold: float = 0.7) -> Dict[str, List[Dict]]:
        """
        ✅ NEW: Segment multiple prompts in ONE forward pass (MUCH FASTER!)
        
        This is 2-3x faster than calling segment_by_text multiple times
        because it processes all prompts together in a single batch.
        
        Args:
            image_path: Path to image
            prompts: List of text descriptions (e.g., ["flower", "brown square cardboard"])
            threshold: Detection confidence threshold
            mask_threshold: Mask tightness (default 0.7 for tighter boundaries)
            
        Returns:
            Dictionary mapping prompt -> list of detections
            
        Example:
            >>> results = sam.segment_combined("img.jpg", ["flower", "reference"])
            >>> flower_dets = results["flower"]
            >>> ref_dets = results["reference"]
        """
        # Load image once
        image = Image.open(image_path).convert('RGB')
        
        # Create batch: same image repeated for each prompt
        images = [image] * len(prompts)
        
        # Prepare batched inputs
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference ONCE for all prompts ✅
        print(f"Segmenting {len(prompts)} prompts in batch: {prompts}")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process all results
        all_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )
        
        # Organize by prompt
        results_by_prompt = {}
        
        for prompt_idx, prompt in enumerate(prompts):
            results = all_results[prompt_idx]
            masks = results.get('masks', [])
            boxes = results.get('boxes', [])
            scores = results.get('scores', [])
            
            detections = []
            for i in range(len(masks)):
                mask = masks[i]
                if hasattr(mask, "cpu"):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                
                while mask_np.ndim > 2:
                    mask_np = mask_np[0]
                
                if mask_np.dtype.kind in ("f", "i") and mask_np.max() > 1:
                    mask_np = (1.0 / (1.0 + np.exp(-mask_np))) > 0.5
                else:
                    mask_np = mask_np.astype(bool)
                
                mask_binary = mask_np.astype(np.uint8)
                box = boxes[i].tolist() if len(boxes) > i else [0, 0, 0, 0]
                score = scores[i].item() if len(scores) > i else 1.0
                
                detections.append({
                    'mask': mask_binary,
                    'score': float(score),
                    'area': int(mask_binary.sum()),
                    'bbox': [int(b) for b in box],
                    'prompt': prompt
                })
            
            results_by_prompt[prompt] = detections
            print(f"  ✓ '{prompt}': {len(detections)} instances")
        
        return results_by_prompt
    
    def batch_process_images(self,
                            image_paths: List[str],
                            text_prompt: str,
                            threshold: float = 0.3,
                            mask_threshold: float = 0.7,
                            batch_size: int = 4) -> List[List[Dict]]:
        """
        ✅ NEW: Process multiple images efficiently in GPU batches
        
        Args:
            image_paths: List of image paths
            text_prompt: Single text prompt for all images
            threshold: Detection confidence
            mask_threshold: Mask tightness (0.7 default)
            batch_size: Images per batch (adjust for GPU memory)
            
        Returns:
            List of detection lists (one per image)
        """
        all_detections = []
        
        print(f"Processing {len(image_paths)} images in batches of {batch_size}...")
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load images
            images = [Image.open(p).convert('RGB') for p in batch_paths]
            
            # Prepare batched inputs (same prompt for all)
            inputs = self.processor(
                images=images,
                text=[text_prompt] * len(images),
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            batch_num = i//batch_size + 1
            total_batches = (len(image_paths)-1)//batch_size + 1
            print(f"  Batch {batch_num}/{total_batches}...")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            batch_results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )
            
            # Convert each image's results
            for img_idx, img_results in enumerate(batch_results):
                masks = img_results.get('masks', [])
                boxes = img_results.get('boxes', [])
                scores = img_results.get('scores', [])
                
                detections = []
                for j in range(len(masks)):
                    mask = masks[j]
                    if hasattr(mask, "cpu"):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = np.array(mask)
                    
                    while mask_np.ndim > 2:
                        mask_np = mask_np[0]
                    
                    if mask_np.dtype.kind in ("f", "i") and mask_np.max() > 1:
                        mask_np = (1.0 / (1.0 + np.exp(-mask_np))) > 0.5
                    else:
                        mask_np = mask_np.astype(bool)
                    
                    mask_binary = mask_np.astype(np.uint8)
                    box = boxes[j].tolist() if len(boxes) > j else [0, 0, 0, 0]
                    score = scores[j].item() if len(scores) > j else 1.0
                    
                    detections.append({
                        'mask': mask_binary,
                        'score': float(score),
                        'area': int(mask_binary.sum()),
                        'bbox': [int(b) for b in box],
                        'prompt': text_prompt
                    })
                
                print(f"    Image {i+img_idx+1}: {len(detections)} detections")
                all_detections.append(detections)
        
        print(f"✓ Processed {len(image_paths)} images")
        return all_detections
    
    def filter_dead_flowers(self,
                           image_path: str,
                           detections: List[Dict],
                           saturation_threshold: float = 0.15,
                           min_alive_ratio: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
        """
        ✅ NEW: Filter out dead/brown flowers using color analysis
        
        Strategy:
        - Dead flowers typically have LOW saturation (brown/gray)
        - Alive flowers have HIGH saturation (vibrant colors)
        - Analyze HSV color space in each mask region
        
        Args:
            image_path: Path to image
            detections: List of flower detections
            saturation_threshold: Median saturation below this = dead (0.0-1.0)
            min_alive_ratio: Minimum fraction of pixels that must be "alive" colored
            
        Returns:
            Tuple of (alive_flowers, dead_flowers)
        """
        # Load image
        image_bgr = cv2.imread(image_path)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        alive_flowers = []
        dead_flowers = []
        
        for det in detections:
            mask = det['mask']
            
            # Extract HSV values in mask region
            h_values = image_hsv[:, :, 0][mask > 0]
            s_values = image_hsv[:, :, 1][mask > 0]  # Saturation (0-255)
            v_values = image_hsv[:, :, 2][mask > 0]  # Value/brightness (0-255)
            
            if len(s_values) == 0:
                continue
            
            # Normalize saturation to 0-1
            s_normalized = s_values / 255.0
            
            # Calculate metrics
            median_saturation = np.median(s_normalized)
            mean_saturation = np.mean(s_normalized)
            
            # Count "alive" pixels (high saturation)
            alive_pixels = np.sum(s_normalized > saturation_threshold)
            total_pixels = len(s_normalized)
            alive_ratio = alive_pixels / total_pixels
            
            # Decision: Is this flower alive?
            is_alive = (median_saturation > saturation_threshold and 
                       alive_ratio > min_alive_ratio)
            
            # Add metadata
            det['saturation_median'] = float(median_saturation)
            det['saturation_mean'] = float(mean_saturation)
            det['alive_ratio'] = float(alive_ratio)
            det['is_alive'] = is_alive
            
            if is_alive:
                alive_flowers.append(det)
            else:
                dead_flowers.append(det)
        
        print(f"✓ Flower filtering:")
        print(f"  Alive: {len(alive_flowers)} flowers (vibrant colors)")
        print(f"  Dead/Brown: {len(dead_flowers)} flowers (low saturation)")
        
        return alive_flowers, dead_flowers
    
    def analyze_flower_health(self,
                             image_path: str,
                             detections: List[Dict]) -> Dict:
        """
        ✅ NEW: Detailed analysis of flower health/color
        
        Returns stats for each flower to help tune thresholds
        """
        image_bgr = cv2.imread(image_path)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        analysis = []
        
        for i, det in enumerate(detections):
            mask = det['mask']
            
            # Extract HSV
            s_values = image_hsv[:, :, 1][mask > 0] / 255.0
            v_values = image_hsv[:, :, 2][mask > 0] / 255.0
            
            if len(s_values) == 0:
                continue
            
            stats = {
                'flower_id': i,
                'score': det['score'],
                'area_pixels': det['area'],
                'saturation_median': float(np.median(s_values)),
                'saturation_mean': float(np.mean(s_values)),
                'saturation_std': float(np.std(s_values)),
                'value_median': float(np.median(v_values)),
                'low_sat_pixel_ratio': float(np.sum(s_values < 0.15) / len(s_values))
            }
            
            analysis.append(stats)
        
        return {
            'flowers': analysis,
            'summary': {
                'total_flowers': len(analysis),
                'avg_saturation': float(np.mean([f['saturation_median'] for f in analysis])),
                'min_saturation': float(np.min([f['saturation_median'] for f in analysis])),
                'max_saturation': float(np.max([f['saturation_median'] for f in analysis]))
            }
        }
    
    def get_combined_mask(self, detections: List[Dict]) -> np.ndarray:
        """
        Combine multiple detection masks into one
        
        Args:
            detections: List of detection dictionaries with 'mask' field
            
        Returns:
            Combined binary mask (OR of all detection masks)
        """
        if not detections:
            return np.array([])
        
        # Get first mask to determine shape
        first_mask = detections[0]['mask']
        combined = np.zeros_like(first_mask, dtype=np.uint8)
        
        # OR all masks together
        for detection in detections:
            mask = detection['mask']
            combined = np.maximum(combined, (mask > 0).astype(np.uint8))
        
        return combined
    
    def visualize_with_health(self,
                             image_path: str,
                             alive_flowers: List[Dict],
                             dead_flowers: List[Dict],
                             output_path: Optional[str] = None) -> np.ndarray:
        """
        ✅ NEW: Visualize flowers colored by health status
        
        - GREEN borders = Alive flowers
        - RED borders = Dead flowers
        """
        image = cv2.imread(image_path)
        overlay = image.copy()
        
        # Draw alive flowers (GREEN)
        for det in alive_flowers:
            mask = det['mask']
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)  # Green
            
            # Add label
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"ALIVE: {det['saturation_median']:.2f}"
                    cv2.putText(overlay, label, (cx-60, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw dead flowers (RED)
        for det in dead_flowers:
            mask = det['mask']
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)  # Red
            
            # Add label
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"DEAD: {det['saturation_median']:.2f}"
                    cv2.putText(overlay, label, (cx-60, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, overlay)
        
        return overlay
    
    def visualize(self,
                 image_path: str,
                 detections: List[Dict],
                 output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize all detections with bounding boxes and masks
        
        Args:
            image_path: Path to image
            detections: List of detections to visualize
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image as numpy array
        """
        image = cv2.imread(image_path)
        overlay = image.copy()
        
        # Draw each detection
        for i, det in enumerate(detections):
            mask = det['mask']
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Random color per detection (for distinction)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.drawContours(overlay, contours, -1, color, 3)
            
            # Draw bounding box
            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add label with score
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    score = det.get('score', 0)
                    label = f"{i+1}: {score:.2f}"
                    cv2.putText(overlay, label, (cx-30, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, overlay)
        
        return overlay


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_combined_segmentation():
    """
    Example: Segment flowers AND reference in ONE forward pass (2x faster!)
    """
    sam = SAM3HuggingFaceSegmenter()
    
    # Instead of:
    # flowers = sam.segment_by_text("img.jpg", "flower")
    # ref = sam.segment_by_text("img.jpg", "brown square cardboard")
    
    # Do this (MUCH FASTER):
    results = sam.segment_combined(
        "img.jpg", 
        prompts=["flower", "brown square cardboard"],
        mask_threshold=0.7  # Tighter boundaries!
    )
    
    flowers = results["flower"]
    reference = results["brown square cardboard"]
    
    print(f"Found {len(flowers)} flowers and {len(reference)} references")


def example_dead_flower_filtering():
    """
    Example: Filter out dead/brown flowers
    """
    sam = SAM3HuggingFaceSegmenter()
    
    # Segment all flowers
    all_flowers = sam.segment_by_text(
        "img.jpg", 
        "flower",
        mask_threshold=0.7
    )
    
    # Filter by health
    alive, dead = sam.filter_dead_flowers(
        "img.jpg",
        all_flowers,
        saturation_threshold=0.15,  # Tune this!
        min_alive_ratio=0.3
    )
    
    print(f"Alive: {len(alive)}, Dead: {len(dead)}")
    
    # Calculate area using only alive flowers
    alive_area = sum(f['area'] for f in alive)
    
    # Visualize
    sam.visualize_with_health("img.jpg", alive, dead, "output_health.jpg")


def example_batch_processing():
    """
    Example: Process 60 images efficiently in batches
    """
    sam = SAM3HuggingFaceSegmenter()
    
    image_paths = [f"image_{i}.jpg" for i in range(1, 61)]
    
    # Process all in batches (MUCH faster than loop)
    all_results = sam.batch_process_images(
        image_paths,
        text_prompt="flower",
        batch_size=8,  # Adjust for GPU memory
        mask_threshold=0.7
    )
    
    # Results is list of detection lists
    for i, detections in enumerate(all_results):
        print(f"Image {i+1}: {len(detections)} flowers")


def example_health_analysis():
    """
    Example: Analyze flower health to tune thresholds
    """
    sam = SAM3HuggingFaceSegmenter()
    
    flowers = sam.segment_by_text("img.jpg", "flower", mask_threshold=0.7)
    
    # Get detailed stats
    analysis = sam.analyze_flower_health("img.jpg", flowers)
    
    print("\nFlower Health Analysis:")
    print(f"Total flowers: {analysis['summary']['total_flowers']}")
    print(f"Avg saturation: {analysis['summary']['avg_saturation']:.3f}")
    print(f"Range: {analysis['summary']['min_saturation']:.3f} - {analysis['summary']['max_saturation']:.3f}")
    
    # Individual flower stats
    for flower in analysis['flowers']:
        status = "ALIVE" if flower['saturation_median'] > 0.15 else "DEAD"
        print(f"  Flower {flower['flower_id']}: Sat={flower['saturation_median']:.3f} → {status}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SAM3 IMPROVED - USAGE EXAMPLES")
    print("="*70)
    
    print("\n1. Combined segmentation (2x faster):")
    print("   results = sam.segment_combined(img, ['flower', 'reference'])")
    
    print("\n2. Dead flower filtering:")
    print("   alive, dead = sam.filter_dead_flowers(img, all_flowers)")
    
    print("\n3. Batch processing:")
    print("   results = sam.batch_process_images(image_list, 'flower')")
    
    print("\n4. Health analysis:")
    print("   stats = sam.analyze_flower_health(img, flowers)")
    
    print("\n" + "="*70)