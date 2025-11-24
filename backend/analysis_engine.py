# File: analysis_engine.py

import time
import traceback
import itertools
from collections import defaultdict
import io
import piexif

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepface import DeepFace
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_BACKGROUND = {}
LOADED_FACE_MODELS = set()

FACE_MODEL_NAMES = ["VGG-Face", "Facenet", "Facenet512", "ArcFace", "SFace"]
BACKGROUND_MODEL_NAMES = ['resnet50', 'densenet121', 'vit', 'dino']
DETECTION_IMAGE_MAX_SIZE = 320
FEATURE_IMAGE_SIZE = 224

# =============================================================================
# 1. การตั้งค่าเกณฑ์มาตรฐาน (THRESHOLDS CONFIGURATION)
#    อ้างอิงจากผลการทดลอง: Protocol A (Face) และ Protocol B (Background)
# =============================================================================
THRESHOLDS_CONFIG = {
    # Protocol A: ใบหน้า (ใช้เกณฑ์ ArcFace สำหรับทุกโมเดล)
    "Face": {
        "default": {"cosine": 0.51, "pearson": 0.51}
    },
    
    # Protocol B: ฉากหลัง (ใช้เกณฑ์ ResNet50 สำหรับทุกโมเดล)
    "Background": {
        "default": {"cosine": [0.68, 0.71], "pearson": [0.64, 0.67]}
    }
}



def load_specific_face_model(model_name):
    global LOADED_FACE_MODELS
    if model_name in LOADED_FACE_MODELS: return
    try:
        print(f"On-demand loading Face Model: {model_name}...")
        DeepFace.build_model(model_name)
        LOADED_FACE_MODELS.add(model_name)
        print(f"Loaded face model: {model_name}")
    except Exception as e:
        print(f"Error loading face model {model_name}: {e}")
       

def load_specific_background_model(model_name):
    global MODELS_BACKGROUND
    if model_name in MODELS_BACKGROUND: return
    try:
        print(f"On-demand loading Background Model: {model_name}...")
        if model_name == 'deeplab': model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT).to(DEVICE).eval()
        elif model_name == 'resnet50': model = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE).children())[:-2]).eval()
        elif model_name == 'densenet121': model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).to(DEVICE).features.eval()
        elif model_name == 'vit': model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(DEVICE); model.heads.head = nn.Identity(); model = model.eval()
        elif model_name == 'dino': model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(DEVICE).eval()
        else: raise ValueError(f"Unknown background model name: {model_name}")
        MODELS_BACKGROUND[model_name] = model
        print(f"Loaded background model: {model_name}")
    except Exception as e:
        print(f"Failed to load background model {model_name}")
        traceback.print_exc()

def resize_for_detection(img_cv, max_size=DETECTION_IMAGE_MAX_SIZE):
    h, w = img_cv.shape[:2] 
    if max(h, w) <= max_size: return img_cv
    if w > h: new_w, new_h = max_size, int(h * (max_size / w))
    else: new_h, new_w = max_size, int(w * (max_size / h))

    new_w, new_h = max(1, new_w), max(1, new_h)
    return cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

def get_preprocessor(is_transformer=False):
    size = FEATURE_IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC if is_transformer else transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def compute_similarity(vec1, vec2):
    vec1 = np.asarray(vec1, dtype=np.float32).reshape(1, -1)
    vec2 = np.asarray(vec2, dtype=np.float32).reshape(1, -1)

    # Cosine
    cos_sim = cosine_similarity(vec1, vec2)[0, 0]

    # Pearson
    vec1_flat = vec1.flatten()
    vec2_flat = vec2.flatten()
    if np.std(vec1_flat) > 1e-6 and np.std(vec2_flat) > 1e-6 and len(vec1_flat) == len(vec2_flat) and len(vec1_flat) > 1:
         try:
             pearson_corr_raw, _ = pearsonr(vec1_flat, vec2_flat)
             if np.isnan(pearson_corr_raw): pearson_corr_norm = 0.0
             else: pearson_corr_norm = pearson_corr_raw # ใช้ค่าดิบ (-1 ถึง 1) ตามการทดลองใหม่
         except ValueError: 
             pearson_corr_norm = 0.0
    else:
        pearson_corr_norm = 0.0

    # Clip ค่าให้อยู่ในช่วงที่ถูกต้อง
    return float(np.clip(cos_sim, -1.0, 1.0)), float(np.clip(pearson_corr_norm, -1.0, 1.0))


def perform_sift_analysis(img1_orig, img2_orig, person_mask1, person_mask2, img1_inpainted, img2_inpainted):
    try:
        sift = cv2.SIFT_create()
        background_mask1 = ((1 - person_mask1) * 255).astype(np.uint8)
        background_mask2 = ((1 - person_mask2) * 255).astype(np.uint8)
        kp1, des1 = sift.detectAndCompute(img1_orig, background_mask1)
        kp2, des2 = sift.detectAndCompute(img2_orig, background_mask2)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return {"matches": 0, "visualization": None}
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_n in matches:
             if len(m_n) == 2: 
                 m, n = m_n
                 if m.distance < 0.75 * n.distance:
                     good_matches.append(m)

        vis = cv2.drawMatches(img1_inpainted, kp1, img2_inpainted, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return {"matches": len(good_matches), "visualization": vis}
    except Exception as e:
        print(f"Error in SIFT analysis: {e}")
        traceback.print_exc() 
        return {"matches": "Error", "visualization": None}

# =============================================================================
# 2. ปรับปรุง Logic การตัดสินใจ (Protocol A & B)
# =============================================================================

def get_face_group(cos_sim, pearson_corr, model_name="default"):
    """
    Protocol A: Face Verification
    เงื่อนไข: ต้องผ่านเกณฑ์ทั้ง Cosine และ Pearson (AND Logic)
    """
    # ดึงค่า Threshold
    thresholds = THRESHOLDS_CONFIG["Face"].get(model_name, THRESHOLDS_CONFIG["Face"]["default"])
    
    # ตรวจสอบเงื่อนไข
    if cos_sim >= thresholds["cosine"] and pearson_corr >= thresholds["pearson"]:
        return "คนเดียวกัน"
    else:
        return "บุคคลที่ไม่ถูกจัดกลุ่ม"

def get_background_group(cos_sim, pearson_corr, model_name="default"):
    """
    Protocol B: Background Verification (Tri-Zone)
    เงื่อนไข:
    - Accept: สูงกว่าเกณฑ์ High Conf ทั้งคู่
    - Reject: ต่ำกว่าเกณฑ์ Low Conf ตัวใดตัวหนึ่ง
    - Ambiguous: ตรงกลาง
    """
    # ดึงค่า Threshold Ranges [low, high]
    limits = THRESHOLDS_CONFIG["Background"].get(model_name, THRESHOLDS_CONFIG["Background"]["default"])
    
    cos_low, cos_high = limits["cosine"]
    pear_low, pear_high = limits["pearson"]
    
    # 1. โซนยอมรับ (Accept Zone)
    if cos_sim >= cos_high and pearson_corr >= pear_high:
        return "สถานที่เดียวกัน"
    
    # 2. โซนปฏิเสธ (Reject Zone)
    elif cos_sim < cos_low or pearson_corr < pear_low:
        return "สถานที่ที่ไม่ถูกจัดกลุ่ม"
        
    # 3. โซนกำกวม (Ambiguous Zone)
    else:
        return "ไม่สามารถระบุได้" # ส่งต่อให้ SIFT

# =============================================================================

def calculate_average_scores(groups, pairwise_results):
    avg_scores = {}
    pair_lookup = {tuple(sorted(p['pair'])): p for p in pairwise_results}
    group_counter = 0
    for group_name, list_of_groups in groups.items():
        for group_members in list_of_groups:
            group_id = f"{group_name}_{group_counter}"
            cos_scores, p_scores = [], []
            if len(group_members) == 1:
                single_member = group_members[0]
                relevant_pairs = [p for p in pairwise_results if single_member in p['pair']]
                if relevant_pairs:
                     cos_scores = [p['cosine'] for p in relevant_pairs]
                     p_scores = [p['pearson'] for p in relevant_pairs]
                else:
                    cos_scores, p_scores = [0.0], [0.0]

            elif len(group_members) > 1:
                for m1, m2 in itertools.combinations(group_members, 2):
                    pair_data = pair_lookup.get(tuple(sorted((m1, m2))))
                    if pair_data:
                        cos_scores.append(pair_data['cosine'])
                        p_scores.append(pair_data['pearson'])

            avg_cos = np.mean(cos_scores) if cos_scores else 0.0
            avg_p = np.mean(p_scores) if p_scores else 0.0

            avg_scores[group_id] = {
                'cosine': float(avg_cos), 
                'pearson': float(avg_p), 
            }
            group_counter += 1
    return avg_scores


def build_report_groups(pairwise_results, all_filenames, type_key):
    hierarchy = {"Face": ["คนเดียวกัน", "ไม่สามารถระบุได้"], "Background": ["สถานที่เดียวกัน", "ไม่สามารถระบุได้"]}
    leftover_name = {"Face": "บุคคลที่ไม่ถูกจัดกลุ่ม", "Background": "สถานที่ที่ไม่ถูกจัดกลุ่ม"}

    final_groups = defaultdict(list)
    assigned_items = set()
    pair_lookup = {tuple(sorted(p['pair'])): p['group'] for p in pairwise_results}

    for group_name in hierarchy[type_key]:
        current_type_pairs = {k: v for k, v in pair_lookup.items() if v == group_name}
        if not current_type_pairs: continue

        adj = defaultdict(list)
        involved_items = set()
        for item1, item2 in current_type_pairs.keys():
            adj[item1].append(item2)
            adj[item2].append(item1)
            involved_items.add(item1)
            involved_items.add(item2)

        visited_in_type = set()
        for item in involved_items:
            if item not in assigned_items and item not in visited_in_type:
                component = []
                q = [item]
                visited_in_type.add(item)
                processed_in_component = set() 

                while q:
                    current = q.pop(0)
                    if current not in assigned_items:
                        if current not in processed_in_component:
                             component.append(current)
                             processed_in_component.add(current)
                             for neighbor in adj.get(current, []):
                                 
                                 if neighbor in involved_items and neighbor not in visited_in_type and neighbor not in assigned_items:
                                     visited_in_type.add(neighbor)
                                     q.append(neighbor)

                if component:
                    final_groups[group_name].append(sorted(component))
                    assigned_items.update(component)

    leftover_items = sorted([item for item in all_filenames if item not in assigned_items])
    if leftover_items:
        
        final_groups[leftover_name[type_key]].append(leftover_items)

    return dict(final_groups)

def _run_face_pipeline(all_images, face_model_name, job_manager, job_id):
    face_features = {}

    filenames_with_faces = [fn for fn, data in all_images.items() if data.get('face_for_represent') is not None]
    
    num_faces = len(filenames_with_faces)
    if num_faces == 0: return {}

    for i, filename in enumerate(filenames_with_faces):
        
        current_img = all_images[filename]['face_for_represent']
        
        job_manager.update_job(job_id, message=f"กำลังวิเคราะห์ใบหน้าด้วย {face_model_name}... ({i + 1}/{num_faces})")
        try:
            embedding_objs = DeepFace.represent(
                img_path=current_img, 
                model_name=face_model_name,
                enforce_detection=False, 
                detector_backend='skip' 
            )
            if embedding_objs:
                face_features[filename] = embedding_objs[0]['embedding']
        except Exception as e:
            print(f"Could not represent single face for {filename} with {face_model_name}: {e}")
            traceback.print_exc()
    return face_features

def _run_background_pipeline(all_images, bg_model_name, job_manager, job_id):
    bg_features = {}
    valid_images_data = [(fname, data['background_cv']) for fname, data in all_images.items() if data.get('background_cv') is not None and data['background_cv'].size > 0]

    if not valid_images_data: return {}

    original_filenames = [fname for fname, _ in valid_images_data]
    bg_images_for_batch = [img for _, img in valid_images_data]

    job_manager.update_job(job_id, message=f"กำลังสกัดคุณลักษณะฉากหลังด้วย {bg_model_name}...") 

    try:
        is_transformer = 'vit' in bg_model_name or 'dino' in bg_model_name
        bg_preprocessor = get_preprocessor(is_transformer=is_transformer)


        processed_batches = []
        valid_indices = []
        for idx, img in enumerate(bg_images_for_batch):
             try:
                 img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                 pil_img = Image.fromarray(img_rgb)
                 processed_img = bg_preprocessor(pil_img)
                 processed_batches.append(processed_img)
                 valid_indices.append(idx)
             except Exception as e:
                 print(f"Skipping background preprocessing for image {original_filenames[idx]} due to error: {e}")

        if not processed_batches:
             print("No background images could be preprocessed.")
             return {}

        batch_input_bg = torch.stack(processed_batches).to(DEVICE)

        model_to_use = MODELS_BACKGROUND.get(bg_model_name)
        if model_to_use is None:
             print(f"Background model {bg_model_name} not loaded.")
             return {}

        with torch.no_grad():
            features = model_to_use(batch_input_bg)
            if isinstance(features, torch.Tensor):
                 if features.ndim == 3: 
                     features = features[:, 0, :]
                 elif features.ndim == 4: 
                     features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1) 

                 if features.ndim > 2:
                     features = features.view(features.size(0), -1) 
            else:
                 print(f"Warning: Unexpected output type from {bg_model_name}: {type(features)}")
                 return {}

        for i, original_idx in enumerate(valid_indices):
            filename = original_filenames[original_idx]
            bg_features[filename] = features[i].cpu().numpy()

    except Exception as e:
         print(f"Error during background feature extraction with {bg_model_name}: {e}")
         traceback.print_exc()

    return bg_features


def run_full_analysis(job_id, image_files, face_models_to_run, bg_models_to_run, job_manager):
    final_results_all_models = {}
    all_images_data_for_report = {} 

    try:
        job_manager.update_job(job_id, status="processing", message="กำลังโหลดโมเดล...")
        models_needed = set(face_models_to_run) | set(bg_models_to_run)
        if face_models_to_run: models_needed.add('retinaface') 
        if bg_models_to_run: models_needed.add('deeplab')

        if 'retinaface' in models_needed:
             pass
        if 'deeplab' in models_needed: load_specific_background_model('deeplab')
        for model_name in face_models_to_run: load_specific_face_model(model_name)
        for model_name in bg_models_to_run: load_specific_background_model(model_name)


        job_manager.update_job(job_id, message="กำลังประมวลผลภาพเบื้องต้น...")
        all_images = {}
        filenames = list(image_files.keys())
        for fname, img_bytes in image_files.items():
            try:
                metadata_datetime = "N/A"
                try:
                    exif_dict = piexif.load(img_bytes)
                    if piexif.ExifIFD.DateTimeOriginal in exif_dict["Exif"]:
                        dt_str = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("utf-8")
                        parts = dt_str.split(" ")
                        if len(parts) == 2:
                            date_part = parts[0].replace(":", "-", 2)
                            metadata_datetime = f"{date_part} {parts[1]}"
                        else: 
                            metadata_datetime = dt_str.replace(":", "-", 2)
                except Exception:
                    pass 

                img_np = np.frombuffer(img_bytes, np.uint8)
                original_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if original_cv is None: raise ValueError("Could not decode image")
                resized_cv = resize_for_detection(original_cv)
                
                all_images[fname] = {
                    "original_cv": resized_cv,
                    "metadata_datetime": metadata_datetime 
                }
            except Exception as e:
                 print(f"Skipping image {fname} due to loading error: {e}")

        if not all_images:
             raise ValueError("No valid images could be loaded.")

        valid_filenames = list(all_images.keys())
        all_pairs = list(itertools.combinations(valid_filenames, 2))

        retinaface_time = 0
        deeplab_time = 0

        if face_models_to_run:
            start_time_retina = time.time()
            processed_faces_preview = {}
            num_images = len(all_images)
            for i, (filename, data) in enumerate(all_images.items()):
                job_manager.update_job(job_id, message=f"กำลังตรวจจับใบหน้า... ({i + 1}/{num_images})")
                try:
                    face_objs = DeepFace.extract_faces(
                        data['original_cv'], 
                        detector_backend='retinaface',
                        enforce_detection=False,
                        align=True
                    )
                    if face_objs and face_objs[0]['face'].size > 0:
                         face_rgb_extracted = face_objs[0]['face'] 
                         data['face_for_represent'] = face_rgb_extracted
                         face_bgr_preview = cv2.cvtColor((face_rgb_extracted * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                         processed_faces_preview[filename] = face_bgr_preview
                    else:
                         data['face_for_represent'] = None
                         processed_faces_preview[filename] = None 
                except Exception as e:
                    print(f"Failed to extract face from {filename}: {e}")
                    data['face_for_represent'] = None
                    processed_faces_preview[filename] = None
            retinaface_time = time.time() - start_time_retina
            job_manager.update_job(job_id, partial_results={'processed_faces': processed_faces_preview})

        if bg_models_to_run:
            start_time_deeplab = time.time()
            job_manager.update_job(job_id, message="กำลังแบ่งส่วนภาพฉากหลัง...")
            deeplab_model = MODELS_BACKGROUND.get('deeplab')
            processed_backgrounds_preview = {}

            if deeplab_model:
                deeplab_preprocessor = get_preprocessor(is_transformer=False)
                image_tensors = []
                valid_indices_deeplab = []
                original_shapes = {}

                for i, (filename, data) in enumerate(all_images.items()):
                     try:
                         img_rgb = cv2.cvtColor(data['original_cv'], cv2.COLOR_BGR2RGB)
                         pil_img = Image.fromarray(img_rgb)
                         image_tensors.append(deeplab_preprocessor(pil_img))
                         valid_indices_deeplab.append(i)
                         original_shapes[filename] = data['original_cv'].shape[:2]
                     except Exception as e:
                          print(f"Skipping deeplab preprocessing for {filename}: {e}")

                if image_tensors:
                    batch_input_deeplab = torch.stack(image_tensors).to(DEVICE)
                    with torch.no_grad():
                        batch_output_deeplab = deeplab_model(batch_input_deeplab)['out']
                    batch_pred_mask = torch.argmax(batch_output_deeplab, dim=1).cpu().numpy().astype(np.uint8)

                    for i, original_idx in enumerate(valid_indices_deeplab):
                         filename = valid_filenames[original_idx]
                         data = all_images[filename]
                         original_h, original_w = original_shapes[filename]

                         person_mask_resized = cv2.resize((batch_pred_mask[i] == 15).astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)

                         data['person_mask_for_sift'] = person_mask_resized
                         data['background_cv'] = cv2.inpaint(data['original_cv'], person_mask_resized, 3, cv2.INPAINT_TELEA)

                         bg_only_cv_for_preview = data['original_cv'].copy()
                         bg_only_cv_for_preview[person_mask_resized == 1] = [0, 0, 0] 
                         processed_backgrounds_preview[filename] = bg_only_cv_for_preview
                else:
                     print("No images could be preprocessed for DeepLab.")


            else:
                 print("DeepLab model not loaded, skipping background segmentation.")
                 for filename, data in all_images.items():
                     data['person_mask_for_sift'] = np.zeros(data['original_cv'].shape[:2], dtype=np.uint8) 
                     data['background_cv'] = data['original_cv'].copy() 
                     processed_backgrounds_preview[filename] = data['original_cv'].copy()


            deeplab_time = time.time() - start_time_deeplab
            job_manager.update_job(job_id, partial_results={'processed_backgrounds': processed_backgrounds_preview})


        for face_model_name in face_models_to_run:
            start_time_analysis = time.time()
            job_manager.update_job(job_id, message=f"กำลังสกัดคุณลักษณะใบหน้าด้วย {face_model_name}...")
            face_features = _run_face_pipeline(all_images, face_model_name, job_manager, job_id)
            face_pairwise = []
            if face_features:
                job_manager.update_job(job_id, message=f"กำลังเปรียบเทียบใบหน้าด้วย {face_model_name}...")
                feature_filenames = list(face_features.keys())
                face_pairs = list(itertools.combinations(feature_filenames, 2))
                for f1, f2 in face_pairs:
                    cos_sim, p_corr = compute_similarity(face_features[f1], face_features[f2])
                    # [แก้ไข] ส่งชื่อโมเดลไปเพื่อดึงเกณฑ์เฉพาะ
                    group = get_face_group(cos_sim, p_corr, model_name=face_model_name)
                    face_pairwise.append({'pair': (f1, f2), 'cosine': cos_sim, 'pearson': p_corr, 'group': group})

            face_groups = build_report_groups(face_pairwise, list(face_features.keys()), "Face")
            avg_face_scores = calculate_average_scores(face_groups, face_pairwise)
            analysis_time = time.time() - start_time_analysis

            final_results_all_models[face_model_name] = {
                "image_count": len(valid_filenames),
                "face_pairwise_results": face_pairwise,
                "face_groups": face_groups,
                "avg_face_scores": avg_face_scores,
                "processing_time": analysis_time + retinaface_time,
            }

        for bg_model_name in bg_models_to_run:
            start_time_analysis = time.time()
            job_manager.update_job(job_id, message=f"กำลังสกัดคุณลักษณะฉากหลังด้วย {bg_model_name}...")
            bg_features = _run_background_pipeline(all_images, bg_model_name, job_manager, job_id)
            bg_pairwise = []
            if bg_features:
                job_manager.update_job(job_id, message=f"กำลังเปรียบเทียบฉากหลังด้วย {bg_model_name}...")
                feature_filenames = list(bg_features.keys())
                bg_pairs = list(itertools.combinations(feature_filenames, 2))
                for f1, f2 in bg_pairs:
                    cos_sim, p_corr = compute_similarity(bg_features[f1], bg_features[f2])
                    # [แก้ไข] ส่งชื่อโมเดลไปเพื่อดึงเกณฑ์เฉพาะ
                    group = get_background_group(cos_sim, p_corr, model_name=bg_model_name)
                    bg_pairwise.append({'pair': (f1, f2), 'cosine': cos_sim, 'pearson': p_corr, 'group': group, 'sift': None})

            background_groups = build_report_groups(bg_pairwise, list(bg_features.keys()), "Background")

            if "ไม่สามารถระบุได้" in background_groups:
                 job_manager.update_job(job_id, message=f"กำลังวิเคราะห์ SIFT สำหรับ {bg_model_name}...")
                 pair_lookup_sift = {tuple(sorted(p['pair'])): p for p in bg_pairwise if p['group'] == "ไม่สามารถระบุได้"}
                 for group_members in background_groups["ไม่สามารถระบุได้"]:
                     for f1, f2 in itertools.combinations(group_members, 2):
                         pair_key = tuple(sorted((f1, f2)))
                         if pair_key in pair_lookup_sift:
                             if 'original_cv' in all_images[f1] and 'original_cv' in all_images[f2] and \
                                'person_mask_for_sift' in all_images[f1] and 'person_mask_for_sift' in all_images[f2] and \
                                'background_cv' in all_images[f1] and 'background_cv' in all_images[f2]:

                                 sift_res = perform_sift_analysis(
                                     all_images[f1]['original_cv'], all_images[f2]['original_cv'],
                                     all_images[f1]['person_mask_for_sift'], all_images[f2]['person_mask_for_sift'],
                                     all_images[f1]['background_cv'], all_images[f2]['background_cv'] 
                                 )
                                 pair_lookup_sift[pair_key]['sift'] = sift_res
                             else:
                                 print(f"Skipping SIFT for pair {pair_key} due to missing data.")


            avg_bg_scores = calculate_average_scores(background_groups, bg_pairwise)
            analysis_time = time.time() - start_time_analysis

            final_results_all_models[bg_model_name] = {
                "image_count": len(valid_filenames),
                "background_pairwise_results": bg_pairwise,
                "background_groups": background_groups,
                "avg_background_scores": avg_bg_scores,
                "processing_time": analysis_time + deeplab_time, 
            }

        all_images_data_for_report = {}
        for fname, data in all_images.items():
            all_images_data_for_report[fname] = {
                 'original_cv': data.get('original_cv'),
                 'face_cv': processed_faces_preview.get(fname),
                 'background_cv': data.get('background_cv'),
                 'metadata_datetime': data.get('metadata_datetime'),
            }

        final_payload = {
            "models": final_results_all_models,
            "all_images_data": all_images_data_for_report 
        }

        job_manager.update_job(job_id, status="completed", final_results=final_payload)

    except Exception as e:
        print(f"Error in job {job_id}: {e}"); traceback.print_exc(); job_manager.update_job(job_id, status="failed", message=f"An error occurred during analysis: {e}")