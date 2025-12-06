import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import base64
import io
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from fpdf import FPDF
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn # 
import analysis_engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: Dict[str, Dict[str, Any]] = {}

class JobManager:
    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {
            "status": "pending",
            "message": "กำลังเตรียมการวิเคราะห์...",
            "image_files": {},
            "partial_results": {},
            "final_results": None,
        }
        return job_id

    def get_job(self, job_id: str) -> Dict[str, Any]:
        job = JOBS.get(job_id)
        return job 

    def update_job(self, job_id: str, status: str = None, message: str = None, image_files: Dict[str, bytes] = None, partial_results: Dict[str, Any] = None, final_results: Dict[str, Any] = None):
        if job_id in JOBS:
            if status: JOBS[job_id]['status'] = status
            if message: JOBS[job_id]['message'] = message
            if image_files: JOBS[job_id]['image_files'] = image_files
            if partial_results:
                for key, value in partial_results.items():
                    if key not in JOBS[job_id]['partial_results']:
                        JOBS[job_id]['partial_results'][key] = {}
                    JOBS[job_id]['partial_results'][key].update(value)
            if final_results:
                JOBS[job_id]['final_results'] = final_results
                JOBS[job_id]['status'] = 'completed'
                JOBS[job_id]['message'] = 'การประมวลผลเสร็จสมบูรณ์!'

job_manager = JobManager()

def convert_numpy_types(data: Any) -> Any:
    if isinstance(data, dict): return {k: convert_numpy_types(v) for k, v in data.items()}
    if isinstance(data, list): return [convert_numpy_types(i) for i in data]
    if isinstance(data, (np.float32, np.float64)): return float(data)
    if isinstance(data, (np.int32, np.int64)): return int(data)
    if isinstance(data, tuple): return list(data)
    return data

def image_to_data_url(img_cv_bgr: np.ndarray) -> str:
    if img_cv_bgr is None: return None
    is_success, buffer = cv2.imencode(".jpg", img_cv_bgr)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}" if is_success else None


class AnalysisRequest(BaseModel):
    mode: str
    face_model: Optional[str] = None
    bg_model: Optional[str] = None


@app.post("/api/submit_images")
async def submit_images_route(images: List[UploadFile] = File(...)):
    
    if not (2 <= len(images) <= 100):
        raise HTTPException(status_code=400, detail="Please upload between 2 and 100 images.")
        
    job_id = job_manager.create_job()
    image_files = {file.filename: await file.read() for file in images}
    job_manager.update_job(job_id, image_files=image_files)
    
    return {"job_id": job_id}


@app.post("/api/start_analysis/{job_id}")
def start_analysis_api(job_id: str, request_data: AnalysisRequest):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    mode = request_data.mode
    face_model = request_data.face_model
    bg_model = request_data.bg_model

    if mode == 'quick_analysis':
        face_models_to_run = [face_model] if face_model else []
        bg_models_to_run = [bg_model] if bg_model else []
    else:
        face_models_to_run = analysis_engine.FACE_MODEL_NAMES
        bg_models_to_run = analysis_engine.BACKGROUND_MODEL_NAMES
        
    if not face_models_to_run and not bg_models_to_run:
        raise HTTPException(status_code=400, detail="No models selected for analysis.")

    analysis_thread = threading.Thread(
        target=analysis_engine.run_full_analysis,
        args=(job_id, job['image_files'], face_models_to_run, bg_models_to_run, job_manager)
    )
    analysis_thread.start()

    return {"job_id": job_id, "message": "Analysis started."}


@app.get("/api/status/{job_id}")
def status_api(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    response_data = {"status": job['status'], "message": job['message']}

    if job.get('partial_results'):
        response_data['partial_results'] = {
            k: {fname: image_to_data_url(img_cv) for fname, img_cv in img_dict.items()}
            for k, img_dict in job['partial_results'].items()
        }

    if job['status'] == 'completed' and job['final_results']:
        final_results_payload = job['final_results']
        if isinstance(final_results_payload, dict) and 'models' in final_results_payload and 'all_images_data' in final_results_payload:
            model_results = final_results_payload['models'].copy()
            all_images_data = final_results_payload['all_images_data'] 

            image_urls = {
                fname: {
                    'original': image_to_data_url(data.get('original_cv')),
                    'face': image_to_data_url(data.get('face_cv')),
                    'background': image_to_data_url(data.get('background_cv')),
                    'metadata_datetime': data.get('metadata_datetime') 
                } for fname, data in all_images_data.items()
            }
            
            for model_name, model_data in model_results.items():
                 if isinstance(model_data, dict):
                    if 'all_images_data' in model_data:
                        del model_data['all_images_data'] 
                    if model_data.get('background_pairwise_results'):
                        sift_visualizations = {}
                        for pair_res in model_data['background_pairwise_results']:
                            if isinstance(pair_res, dict) and pair_res.get('sift') and isinstance(pair_res['sift'], dict) and pair_res['sift'].get('visualization') is not None:
                                try:
                                    pair_tuple = tuple(sorted(pair_res['pair']))
                                    sift_key = "_vs_".join(pair_tuple)
                                    sift_visualizations[sift_key] = image_to_data_url(pair_res['sift']['visualization'])
                                    del pair_res['sift']['visualization']
                                except Exception as e:
                                     print(f"Error processing SIFT for pair {pair_res.get('pair')}: {e}")
                        model_data['sift_visualizations'] = sift_visualizations
                 else:
                     print(f"Warning: Unexpected data type for model '{model_name}' results: {type(model_data)}")

            response_data['results'] = {
                "models": convert_numpy_types(model_results),
                "all_images_data_urls": image_urls
            }
        else:
             print(f"Warning: Unexpected final_results structure for job {job_id}")
             response_data['status'] = 'completed'
             response_data['message'] = 'Processing completed, but result structure is invalid.'
    
    return response_data

@app.get("/api/download_report/{job_id}")
def download_report(job_id: str):
    job = job_manager.get_job(job_id)
    if not job or job['status'] != 'completed':
        raise HTTPException(status_code=404, detail="Report not ready or job not found")

    final_results_payload = job.get('final_results', {})
    if not isinstance(final_results_payload, dict) or 'models' not in final_results_payload:
         raise HTTPException(status_code=500, detail="Invalid or missing model results for report generation")

    results_by_model = convert_numpy_types(final_results_payload['models'])

    class PDF(FPDF):
        def header(self):
            self.add_font('Sarabun', 'B', 'static/fonts/Sarabun-Bold.ttf', uni=True)
            self.set_font('Sarabun', 'B', 16)
            self.cell(0, 10, 'รายงานสรุปผลการเปรียบเทียบโมเดล FaceBack SimSnap', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.add_font('Sarabun', '', 'static/fonts/Sarabun-Regular.ttf', uni=True)
            self.set_font('Sarabun', '', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        def create_master_summary_table(self, all_results):
            self.ln(5)
            self.set_font('Sarabun', 'B', 14)
            self.cell(0, 10, 'ตารางสรุปเปรียบเทียบประสิทธิภาพโมเดล', 0, 1, 'L')
            self.set_font('Sarabun', 'B', 10)
            self.cell(45, 7, 'ชื่อโมเดล', 1, 0, 'C')
            self.cell(35, 7, 'ประเภท', 1, 0, 'C')
            self.cell(35, 7, 'เวลาประมวลผล (วินาที)', 1, 0, 'C')
            self.cell(35, 7, 'เฉลี่ย Cosine', 1, 0, 'C')
            self.cell(40, 7, 'เฉลี่ย Pearson', 1, 1, 'C')
            self.set_font('Sarabun', '', 9)
            for model_name, data in all_results.items():
                if isinstance(data, dict):
                    model_type = "ใบหน้า" if 'face_groups' in data else "ฉากหลัง"
                    self.cell(45, 7, model_name, 1, 0, 'L')
                    self.cell(35, 7, model_type, 1, 0, 'C')
                    self.cell(35, 7, f"{data.get('processing_time', 0):.2f}", 1, 0, 'C')
                    avg_scores = data.get('avg_face_scores', data.get('avg_background_scores', {}))
                    cos_avg = np.mean([s.get('cosine', 0) for s in avg_scores.values()]) if avg_scores else 0
                    pearson_avg = np.mean([s.get('pearson', 0) for s in avg_scores.values()]) if avg_scores else 0
                    self.cell(35, 7, f"{(cos_avg * 100):.2f}%", 1, 0, 'C')
                    self.cell(40, 7, f"{(pearson_avg * 100):.2f}%", 1, 1, 'C')
                else:
                     print(f"Skipping invalid data for model {model_name} in master summary")
            self.ln(10)

        def create_detailed_table(self, title, model_name, groups, avg_scores):
            self.ln(5)
            self.set_font('Sarabun', 'B', 12)
            self.cell(0, 10, f'{title} (โมเดล: {model_name})', 0, 1, 'L')
            self.set_font('Sarabun', 'B', 10)
            self.cell(40, 7, 'ประเภทกลุ่ม', 1, 0, 'C')
            self.cell(20, 7, 'กลุ่มที่', 1, 0, 'C')
            self.cell(20, 7, 'จำนวน', 1, 0, 'C')
            self.cell(50, 7, 'รายชื่อไฟล์', 1, 0, 'C')
            self.cell(30, 7, 'เฉลี่ย Cosine', 1, 0, 'C')
            self.cell(30, 7, 'เฉลี่ย Pearson', 1, 1, 'C')
            self.set_font('Sarabun', '', 9)
            if not groups or not isinstance(groups, dict):
                self.cell(190, 7, 'ไม่พบข้อมูลการจัดกลุ่ม', 1, 1, 'C')
                return
            avg_score_counter = 0
            for group_name, list_of_groups in groups.items():
                if not isinstance(list_of_groups, list):
                    print(f"Warning: Expected list for group '{group_name}', got {type(list_of_groups)}")
                    continue
                for i, members in enumerate(list_of_groups):
                    if not isinstance(members, list):
                         print(f"Warning: Expected list for members in group '{group_name}', got {type(members)}")
                         continue
                    avg_score_key = f"{group_name}_{avg_score_counter}"
                    avg = avg_scores.get(avg_score_key, {'cosine': 0, 'pearson': 0})
                    avg_cosine = avg.get('cosine', 0) if isinstance(avg, dict) else 0
                    avg_pearson = avg.get('pearson', 0) if isinstance(avg, dict) else 0

                    self.cell(40, 7, group_name, 1, 0, 'L')
                    group_index_str = str(i + 1)
                    if "ไม่ถูกจัดกลุ่ม" in group_name:
                         group_index_str = '-'
                    self.cell(20, 7, group_index_str, 1, 0, 'C')
                    self.cell(20, 7, str(len(members)), 1, 0, 'C')
                    current_y = self.get_y()
                    self.multi_cell(50, 7, ", ".join(members), 1, 'L')
                    new_y = self.get_y() 
                    y_after_multicell = new_y if new_y > current_y else current_y 
                    self.set_xy(10 + 40 + 20 + 20 + 50, current_y) 

                    self.cell(30, 7, f"{(avg_cosine * 100):.2f}%", 1, 0, 'C')
                    self.cell(30, 7, f"{(avg_pearson * 100):.2f}%", 1, 0, 'C') 
                    self.set_y(y_after_multicell) 
                    self.ln(7) 

                    avg_score_counter += 1

    pdf = PDF('P', 'mm', 'A4')
    pdf.add_font('Sarabun', '', 'static/fonts/Sarabun-Regular.ttf', uni=True)
    pdf.add_font('Sarabun', 'B', 'static/fonts/Sarabun-Bold.ttf', uni=True)
    pdf.add_page()

    first_model_results = next(iter(results_by_model.values()), None)
    if not first_model_results or not isinstance(first_model_results, dict):
        raise HTTPException(status_code=500, detail="Invalid result format found for report header")

    pdf.set_font('Sarabun', 'B', 14)
    pdf.cell(0, 10, 'ข้อมูลทั่วไป', 0, 1, 'L')
    pdf.set_font('Sarabun', '', 12)
    pdf.cell(60, 8, 'วัน-เวลาประมวลผล', 1)
    pdf.cell(0, 8, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 1, 1)
    pdf.cell(60, 8, 'จำนวนภาพ', 1)
    pdf.cell(0, 8, str(first_model_results.get('image_count', 'N/A')), 1, 1)
    pdf.create_master_summary_table(results_by_model)

    for model_name, results_data in results_by_model.items():
         if isinstance(results_data, dict):
            pdf.add_page()
            face_groups = results_data.get('face_groups')
            avg_face = results_data.get('avg_face_scores')
            bg_groups = results_data.get('background_groups')
            avg_bg = results_data.get('avg_background_scores')

            if face_groups and avg_face and isinstance(face_groups, dict) and isinstance(avg_face, dict):
                pdf.create_detailed_table('ตารางสรุปผลใบหน้า', model_name, face_groups, avg_face)
            if bg_groups and avg_bg and isinstance(bg_groups, dict) and isinstance(avg_bg, dict):
                pdf.create_detailed_table('ตารางสรุปผลฉากหลัง', model_name, bg_groups, avg_bg)
         else:
             print(f"Skipping invalid data for model {model_name} in PDF generation")

    try:
        pdf_output_bytes = pdf.output(dest='S')
    except Exception as e:
         print(f"Error during PDF output generation: {e}")
         raise HTTPException(status_code=500, detail=f"Error generating PDF: {e}")

    return Response(
        content=bytes(pdf_output_bytes),  
        media_type='application/pdf', 
        headers={'Content-Disposition': f'attachment; filename="report_{job_id}.pdf"'}
    )


def preload_all_models():
    print("--- 🚀 Preloading all models, please wait... ---")
    for model_name in analysis_engine.FACE_MODEL_NAMES:
        try: analysis_engine.load_specific_face_model(model_name)
        except Exception as e: print(f"Could not preload face model {model_name}: {e}")
    try: analysis_engine.load_specific_background_model('deeplab')
    except Exception as e: print(f"Could not preload background model deeplab: {e}")
    for model_name in analysis_engine.BACKGROUND_MODEL_NAMES:
        try: analysis_engine.load_specific_background_model(model_name)
        except Exception as e: print(f"Could not preload background model {model_name}: {e}")
    print("--- ✅ All models preloaded successfully! ---")

@app.on_event("startup")
def on_startup():
    print("--- FastAPI server is starting up... ---")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)