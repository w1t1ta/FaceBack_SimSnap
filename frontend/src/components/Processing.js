import React, { useState, useEffect, useRef } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import Results from './Results';

const API_URL = "http://localhost:8000";

const funFacts = [
    "เคล็ดลับ: ภาพที่มีแสงสว่างดีและเห็นใบหน้าชัดเจนให้ผลลัพธ์ที่ดีที่สุด",
    "รู้หรือไม่? โมเดล VGG-Face ถูกฝึกฝนด้วยภาพใบหน้ากว่า 2.6 ล้านภาพ",
    "การวิเคราะห์ฉากหลังช่วยให้รู้ว่าภาพถูกถ่ายในสถานที่เดียวกันหรือไม่",
    "โมเดล SFace เป็นหนึ่งในโมเดลวิเคราะห์ใบหน้าที่เร็วและทันสมัยที่สุด",
    "ระบบกำลังเปรียบเทียบคุณลักษณะหลายพันจุดบนใบหน้าและฉากหลัง",
    "รู้หรือไม่? FaceNet พัฒนาโดยทีมวิจัยของ Google ซึ่งสร้างมาตรฐานใหม่ให้วงการจดจำใบหน้า",
];

const Processing = () => {
    const { jobId } = useParams();
    const location = useLocation();
    const mode = new URLSearchParams(location.search).get('mode');

    const [status, setStatus] = useState({ message: 'กำลังเริ่มต้น...', detail: 'กรุณารอสักครู่...' });
    const [funFact, setFunFact] = useState(funFacts[0]);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const pollingIntervalRef = useRef(null);
    const factIntervalRef = useRef(null);

    const clearIntervals = () => {
        if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);
        if (factIntervalRef.current) clearInterval(factIntervalRef.current);
    };

    useEffect(() => {
        const startAnalysis = async () => {
            try {
                const response = await fetch(`${API_URL}/api/start_analysis/${jobId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: 'comparison' })
                });
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Server error when starting analysis.');
                }
            } catch (err) {
                setError(`ไม่สามารถเริ่มการวิเคราะห์ได้: ${err.message}`);
                clearIntervals();
            }
        };

        if (mode === 'comparison') {
            startAnalysis();
        }
    }, [jobId, mode]);

    useEffect(() => {
        const pollStatus = async () => {
            let response;
            try {
                response = await fetch(`${API_URL}/api/status/${jobId}`);

                if (!response.ok) {
                    throw new Error(`Network response was not ok, status: ${response.status}`);
                }

                const text = await response.text();

                if (!text) {
                    return;
                }

                const data = JSON.parse(text);

                if (data.status === 'completed' && data.results) {
                    setResults(data.results); 
                    clearIntervals();
                } else if (data.status === 'failed') {
                    setError(`เกิดข้อผิดพลาดจากการประมวลผล: ${data.message}`);
                    clearIntervals();
                } else {
                    const match = (data.message || '').match(/(.*?)(\s\(.+?\))?$/);
                    setStatus(match
                        ? { message: match[1] || 'กำลังประมวลผล...', detail: (match[2] || '').trim() }
                        : { message: data.message, detail: '' }
                    );
                }
            } catch (err) {
                console.error("A non-critical polling error occurred:", err);
            }
        };

        pollingIntervalRef.current = setInterval(pollStatus, 2000);
        factIntervalRef.current = setInterval(() => {
            setFunFact(funFacts[Math.floor(Math.random() * funFacts.length)]);
        }, 10000);

        pollStatus();

        return () => clearIntervals();
    }, [jobId]);

    if (error) {
        return (
             <div className="max-w-7xl mx-auto p-8"><div className="bg-white p-8 rounded-xl shadow-lg text-center">
                 <h1 className="text-2xl font-bold text-red-600">เกิดข้อผิดพลาด</h1>
                 <p className="text-gray-600 mt-2">{error}</p>
             </div></div>
        );
    }

    if (results) return <Results results={results} jobId={jobId} />;

    return (
        <div className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
            <div className="bg-white p-8 rounded-xl shadow-lg text-center mb-8">
                <div className="flex justify-center items-center mb-4">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                </div>
                <h1 className="text-2xl font-bold text-gray-800">{status.message}</h1>
                <p className="text-gray-500 mt-2 font-medium">{status.detail}</p>
            </div>
            <p className="text-center text-gray-600 mt-8 px-4">{funFact}</p>
        </div>
    );
};

export default Processing;