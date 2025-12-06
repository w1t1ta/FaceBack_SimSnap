import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

const API_URL = "http://localhost:8000";

const CustomDropdown = ({ label, options, selectedValue, onValueChange }) => {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef(null);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    const handleOptionClick = (value) => {
        onValueChange(value);
        setIsOpen(false);
    };

    return (
        <div className="relative" ref={dropdownRef}>
            <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
            <button
                type="button"
                className="w-full bg-white border border-gray-300 rounded-lg shadow-sm pl-4 pr-10 py-3 text-left cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500"
                onClick={() => setIsOpen(!isOpen)}
            >
                <span className="block truncate">{selectedValue}</span>
                <span className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fillRule="evenodd" d="M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                </span>
            </button>

            {isOpen && (
                <ul className="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                    {options.map((option) => (
                        <li
                            key={option}
                            className={`cursor-pointer select-none relative py-2 pl-4 pr-4 ${selectedValue === option ? 'text-white bg-blue-600' : 'text-gray-900 hover:bg-gray-100'}`}
                            onClick={() => handleOptionClick(option)}
                        >
                            <span className={`block truncate ${selectedValue === option ? 'font-semibold' : 'font-normal'}`}>
                                {option}
                            </span>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};


const ModelSelector = () => {
    const { jobId } = useParams();
    const navigate = useNavigate();
    const [faceModel, setFaceModel] = useState("VGG-Face");
    const [bgModel, setBgModel] = useState("resnet50");
    const [isLoading, setIsLoading] = useState(false);

    const faceModels = ["VGG-Face", "Facenet", "Facenet512", "ArcFace", "SFace"];
    const bgModels = ['resnet50', 'densenet121', 'vit', 'dino'];

    const handleStartAnalysis = async () => {
        setIsLoading(true);
        const payload = {
            mode: 'quick_analysis',
            face_model: faceModel,
            bg_model: bgModel
        };

        try {
            const response = await fetch(`${API_URL}/api/start_analysis/${jobId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Server error when starting analysis.');
            }
            
            navigate(`/processing/${jobId}?mode=quick_analysis`);

        } catch (error) {
            console.error('Error:', error);
            alert('ไม่สามารถเริ่มการวิเคราะห์ได้: ' + error.message);
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-8">
            <div className="bg-white p-8 rounded-xl shadow-lg text-center">
                <h1 className="text-3xl font-bold text-gray-800">โหมดเจาะจง</h1>
                <p className="text-gray-500 mt-2">เลือกโมเดลที่ต้องการใช้ในการวิเคราะห์</p>

                <div className="mt-8 grid md:grid-cols-2 gap-8 text-left">
                    <CustomDropdown
                        label="โมเดลวิเคราะห์ใบหน้า"
                        options={faceModels}
                        selectedValue={faceModel}
                        onValueChange={setFaceModel}
                    />
                    <CustomDropdown
                        label="โมเดลวิเคราะห์ฉากหลัง"
                        options={bgModels}
                        selectedValue={bgModel}
                        onValueChange={setBgModel}
                    />
                </div>

                <div className="mt-8">
                    <button onClick={handleStartAnalysis} disabled={isLoading} className="w-full md:w-1/2 bg-blue-600 text-white font-bold py-4 px-12 rounded-lg hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:-translate-y-1">
                        {isLoading ? 'กำลังเริ่มต้น...' : 'เริ่มการวิเคราะห์'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ModelSelector;